"""三路召回 + RRF + rerank + 预算（契约 4.4~4.7）。

- 三路：always-on（核心设定）/ keyword（jieba 分词 → tsv @@ to_tsquery）/ vector（余弦距离）；
- 融合：`score(d) = Σ_{r} 1 / (60 + rank_r(d))`，rank 从 1 起，并列按 `source_ref` 升序；
- rerank：`BgeReranker` 懒加载，失败降级 `NullReranker`（`rerank=false`，不抛异常）；
- 预算：always-on ≤ 8%、lore 合计 ≤ 20%，先占预算者优先。

查询改写（4.6）通过注入的 `rewriter` 实现；默认关闭（CI 与单测用 false）。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from langchain_core.embeddings import Embeddings
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import LoreChunk, LoreDoc
from app.lore.keywords import cut

# RRF 常数（契约 4.5）
RRF_K = 60
# 单路 / 融合后进入 rerank 的条数上限（契约 4.4/4.5）
TOP_K = 20
# 代词（契约 4.6）
PRONOUNS = ("这", "那", "她", "他", "它", "其")
REWRITE_MIN_LEN = 8
REWRITE_TIMEOUT = 1.5
# 改写缓存落盘路径（契约 4.6 / 复跑一致性）：仅在显式传入 `cache_path` 时启用。
# 生产默认**不落盘**（避免把用户 query 写进仓库文件）；消融脚本显式传它，
# 使含 LLM 的 D 组位级可复现（见 §4.10 / §8.3）。
REWRITE_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "eval" / "calibration" / "rewrites.json"
)


@dataclass
class LoreHit:
    """一条最终注入的 lore chunk（对齐 state.recalled_lore 元素）。"""

    source_ref: str
    header_path: str
    content: str
    token_count: int
    score: float
    routes: tuple[str, ...]

    def as_recalled(self) -> dict:
        """转成 `recalled_lore` 的字典形态（契约 4.8）。"""
        return {
            "source_ref": self.source_ref,
            "header_path": self.header_path,
            "content": self.content,
            "score": self.score,
            "routes": list(self.routes),
        }


@dataclass
class RetrievalResult:
    """一次检索的完整中间结果（供 preview 与消融归因，契约 4.9）。"""

    query: str
    rewritten: str
    rewrite_used: bool
    routes: dict[str, list[str]]
    fused: list[tuple[str, float]]
    reranked: list[tuple[str, float]]
    rerank: bool
    injected: list[LoreHit] = field(default_factory=list)
    lore_tokens: int = 0
    lore_cap: int = 0


class Reranker(Protocol):
    def rank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        """返回 [(原下标, 分数)]，按分数降序。"""
        ...


class NullReranker:
    """无 sentence-transformers 时的降级：不做重排，`available=False`。"""

    available = False

    def rank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        return list(enumerate([0.0] * len(docs)))[:top_n]


class BgeReranker:
    """CrossEncoder `BAAI/bge-reranker-v2-m3` 懒加载；加载失败 `available=False`。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._model = None
        self._loaded = False

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            # 与嵌入一致：权重来自本地 HF cache，默认离线加载
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        except Exception:  # noqa: BLE001 - 加载失败一律降级，不打断对话
            self._model = None

    def rank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        self._ensure()
        if self._model is None:
            return NullReranker().rank(query, docs, top_n)
        scores = self._model.predict([(query, doc) for doc in docs])
        order = sorted(
            range(len(docs)), key=lambda i: (-float(scores[i]), i)
        )
        return [(i, float(scores[i])) for i in order[:top_n]]


class Rewriter(Protocol):
    async def rewrite(self, query: str, history: list[str]) -> str: ...


class LlmRewriter:
    """用 LLM 把口语化指代改写为一行检索 query（契约 4.6）。

    - `temperature=0` 经 `llm._client` 显式传参（照 `eval/judge.py` 既有写法，
      **不修改 `src/app/llm/**`**）；测试注入的 fake llm 无底层客户端时回落 `generate`；
    - 改写结果按 `query` 缓存；默认**仅内存**（不落盘），消融脚本显式传
      `cache_path=REWRITE_CACHE_PATH` 以落盘复用——含 LLM 的 D 组据此位级可复现（见 §4.10 / §8.3）。
    """

    def __init__(self, llm, *, cache_path: str | Path | None = None) -> None:  # noqa: ANN001
        self._llm = llm
        # cache_path=None → 仅内存缓存（生产默认不落盘，避免把用户 query 写进仓库文件）；
        # 消融脚本显式传 REWRITE_CACHE_PATH，以取得 D 组位级复跑所需的落盘缓存。
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[str, str] = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        if self._cache_path is None:
            return {}
        try:
            if self._cache_path.exists():
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
        except Exception:  # noqa: BLE001 - 缓存损坏不影响改写
            pass
        return {}

    def _store(self, query: str, rewritten: str) -> None:
        self._cache[query] = rewritten
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 - 落盘失败只影响复跑一致性，不打断改写
            pass

    async def _call_llm(self, messages: list[dict]) -> str:
        """以 temperature=0 调用改写模型（照 `eval/judge.py:147-161` 的写法）。"""
        client = getattr(self._llm, "_client", None)
        model = getattr(self._llm, "_model", None)
        if client is not None and model is not None and hasattr(client, "chat"):
            resp = await client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
            return resp.choices[0].message.content or ""
        return await self._llm.generate(messages)

    async def rewrite(self, query: str, history: list[str]) -> str:
        cached = self._cache.get(query)
        if cached:
            return cached
        # 历史里若仍含本轮 query（调用方兜底），从「最近对话」剔除，避免 prompt 重复当前输入
        prior = [h for h in history if h != query]
        recent = "\n".join(prior[-2:])
        messages = [
            {"role": "system", "content": "把用户输入改写成一行用于检索的查询词，只输出查询词。"},
            {"role": "user", "content": f"最近对话：\n{recent}\n\n当前输入：{query}"},
        ]
        rewritten = await self._call_llm(messages)
        self._store(query, rewritten)
        return rewritten


def needs_rewrite(query: str, history: list[str] | None = None) -> bool:
    """是否满足改写触发条件（契约 4.6）：含代词、长度 < 8 字，或上一轮有未消解指代。

    第三条依赖 `history`：`retrieve_lore._history` 传入的是**本轮之前的用户消息**
    （已排除当前用户消息），其末元素即"上一轮用户消息"——末轮含代词即视为"未消解指代"。
    """
    if any(word in query for word in PRONOUNS) or len(query) < REWRITE_MIN_LEN:
        return True
    return bool(history) and any(word in history[-1] for word in PRONOUNS)


def _keyword_terms(query: str) -> list[str]:
    """查询侧 jieba 分词：丢弃 < 2 字与标点（契约 4.4）。"""
    terms: list[str] = []
    for word in cut(query):
        cleaned = re.sub(r"\W", "", word)
        if len(cleaned) >= 2 and cleaned not in terms:
            terms.append(cleaned)
    return terms


class Retriever:
    """三路召回 → RRF → rerank → 预算截断。"""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embeddings | None = None,
        reranker: Reranker | None = None,
        rewriter: Rewriter | None = None,
        rewrite_enabled: bool = False,
        budget_tokens: int = 8000,
        lore_ratio: float = 0.20,
        always_on_ratio: float = 0.08,
        top_k: int = TOP_K,
        rewrite_timeout: float = REWRITE_TIMEOUT,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._reranker = reranker
        self._rewriter = rewriter
        self._rewrite_enabled = rewrite_enabled
        self._budget_tokens = budget_tokens
        self._lore_ratio = lore_ratio
        self._always_on_ratio = always_on_ratio
        self._top_k = top_k
        self._rewrite_timeout = rewrite_timeout

    async def retrieve(
        self, *, agent_id: int, query: str, history: list[str] | None = None
    ) -> RetrievalResult:
        effective, used = await self._maybe_rewrite(query, history or [])

        async with self._session_factory() as db:
            routes = await self._recall(db, agent_id=agent_id, query=effective)

        fused = _fuse(routes)[: self._top_k]
        refs = {ref for refs in routes.values() for ref in refs} | {ref for ref, _ in fused}
        rows = await self._load_rows(agent_id=agent_id, refs=refs)

        reranked, rerank_ok = await self._rerank(effective, fused, rows)
        lore_cap = int(self._budget_tokens * self._lore_ratio)
        always_cap = int(self._budget_tokens * self._always_on_ratio)
        injected, total = self._fit_budget(
            routes=routes,
            reranked=reranked,
            rows=rows,
            always_cap=always_cap,
            lore_cap=lore_cap,
        )
        return RetrievalResult(
            query=query,
            rewritten=effective,
            rewrite_used=used,
            routes=routes,
            fused=list(fused),
            reranked=list(reranked),
            rerank=rerank_ok,
            injected=injected,
            lore_tokens=total,
            lore_cap=lore_cap,
        )

    # ---- 三路召回 ----

    async def _recall(
        self, db: AsyncSession, *, agent_id: int, query: str
    ) -> dict[str, list[str]]:
        return {
            "always_on": await self._route_always_on(db, agent_id),
            "keyword": await self._route_keyword(db, agent_id, query),
            "vector": await self._route_vector(db, agent_id, query),
        }

    async def _route_always_on(self, db: AsyncSession, agent_id: int) -> list[str]:
        stmt = (
            select(LoreChunk.source_ref)
            .join(LoreDoc, LoreDoc.id == LoreChunk.doc_id)
            .where(
                LoreChunk.agent_id == agent_id,
                LoreChunk.status == "active",
                LoreDoc.status == "active",
                LoreDoc.always_on.is_(True),
            )
            .order_by(LoreChunk.source_ref.asc())
        )
        return list((await db.execute(stmt)).scalars().all())

    async def _route_keyword(
        self, db: AsyncSession, agent_id: int, query: str
    ) -> list[str]:
        terms = _keyword_terms(query)
        if not terms:
            return []  # 查询无可用词 → 跳过 keyword 路，不报错
        tsquery = func.to_tsquery("simple", " | ".join(terms))
        stmt = (
            select(LoreChunk.source_ref)
            .where(
                LoreChunk.agent_id == agent_id,
                LoreChunk.status == "active",
                LoreChunk.tsv.op("@@")(tsquery),
            )
            .order_by(
                func.ts_rank_cd(LoreChunk.tsv, tsquery).desc(),
                LoreChunk.source_ref.asc(),
            )
            .limit(self._top_k)
        )
        return list((await db.execute(stmt)).scalars().all())

    async def _route_vector(
        self, db: AsyncSession, agent_id: int, query: str
    ) -> list[str]:
        if self._embedder is None:
            return []
        vector = await asyncio.to_thread(self._embedder.embed_query, query)
        stmt = (
            select(LoreChunk.source_ref)
            .where(
                LoreChunk.agent_id == agent_id,
                LoreChunk.status == "active",
                LoreChunk.embedding.is_not(None),
            )
            .order_by(LoreChunk.embedding.cosine_distance(vector))
            .limit(self._top_k)
        )
        return list((await db.execute(stmt)).scalars().all())

    # ---- 融合与重排 ----

    async def _load_rows(
        self, *, agent_id: int, refs: set[str]
    ) -> dict[str, LoreChunk]:
        if not refs:
            return {}
        async with self._session_factory() as db:
            rows = (
                await db.execute(
                    select(LoreChunk).where(
                        LoreChunk.agent_id == agent_id,
                        LoreChunk.source_ref.in_(list(refs)),
                        LoreChunk.status == "active",
                    )
                )
            ).scalars()
            return {row.source_ref: row for row in rows}

    async def _rerank(
        self,
        query: str,
        fused: list[tuple[str, float]],
        rows: dict[str, LoreChunk],
    ) -> tuple[list[tuple[str, float]], bool]:
        docs = [(ref, rows[ref].content) for ref, _ in fused if ref in rows]
        if not docs or self._reranker is None:
            return list(fused), False
        if not getattr(self._reranker, "available", True):
            return list(fused), False  # 降级：保持 RRF 顺序
        order = await asyncio.to_thread(
            self._reranker.rank, query, [d for _, d in docs], len(docs)
        )
        return [(docs[idx][0], score) for idx, score in order], True

    def _fit_budget(
        self,
        *,
        routes: dict[str, list[str]],
        reranked: list[tuple[str, float]],
        rows: dict[str, LoreChunk],
        always_cap: int,
        lore_cap: int,
    ) -> tuple[list[LoreHit], int]:
        injected: list[LoreHit] = []
        seen: set[str] = set()
        total = 0

        # always-on：按 source_ref 升序先占预算（超 8% 时按 source_ref 倒序丢）
        for ref in routes["always_on"]:
            row = rows.get(ref)
            if row is None or ref in seen:
                continue
            if injected and total + row.token_count > always_cap:
                break
            injected.append(
                LoreHit(
                    source_ref=ref,
                    header_path=row.header_path,
                    content=row.content,
                    token_count=row.token_count,
                    score=1.0,  # 常驻标记（无 rerank 分）
                    routes=("always_on",),
                )
            )
            seen.add(ref)
            total += row.token_count

        # 检索 lore：按 rerank 分降序填剩余预算（超 20% 的丢）
        for ref, score in reranked:
            row = rows.get(ref)
            if row is None or ref in seen:
                continue
            if total + row.token_count > lore_cap:
                continue
            injected.append(
                LoreHit(
                    source_ref=ref,
                    header_path=row.header_path,
                    content=row.content,
                    token_count=row.token_count,
                    score=score,
                    routes=tuple(_routes_of(ref, routes)),
                )
            )
            seen.add(ref)
            total += row.token_count
        return injected, total

    async def _maybe_rewrite(self, query: str, history: list[str]) -> tuple[str, bool]:
        if (
            not self._rewrite_enabled
            or self._rewriter is None
            or not needs_rewrite(query, history)
        ):
            return query, False
        try:
            rewritten = await asyncio.wait_for(
                self._rewriter.rewrite(query, history), self._rewrite_timeout
            )
        except Exception:  # noqa: BLE001 - 超时/异常一律回退原 query（契约 4.6）
            return query, False
        rewritten = (rewritten or "").strip().replace("\n", " ")[:50]
        return (rewritten, True) if rewritten else (query, False)


def _routes_of(ref: str, routes: dict[str, list[str]]) -> list[str]:
    return [name for name, refs in routes.items() if ref in refs]


def _fuse(routes: dict[str, list[str]]) -> list[tuple[str, float]]:
    """RRF 融合：`Σ 1/(60 + rank)`；并列按 source_ref 升序（契约 4.5）。"""
    scores: dict[str, float] = {}
    for refs in routes.values():
        for rank, ref in enumerate(refs, start=1):
            scores[ref] = scores.get(ref, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
