"""M5-P0：检索分段计时（P4）。

覆盖四条：
1. `stage_ms` 9 个键齐全，值为 `int` 或 `None`；
2. 自洽性：`sum(非 None 段) <= total_ms * 1.15`（只允许取整/计时误差）；
3. 不改行为：计时前后注入的 `source_ref` 列表逐字相同（固定 fixture 下快照）；
4. 异常路径：内部抛错时 `total_ms` 仍被记录，其余段为 `None`。

真库用例需 `alembic upgrade head`；库不可用时由 `real_session_factory` 自动跳过。
"""

import asyncio
import hashlib
import time

import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy import delete

from app.db.models import Agent, LoreChunk, LoreDoc
from app.lore.indexer import Indexer
from app.lore.retriever import STAGE_KEYS, NullReranker, Retriever

_HIST = "祖父在民国年间盘下这间铺面，取名一盏春茶馆，从此以茶为生。" * 3
_SIGN = "招牌是一盏春茶馆的茉莉花茶，客人来了先奉一盏。" * 3
_SYSTEM = "系统设定：茶馆只做熟客生意，每日只卖三锅茶。" * 3
_QUERY = "祖父开的茶馆叫什么名字"


class _FakeEmbedder(Embeddings):
    """确定性哈希向量（与 tests/test_lore_retriever.py 同口径）。"""

    def __init__(self, dims: int = 1024) -> None:
        self._dims = dims

    def _vec(self, one: str) -> list[float]:
        digest = hashlib.sha256(one.encode("utf-8")).digest()
        raw = (digest * ((self._dims // len(digest)) + 1))[: self._dims]
        return [b / 255.0 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, one: str) -> list[float]:
        return self._vec(one)


def _retriever(session_factory) -> Retriever:
    return Retriever(
        session_factory=session_factory,
        embedder=_FakeEmbedder(),
        reranker=NullReranker(),
    )


async def _seed(session_factory) -> int:
    async with session_factory() as db:
        agent = Agent(name="m5p0-timings", persona_prompt="测试人设")
        db.add(agent)
        await db.flush()
        agent_id = agent.id
        await db.commit()
    indexer = Indexer(session_factory=session_factory, embedder=_FakeEmbedder())
    await indexer.upsert_doc(
        agent_id=agent_id,
        slug="tea_house",
        title="一盏春茶馆",
        content=f"# 一盏春\n\n## 历史\n{_HIST}\n\n## 招牌\n{_SIGN}\n",
    )
    await indexer.upsert_doc(
        agent_id=agent_id,
        slug="rules",
        title="茶楼规矩",
        content=f"# 规矩\n\n{_SYSTEM}\n",
        always_on=True,
    )
    return agent_id


async def _cleanup(session_factory, agent_id: int) -> None:
    async with session_factory() as db:
        await db.execute(delete(LoreChunk).where(LoreChunk.agent_id == agent_id))
        await db.execute(delete(LoreDoc).where(LoreDoc.agent_id == agent_id))
        await db.execute(delete(Agent).where(Agent.id == agent_id))
        await db.commit()


@pytest.fixture
async def seeded(real_session_factory):
    agent_id = await _seed(real_session_factory)
    try:
        yield real_session_factory, agent_id
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_stage_keys_complete(seeded) -> None:
    """1）13 段齐全（M5 手段①新增 4 个 queue/compute 细分），值为 int 或 None。"""
    session_factory, agent_id = seeded
    result = await _retriever(session_factory).retrieve(agent_id=agent_id, query=_QUERY)
    assert list(result.stage_ms) == list(STAGE_KEYS)
    assert len(result.stage_ms) == 13
    for name, value in result.stage_ms.items():
        assert isinstance(value, (int, type(None))), name
    # 未启用改写记 0；NullReranker 配了 reranker → 记真实耗时（P6）、attempted=true
    assert result.stage_ms["rewrite_ms"] == 0
    assert result.rewrite_attempted is False
    assert result.rerank_attempted is True
    assert isinstance(result.stage_ms["total_ms"], int)


async def test_stage_sum_within_total(seeded) -> None:
    """2）分段之和不得超过总时长的 1.15 倍。"""
    session_factory, agent_id = seeded
    result = await _retriever(session_factory).retrieve(agent_id=agent_id, query=_QUERY)
    total = result.stage_ms["total_ms"]
    assert total is not None and total > 0
    parts = sum(v for v in result.stage_ms.values() if v is not None) - total
    assert parts <= total * 1.15, result.stage_ms


async def test_timing_does_not_change_injected(seeded) -> None:
    """3）不改行为：注入顺序与内容由同一 fixture 快照锁定。

    人为改动算法（如翻转 `_fuse` 排序、改 top_k/预算）会让该断言失败（见 V6 说明）。
    """
    session_factory, agent_id = seeded
    result = await _retriever(session_factory).retrieve(agent_id=agent_id, query=_QUERY)
    assert [hit.source_ref for hit in result.injected] == [
        "rules:01",
        "tea_house:01",
        "tea_house:02",
    ]
    assert result.routes["always_on"] == ["rules:01"]
    assert result.lore_tokens == sum(hit.token_count for hit in result.injected)
    assert result.rerank is False


async def test_exception_path_records_total(monkeypatch) -> None:
    """4）内部抛错：异常照旧抛出，`last_stage_ms.total_ms` 有值、其余段为 None。"""

    async def _boom(query, history):  # noqa: ANN001 - 实例级桩（不绑定 self）
        raise RuntimeError("boom")

    retriever = Retriever(session_factory=object(), reranker=NullReranker())
    monkeypatch.setattr(retriever, "_maybe_rewrite", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        await retriever.retrieve(agent_id=1, query=_QUERY)
    stage = retriever.last_stage_ms
    assert isinstance(stage["total_ms"], int)
    assert [stage[name] for name in STAGE_KEYS if name != "total_ms"] == [None] * (
        len(STAGE_KEYS) - 1
    )


async def test_rewrite_失败时仍记录真实耗时(seeded) -> None:
    """P6：改写被触发但返回空串 → used=False，但 rewrite_ms 记真实耗时。"""
    session_factory, agent_id = seeded

    class _SlowEmptyRewriter:
        async def rewrite(self, query, history):  # noqa: ANN001
            await asyncio.sleep(0.05)
            return ""

    retriever = Retriever(
        session_factory=session_factory,
        embedder=_FakeEmbedder(),
        reranker=NullReranker(),
        rewriter=_SlowEmptyRewriter(),
        rewrite_enabled=True,
    )
    result = await retriever.retrieve(agent_id=agent_id, query="这茶")
    assert result.rewrite_attempted is True
    assert result.rewrite_used is False
    assert result.stage_ms["rewrite_ms"] >= 40


async def test_rerank_不可用时仍记录耗时(seeded) -> None:
    """P6：reranker 存在但可用性探测失败（耗时） → rerank=False，rerank_ms 记真实耗时。"""
    session_factory, agent_id = seeded

    class _SlowUnavailableReranker:
        @property
        def available(self) -> bool:
            time.sleep(0.05)  # 模拟“加载失败前的探测/加载耗时”
            return False

        def rank(self, query, docs, top_n):  # noqa: ANN001
            return list(enumerate([0.0] * len(docs)))[:top_n]

    retriever = Retriever(
        session_factory=session_factory,
        embedder=_FakeEmbedder(),
        reranker=_SlowUnavailableReranker(),
    )
    result = await retriever.retrieve(agent_id=agent_id, query=_QUERY)
    assert result.rerank_attempted is True
    assert result.rerank is False
    assert result.stage_ms["rerank_ms"] >= 40


# ===== P12：rerank doc 侧 token 截断（默认 None 行为不变） =====


class _FakeTokenizer:
    """按空格切词；`decode` 还原为 `t{i}`，便于断言截断后的 doc 内容。"""

    def __init__(self) -> None:
        self.calls: list[int | None] = []

    def __call__(
        self, text: str, *, truncation=False, max_length=None, add_special_tokens=False
    ):  # noqa: ANN001 - 模仿 HF tokenizer 的关键参数
        ids = list(range(len(text.split())))
        self.calls.append(max_length)
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}

    def decode(self, ids, *, skip_special_tokens=True):  # noqa: ANN001
        return " ".join(f"t{i}" for i in ids)


class _FakeCrossEncoder:
    """记录实际进入 predict 的 (query, doc) 对；不加载真实模型。"""

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs):  # noqa: ANN001
        self.seen = list(pairs)
        return [1.0 - i / 10.0 for i in range(len(pairs))]


def test_rerank_默认不截断_输入逐字一致() -> None:
    """P12：`max_tokens=None`（默认）不调 tokenizer，doc 原样进入 predict。"""
    from app.lore.retriever import BgeReranker

    reranker = BgeReranker()
    fake = _FakeCrossEncoder()
    reranker._model = fake  # noqa: SLF001 - 绕过真实模型加载
    reranker._loaded = True  # noqa: SLF001
    docs = ["一二 三四 五六 七八", "九十"]
    assert reranker.available is True
    reranker.rank("q", docs, top_n=2)
    assert fake.seen == [("q", docs[0]), ("q", docs[1])]
    assert fake.tokenizer.calls == []


def test_rerank_设置max_tokens时只截断doc侧() -> None:
    """P12：设置 `max_tokens` 后只截断 doc，query 保持不变；返回仍是原下标。"""
    from app.lore.retriever import BgeReranker

    reranker = BgeReranker(max_tokens=2)
    fake = _FakeCrossEncoder()
    reranker._model = fake  # noqa: SLF001
    reranker._loaded = True  # noqa: SLF001
    docs = ["a b c d", "e f g h i"]
    out = reranker.rank("q", docs, top_n=2)
    assert fake.seen == [("q", "t0 t1"), ("q", "t0 t1")]
    assert fake.tokenizer.calls == [2, 2]
    assert [idx for idx, _ in out] == [0, 1]
