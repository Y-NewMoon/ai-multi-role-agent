"""四组消融（契约 4.10 / T8）：纯向量 / +关键词(RRF) / +rerank / +query 改写。

硬约束（§0.4 硬禁止 3）：**必须"真库 + 真嵌入"**。脚本在进入消融前校验
`TEST_DATABASE_URL` 与 sentence-transformers 可用、bge-m3 能加载；任一不满足即报错退出，
绝不用 FakeEmbeddings 顶替——哈希向量无语义，跑出来的消融表是废数据。

- 四组共用同一批 40 条 `retrieval_lore` 查询（否则不可比）；
- 逐条计时，统计注入 token 与 query 改写触发率；
- 输出 Markdown 表 + 保存 `eval/report/ablation-YYYYmmdd-HHMMSS.md`。

用法：
    uv run python scripts/ablate_lore.py --agent 1
"""

from __future__ import annotations

import os

# 必须早于任何把 huggingface_hub 拉进来的导入（huggingface_hub.constants 在 import 时读该变量）。
# 与 src/app/llm/embedding.py 同一约定：bge-m3 / reranker 权重来自本地 HF cache，默认离线加载，
# 避免在线校验/大文件下载导致的挂起与"Unrecognized processing class"。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

# Windows 下 psycopg 异步需 SelectorEventLoop（照抄 eval/run_eval.py）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import lore_eval  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db.base import create_engine, create_session_factory  # noqa: E402
from app.lore.retriever import (  # noqa: E402
    REWRITE_CACHE_PATH,
    BgeReranker,
    LlmRewriter,
    Retriever,
    needs_rewrite,
)

OUT_DIR = ROOT / "eval" / "report"
SUITE_PATH = ROOT / "eval" / "suites" / "retrieval_lore.jsonl"

# 组别 → 配置说明（对齐契约 4.10 §8.3 表）
GROUP_LABELS = {
    "A": "纯向量",
    "B": "向量 + 关键词（RRF）",
    "C": "B + rerank",
    "D": "C + query 改写",
}


class AblationRetriever(Retriever):
    """真实 Retriever + keyword 路开关（其余代码路径逐字复用真实现）。"""

    def __init__(self, *, keyword_enabled: bool = True, **kwargs) -> None:
        self._keyword_enabled = keyword_enabled
        super().__init__(**kwargs)

    async def _route_keyword(self, db, agent_id, query):  # noqa: ANN001
        if not self._keyword_enabled:
            return []
        return await super()._route_keyword(db, agent_id, query)


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in SUITE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def _measure_group(name: str, retriever: Retriever, cases: list[dict]) -> dict:
    ranked: dict[str, list[str]] = {}
    tokens: list[int] = []
    elapsed: list[float] = []
    rerank_any = False
    rewrite_used = 0
    rewrite_attempts = 0
    for case in cases:
        t0 = time.perf_counter()
        result = await retriever.retrieve(agent_id=case["agent_id"], query=case["query"])
        elapsed.append((time.perf_counter() - t0) * 1000.0)
        ranked[case["id"]] = [ref for ref, _ in result.reranked]
        tokens.append(result.lore_tokens)
        rerank_any = rerank_any or bool(result.rerank)
        rewrite_attempts += int(needs_rewrite(case["query"]))
        rewrite_used += int(bool(result.rewrite_used))
    metrics = lore_eval.measure(cases, ranked, rerank=rerank_any)["metrics"]
    n = len(cases) or 1
    return {
        "name": name,
        "label": GROUP_LABELS[name],
        "hit@1": metrics["hit@1"],
        "hit@3": metrics["hit@3"],
        "mrr": metrics["mrr"],
        "ndcg@10": metrics["ndcg@10"],
        "avg_tokens": sum(tokens) / n,
        "avg_ms": sum(elapsed) / n,
        "rerank": rerank_any,
        "rewrite_attempts": rewrite_attempts,
        "rewrite_used": rewrite_used,
    }


def _fmt(value, ndigits: int = 4) -> str:
    if value is None:
        return "无"
    return f"{value:.{ndigits}f}"


def render_markdown(rows: list[dict], meta: dict) -> str:
    lines = [
        "# V2.1 世界书检索消融报告（T8）",
        "",
        f"- 生成时间：{meta['timestamp']}",
        f"- git commit：`{meta['git_sha']}`",
        f"- 存储：`{meta['storage']}`；嵌入：`{meta['embedding']}`",
        f"- 语料：`eval/fixtures/lore_long/` 2 篇；导入 chunk 数 {meta['chunks']}；agent_id={meta['agent']}",
        f"- 用例：`eval/suites/retrieval_lore.jsonl` {meta['queries']} 条（四组同一批）",
        f"- rerank 模型加载：{'成功' if meta['rerank_available'] else '失败（C 组退化为 B 组的 RRF 顺序）'}",
        "",
        "| 组 | 配置 | hit@1 | hit@3 | MRR | nDCG@10 | 平均注入 token | 平均耗时 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {label} | {h1} | {h3} | {mrr} | {ndcg} | {tok} | {ms} |".format(
                name=row["name"],
                label=row["label"],
                h1=_fmt(row["hit@1"]),
                h3=_fmt(row["hit@3"]),
                mrr=_fmt(row["mrr"]),
                ndcg=_fmt(row["ndcg@10"]),
                tok=_fmt(row["avg_tokens"], 1),
                ms=_fmt(row["avg_ms"], 1),
            )
        )
    lines += [
        "",
        "## 补充",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| rerank 模型是否加载成功 | {'是' if meta['rerank_available'] else '否'} |",
        f"| query 改写（D 组）触发 / 成功 | {meta['rewrite_attempts']} / {meta['rewrite_used']}"
        f"（成功率 {_fmt(meta['rewrite_used'] / max(1, meta['rewrite_attempts']) * 100, 1)}%，"
        f"超时回退算未成功） |",
        "| 每组实际 rerank 生效 | "
        + "，".join(f"{r['name']}={r['rerank']}" for r in rows)
        + " |",
        "",
        "> 生成方式：`uv run python scripts/ablate_lore.py --agent %d`（真库 + 真 bge-m3，禁用 FakeEmbeddings）。"
        % meta["agent"],
        "",
    ]
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.test_database_url:
        print("[error] TEST_DATABASE_URL 为空：消融必须用真库", file=sys.stderr)
        return 2

    try:
        import sentence_transformers  # noqa: F401
    except Exception:  # noqa: BLE001
        print(
            "[error] 无 sentence-transformers：无法算真向量（禁止 FakeEmbeddings 顶替）",
            file=sys.stderr,
        )
        return 2

    from app.llm.factory import create_embedding_provider, create_llm_provider

    try:
        embed = create_embedding_provider(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] 真嵌入器加载失败：{exc}", file=sys.stderr)
        return 2

    engine = create_engine(settings.test_sqlalchemy_url)
    session_factory = create_session_factory(engine)
    cases = load_cases()
    try:
        chunks = await lore_eval.import_fixtures(session_factory, embed, agent_id=args.agent)
        active = await lore_eval.count_active_chunks(session_factory, agent_id=args.agent)
        if chunks == 0 or active == 0:
            print("[error] 语料导入后 chunk 数为 0，消融无法进行", file=sys.stderr)
            return 2
        print(f"[info] 导入 {chunks} chunks（active={active}），用例 {len(cases)} 条", flush=True)

        reranker = BgeReranker()
        rerank_available = reranker.available
        if not rerank_available:
            print("[warn] bge-reranker-v2-m3 加载失败：C/D 组将退化为 RRF 顺序", flush=True)

        llm = None
        if settings.llm_api_key:
            try:
                llm = create_llm_provider(settings)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] LLM provider 构造失败，D 组不可用：{exc}", flush=True)
        else:
            print("[warn] LLM_API_KEY 为空，D 组不可用", flush=True)

        def make(**kw) -> AblationRetriever:
            return AblationRetriever(session_factory=session_factory, embedder=embed, **kw)

        plan = [
            ("A", make(keyword_enabled=False, reranker=None, rewrite_enabled=False)),
            ("B", make(keyword_enabled=True, reranker=None, rewrite_enabled=False)),
            ("C", make(keyword_enabled=True, reranker=reranker, rewrite_enabled=False)),
        ]
        if llm is not None:
            plan.append(
                (
                    "D",
                    make(
                        keyword_enabled=True,
                        reranker=reranker,
                        rewriter=LlmRewriter(llm, cache_path=REWRITE_CACHE_PATH),
                        rewrite_enabled=True,
                    ),
                )
            )
        else:
            print("[warn] 跳过 D 组（无可用 LLM）", flush=True)

        rows: list[dict] = []
        for name, retriever in plan:
            print(f"[info] 跑 {name} 组（{GROUP_LABELS[name]}）…", flush=True)
            rows.append(await _measure_group(name, retriever, cases))
    finally:
        await engine.dispose()

    git_sha = _git_sha()
    group_d = next((r for r in rows if r["name"] == "D"), None)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "storage": f"postgres(test_database_url, agent_id={args.agent})",
        "embedding": f"local:{settings.embedding_model}({settings.embedding_dims}d)",
        "chunks": chunks,
        "queries": len(cases),
        "agent": args.agent,
        "rerank_available": rerank_available,
        "rewrite_attempts": group_d["rewrite_attempts"] if group_d else 0,
        "rewrite_used": group_d["rewrite_used"] if group_d else 0,
    }
    markdown = render_markdown(rows, meta)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = OUT_DIR / f"ablation-{stamp}.md"
    out_path.write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"报告：{out_path}")
    return 0


def _git_sha() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(ROOT),
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="V2.1 世界书检索四组消融")
    ap.add_argument("--agent", type=int, default=1, help="评测角色 id，默认 1")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
