"""单请求检索探针（M5-P0 P3）：真实 embedding + 真实 rerank 的分段延迟测量。

与 `scripts/load_test.py`（20 并发、强制 fake LLM）定位相反：
本脚本**单请求串行**，用于看清单次 `Retriever.retrieve()` 的时间花在哪一段，
避免并发争 CPU 把分段数据糊掉。

用法：
    uv run python scripts/probe_lore_latency.py --agent-id 1 \
      --runs 20 --warmup 3 --out eval/report/probe-lore-20261101.json

约束：
- `EMBEDDING_PROVIDER=fake` 时 **exit 2**（fake 向量不能产出“真实链路”结论）；
- 开启改写（`LORE_QUERY_REWRITE=true`）时要求 `LLM_PROVIDER=fake`，否则拒绝（0 真实 token）；
- 预热样本不进统计，只保留在 `warmup_samples` 供“首次 vs 预热后”对比；
- 产物 JSON 带 git/嵌入/重排/查询等可复现字段；控制台打印分段 × p50/p95/max 表。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUT_DIR = ROOT / "eval" / "report"
SCHEMA_VERSION = "1.0"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_QUERIES = (
    "祖父开的茶馆叫什么名字",
    "那家茶楼的招牌茶是什么",
    "一盏春茶馆有哪些规矩",
)


# ===== 通用 =====


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - 非 git 环境也要能出报告
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def percentile(values: list[float], p: float) -> float | None:
    """nearest-rank 分位数；空列表返回 None（与 load_test/trace_report 口径一致）。"""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[min(k, len(ordered) - 1)]


def _int_pct(values: list[float], p: float) -> int | None:
    value = percentile(values, p)
    return None if value is None else int(round(value))


# ===== 采样与统计 =====


async def probe_once(retriever, *, agent_id: int, query: str) -> dict:
    """跑一次 `retrieve()`；失败也保留 `last_stage_ms`（异常路径只含已执行段）。"""
    try:
        result = await retriever.retrieve(agent_id=agent_id, query=query)
    except Exception as exc:  # noqa: BLE001 - 单次失败计数，不中断整轮探针
        return {
            "query": query,
            "error": f"{type(exc).__name__}: {exc}",
            "stage_ms": dict(retriever.last_stage_ms),
        }
    return {
        "query": query,
        "stage_ms": dict(result.stage_ms),
        "rerank": bool(result.rerank),
        "rerank_top": len(result.reranked) if result.rerank else None,
        "lore_tokens": int(result.lore_tokens),
        "lore_ids": [hit.source_ref for hit in result.injected],
    }


def summarize(samples: list[dict], stage_keys: tuple[str, ...]) -> dict:
    """逐段 p50/p95/max（样本数 = 该段有耗时值的次数）。"""
    out: dict[str, dict] = {}
    for name in stage_keys:
        values = [
            float(s["stage_ms"][name])
            for s in samples
            if isinstance(s.get("stage_ms", {}).get(name), (int, float))
        ]
        out[name] = {
            "p50": _int_pct(values, 50),
            "p95": _int_pct(values, 95),
            "max": None if not values else int(round(max(values))),
            "count": len(values),
        }
    return out


def render_table(stats: dict) -> str:
    """分段 × p50/p95/max 的 markdown 表（可直接粘进报告）。"""
    lines = ["| 段 | p50 | p95 | max | 样本 |", "|---|---|---|---|---|"]
    for name, row in stats.items():
        lines.append(
            f"| {name} | {row['p50']} | {row['p95']} | {row['max']} | {row['count']} |"
        )
    return "\n".join(lines)


# ===== 入口 =====


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单请求检索探针（真实 embedding + 真实 rerank，M5-P0）"
    )
    parser.add_argument("--agent-id", type=int, required=True, help="被测角色 id")
    parser.add_argument(
        "--query", action="append", default=None,
        help="查询（可重复）；不给则用内置 3 条固定中文查询",
    )
    parser.add_argument("--runs", type=int, default=20, help="统计样本数（不含预热）")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数（不计入统计）")
    parser.add_argument("--out", help="输出 JSON 路径（默认 eval/report/probe-lore-<run_id>.json）")
    parser.add_argument("--dsn", help="覆盖 .env 的 DATABASE_URL（仅本脚本）")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    from app.config import get_settings
    from app.db.base import create_engine, create_session_factory
    from app.llm.factory import create_embedding_provider, create_llm_provider
    from app.lore.retriever import BgeReranker, LlmRewriter, Retriever, STAGE_KEYS

    settings = get_settings()
    if settings.embedding_provider == "fake":
        print(
            "[probe] 拒绝测量：EMBEDDING_PROVIDER=fake（fake 向量不能产出真实链路结论）",
            file=sys.stderr,
        )
        return 2
    if settings.lore_query_rewrite and settings.llm_provider != "fake":
        print(
            "[probe] 拒绝测量：LORE_QUERY_REWRITE=true 需要真实 LLM，"
            f"当前 LLM_PROVIDER={settings.llm_provider!r}（只允许 fake，0 token）",
            file=sys.stderr,
        )
        return 2

    queries = list(args.query) if args.query else list(DEFAULT_QUERIES)
    dsn = args.dsn or settings.database_url
    engine = create_engine(settings._to_psycopg(dsn))
    session_factory = create_session_factory(engine)
    embedder = create_embedding_provider(settings)
    reranker = BgeReranker(RERANK_MODEL, max_tokens=settings.lore_rerank_max_tokens)
    rerank_available = bool(reranker.available)
    rewriter = None
    if settings.lore_query_rewrite:
        rewriter = LlmRewriter(create_llm_provider(settings))
    retriever = Retriever(
        session_factory=session_factory,
        embedder=embedder,
        reranker=reranker,
        rewriter=rewriter,
        rewrite_enabled=settings.lore_query_rewrite,
        budget_tokens=settings.prompt_token_budget,
        lore_ratio=settings.lore_max_ratio,
        always_on_ratio=settings.lore_always_on_ratio,
        top_k=settings.lore_top_k,
    )

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    warmup_samples: list[dict] = []
    samples: list[dict] = []
    try:
        for i in range(max(0, args.warmup)):
            warmup_samples.append(
                await probe_once(retriever, agent_id=args.agent_id, query=queries[i % len(queries)])
            )
        print(
            f"[probe] 已预热 {len(warmup_samples)} 次（不进统计；rerank_available={rerank_available}）"
        )
        for i in range(max(0, args.runs)):
            samples.append(
                await probe_once(retriever, agent_id=args.agent_id, query=queries[i % len(queries)])
            )
    finally:
        await engine.dispose()

    errors = [s for s in samples if "error" in s]
    ok_samples = [s for s in samples if "error" not in s]
    stats = summarize(ok_samples, STAGE_KEYS)
    rerank_effective = sum(1 for s in ok_samples if s.get("rerank"))
    last = ok_samples[-1] if ok_samples else {}
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")

    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "command": "uv run python scripts/probe_lore_latency.py " + " ".join(args.argv),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "agent_id": args.agent_id,
        "database": dsn.rsplit("/", 1)[-1].split("?", 1)[0] or None,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": (
            settings.embedding_model if settings.embedding_provider == "local" else None
        ),
        "rerank_model": RERANK_MODEL,
        "rerank_max_tokens": settings.lore_rerank_max_tokens,
        "lore_top_k": settings.lore_top_k,
        "rerank_available": rerank_available,
        "lore_enabled": settings.lore_enabled,
        "lore_query_rewrite": settings.lore_query_rewrite,
        "runs": args.runs,
        "warmup": args.warmup,
        "queries": queries,
        "started_at": started_at,
        "finished_at": finished_at,
        "sample_count": len(ok_samples),
        "error_count": len(errors),
        # 本契约不实现降级：`degraded` 只表示“重排不可用或本次探针有失败样本”
        "degraded": bool(not rerank_available or errors),
        "rerank_effective": rerank_effective,
        "rerank_top": last.get("rerank_top"),
        "stage_ms": stats,
        "warmup_samples": warmup_samples,
        "samples": samples,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("---- lore latency probe（单请求 · 真实 embedding/rerank）----")
    print(f"agent_id      : {args.agent_id}  queries: {len(queries)}  "
          f"runs: {len(ok_samples)}  errors: {len(errors)}")
    print(f"embedding     : {settings.embedding_provider}/{settings.embedding_model}")
    print(f"rerank        : {RERANK_MODEL}（effective={rerank_effective}/{len(ok_samples)}）")
    print(render_table(stats))
    print(f"产物          : {args.out_path.as_posix()}")

    if not ok_samples:
        print("[probe] 没有任何成功样本，拒绝产出结论", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    if args.agent_id < 1:
        print("[probe] --agent-id 必须 ≥ 1", file=sys.stderr)
        return 2
    if args.runs < 1 or args.warmup < 0:
        print("[probe] --runs 必须 ≥ 1，--warmup 必须 ≥ 0", file=sys.stderr)
        return 2

    args.argv = argv
    out = (
        Path(args.out) if args.out
        else DEFAULT_OUT_DIR / f"probe-lore-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}.json"
    )
    args.out_path = out if out.is_absolute() else ROOT / out
    stem = args.out_path.stem
    args.run_id = stem[len("probe-lore-"):] if stem.startswith("probe-lore-") else stem

    if sys.platform == "win32":  # psycopg 异步在 Windows 需 SelectorEventLoop（同 tests/conftest.py）
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
