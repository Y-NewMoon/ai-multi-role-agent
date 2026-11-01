"""rerank 微观基准（M5-P0 P7）：判断 257ms/对 是"每对成本高"还是"固定开销大"。

只在**本脚本进程内**做实验（可调 `torch.set_num_threads`），**不改生产代码/行为**。
从库读真实 chunk（按 `source_ref` 升序取前 max(n) 条，不足即报错退出），
预热 1 轮后按 n 重复测 p50，再做最小二乘 `p50 = slope * n + intercept`，
可选扫描 torch 线程数。

用法：
    uv run python scripts/bench_rerank.py --agent-id 1 --repeat 5 `
      --n 1,5,10,20 --threads 1,2,4,8

产物：eval/report/bench-rerank-<run_id>.json（含环境、拟合、线程扫描、结论）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUT_DIR = ROOT / "eval" / "report"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_QUERY = "祖父开的茶馆叫什么名字"
# 只用于环境记录（风险 #5）：这些常驻进程会抢 CPU
WATCH_PROCESSES = ("QQ.exe", "SnowLume", "MuMu", "wallpaper", "WeChat")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
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


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:  # noqa: BLE001
            pass
    return platform.processor() or platform.machine() or "unknown"


def _running_watched() -> dict[str, bool]:
    """记录看护进程是否在跑（Windows 用 tasklist；其它平台返回空）。"""
    if sys.platform != "win32":
        return {}
    try:
        out = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"],
            capture_output=True, text=True, timeout=15,
        )
        text = out.stdout.lower()
        return {name: (name.lower() in text) for name in WATCH_PROCESSES}
    except Exception:  # noqa: BLE001
        return {}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[min(k, len(ordered) - 1)]


def p50(values: list[float]) -> float | None:
    value = percentile(values, 50)
    return None if value is None else round(value, 1)


def linear_fit(xs: list[float], ys: list[float]) -> dict:
    """最小二乘 `y = slope*x + intercept`，并给出 R²（纯 python，不引入新依赖）。"""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "slope_ms_per_pair": round(slope, 2),
        "intercept_ms": round(intercept, 2),
        "r2": round(r2, 4),
    }


def render_table(title: str, header: list[str], rows: list[list]) -> str:
    lines = [f"**{title}**", "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _parse_int_list(raw: str, name: str) -> list[int]:
    try:
        values = sorted({int(part) for part in raw.split(",") if part.strip()})
    except ValueError:
        raise SystemExit(f"[bench] {name} 需为逗号分隔的整数：{raw!r}")
    if not values or any(v < 1 for v in values):
        raise SystemExit(f"[bench] {name} 必须为正整数：{raw!r}")
    return values


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rerank 微观基准（M5-P0 P7）")
    parser.add_argument("--agent-id", type=int, required=True)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--n", default="1,5,10,20", help="候选数（逗号分隔）")
    parser.add_argument("--repeat", type=int, default=5, help="每个 n 重复次数")
    parser.add_argument("--threads", help="可选：torch 线程数扫描（逗号分隔）")
    parser.add_argument("--out", help="输出 JSON（默认 eval/report/bench-rerank-<run_id>.json）")
    parser.add_argument("--dsn", help="覆盖 .env 的 DATABASE_URL（仅本脚本）")
    return parser.parse_args(argv)


async def _load_chunks(agent_id: int, limit: int, dsn: str) -> list[str]:
    from sqlalchemy import select

    from app.db.base import create_engine, create_session_factory
    from app.db.models import LoreChunk

    engine = create_engine(dsn)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as db:
            rows = (
                await db.execute(
                    select(LoreChunk.source_ref, LoreChunk.content)
                    .where(
                        LoreChunk.agent_id == agent_id,
                        LoreChunk.status == "active",
                        LoreChunk.embedding.is_not(None),
                    )
                    .order_by(LoreChunk.source_ref.asc())
                    .limit(limit)
                )
            ).all()
    finally:
        await engine.dispose()
    return [row.content for row in rows]


def _time_rank(reranker, query: str, docs: list[str], n: int) -> float:
    started = time.perf_counter()
    reranker.rank(query, docs[:n], n)
    return (time.perf_counter() - started) * 1000.0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    if args.agent_id < 1:
        print("[bench] --agent-id 必须 ≥ 1", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("[bench] --repeat 必须 ≥ 1", file=sys.stderr)
        return 2
    ns = _parse_int_list(args.n, "--n")
    thread_counts = _parse_int_list(args.threads, "--threads") if args.threads else []

    out = (
        Path(args.out) if args.out
        else DEFAULT_OUT_DIR / f"bench-rerank-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}.json"
    )
    out = out if out.is_absolute() else ROOT / out
    args.out_path = out
    stem = out.stem
    args.run_id = stem[len("bench-rerank-"):] if stem.startswith("bench-rerank-") else stem

    if sys.platform == "win32":  # psycopg 异步需 SelectorEventLoop（同 tests/conftest.py）
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    from app.config import get_settings

    settings = get_settings()
    dsn = settings._to_psycopg(args.dsn or settings.database_url)
    docs = asyncio.run(_load_chunks(args.agent_id, max(ns), dsn))
    if len(docs) < max(ns):
        print(
            f"[bench] chunk 不足：需要 {max(ns)} 条，库里只有 {len(docs)} 条（agent_id={args.agent_id}）",
            file=sys.stderr,
        )
        return 2

    import torch

    from app.lore.retriever import BgeReranker

    reranker = BgeReranker(RERANK_MODEL)
    if not reranker.available:
        print("[bench] reranker 不可用（模型加载失败），拒绝产出结论", file=sys.stderr)
        return 2
    default_threads = torch.get_num_threads()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.perf_counter()

    # 预热一轮（20 对，不计入统计）
    _time_rank(reranker, args.query, docs, max(ns))
    print(f"[bench] 已预热 1 轮（{max(ns)} 对，不计入统计）")

    results: list[dict] = []
    for n in ns:
        samples = [_time_rank(reranker, args.query, docs, n) for _ in range(args.repeat)]
        med = p50(samples)
        results.append({
            "n": n,
            "repeat": args.repeat,
            "p50_ms": med,
            "per_pair_ms": round(med / n, 2) if med is not None else None,
            "min_ms": round(min(samples), 1),
            "max_ms": round(max(samples), 1),
        })
        print(f"[bench] n={n:>2}  p50={med}ms  per_pair={results[-1]['per_pair_ms']}ms")

    fit = linear_fit([r["n"] for r in results], [r["p50_ms"] for r in results])

    thread_results: list[dict] = []
    if thread_counts:
        original = default_threads
        try:
            for k in thread_counts:
                torch.set_num_threads(k)
                _time_rank(reranker, args.query, docs, max(ns))  # 每个线程数下预热一次
                samples = [
                    _time_rank(reranker, args.query, docs, max(ns))
                    for _ in range(args.repeat)
                ]
                thread_results.append({
                    "threads": k,
                    "p50_ms": p50(samples),
                    "per_pair_ms": round((p50(samples) or 0) / max(ns), 2),
                })
                print(f"[bench] threads={k:>2}  p50={thread_results[-1]['p50_ms']}ms")
        finally:
            torch.set_num_threads(original)

    max_n = max(ns)
    p50_max = next(r["p50_ms"] for r in results if r["n"] == max_n)
    fixed_share = (fit["intercept_ms"] / p50_max) if p50_max else 0.0
    if fixed_share >= 0.25:
        bottleneck = "固定开销为主"
        advice = "先降低“固定开销”（换推理后端/批处理/线程），单纯减候选收益有限"
    else:
        bottleneck = "每对成本为主"
        advice = "先做 M5 手段 3（减少候选）或换更快的推理后端；线程调优收益有限"

    best_thread = None
    if thread_results:
        by_p50 = sorted(thread_results, key=lambda r: (r["p50_ms"] is None, r["p50_ms"] or 0))
        best_thread = by_p50[0]
    conclusion = (
        f"n={max_n} 的 p50={p50_max}ms，拟合 slope={fit['slope_ms_per_pair']}ms/对、"
        f"intercept={fit['intercept_ms']}ms、R²={fit['r2']}；固定开销占比 {fixed_share:.0%} → "
        f"{bottleneck}。建议：{advice}。"
    )
    if best_thread:
        baseline = next(
            (r["p50_ms"] for r in thread_results if r["threads"] == default_threads),
            None,
        )
        gain = (
            f"相对默认 {default_threads} 线程（{baseline}ms）提升 "
            f"{(baseline - best_thread['p50_ms']) / baseline:.0%}"
            if baseline else f"默认 {default_threads} 线程不在扫描列表，无法算相对收益"
        )
        conclusion += f" 线程最优 {best_thread['threads']}（{best_thread['p50_ms']}ms，{gain}）。"

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "command": "uv run python scripts/bench_rerank.py " + " ".join(argv),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "agent_id": args.agent_id,
        "query": args.query,
        "model": RERANK_MODEL,
        "n": ns,
        "repeat": args.repeat,
        "warmup_pairs": max_n,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "started_at": started_at,
        "finished_at": finished_at,
        "cpu": _cpu_name(),
        "torch_version": torch.__version__,
        "torch_threads_default": default_threads,
        "watch_processes": _running_watched(),
        "results": results,
        "fit": fit,
        "threads": thread_results,
        "best_threads": best_thread,
        "bottleneck": bottleneck,
        "conclusion": conclusion,
    }

    out = args.out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("---- rerank micro-bench ----")
    print(f"cpu={report['cpu']}  torch={report['torch_version']}  "
          f"threads={default_threads}  chunks={len(docs)}")
    print(render_table(
        "候选数 × 延迟",
        ["n", "p50", "per_pair", "min", "max"],
        [[r["n"], f"{r['p50_ms']}ms", f"{r['per_pair_ms']}ms", r["min_ms"], r["max_ms"]]
         for r in results],
    ))
    print(f"\n**线性拟合**：slope={fit['slope_ms_per_pair']}ms/对  "
          f"intercept={fit['intercept_ms']}ms  R²={fit['r2']}")
    if thread_results:
        print("\n" + render_table(
            f"线程扫描（n={max_n}）",
            ["threads", "p50", "per_pair"],
            [[r["threads"], f"{r['p50_ms']}ms", f"{r['per_pair_ms']}ms"]
             for r in thread_results],
        ))
    print(f"\n**结论**：{conclusion}")
    print(f"\n产物：{out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
