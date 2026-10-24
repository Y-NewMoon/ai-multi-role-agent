"""SSE 压测（V2.4 §2.10 / C4）：只允许 fake LLM，20 并发 × N 轮，不烧真实 token。

用法：
    uv run python scripts/load_test.py --base-url http://127.0.0.1:8010 \
      --concurrency 20 --rounds 20 --session-ids 1,2,...,20 \
      --out eval/report/loadtest-20261021.json

启动前先 `GET /health`，`llm_provider` 不是 `fake` 时退出 2（绝不指向真实生产 server）。
无 `httpx` 时写 `skipped=true` 报告并退出 0；不访问网络、不调用 LLM。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "eval" / "report"
SCHEMA_VERSION = "1.0"
HEALTH_TIMEOUT_S = 5.0


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


def _sha256_file(path: Path) -> str | None:
    return None if not path.exists() else hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], p: float) -> float | None:
    """nearest-rank 分位数；空列表返回 None（与 trace_report 口径一致）。"""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[min(k, len(ordered) - 1)]


def _int_pct(values: list[float], p: float) -> int | None:
    value = percentile(values, p)
    return None if value is None else int(round(value))


def _rate(count: int, total: int) -> float | None:
    return None if total == 0 else round(count / total, 6)


# ===== SSE 解析 =====


class SseParser:
    """逐行喂 SSE，凑齐一帧返回 `(event, payload)`；空行即帧结束。"""

    def __init__(self) -> None:
        self._event: str | None = None
        self._data: list[str] = []

    def feed(self, line: str) -> tuple[str, dict] | None:
        if line.startswith("event:"):
            self._event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            self._data.append(line[len("data:"):].strip())
        elif not line.strip() and self._event is not None:
            event, raw = self._event, "\n".join(self._data)
            self._event, self._data = None, []
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"raw": raw}
            return event, payload if isinstance(payload, dict) else {"value": payload}
        return None


# ===== 单请求 =====


def _result(*, ok: bool, status: int | None, first_token_ms: int | None,
            total_ms: int | None, error: str | None,
            session_id: int | None = None, round_no: int | None = None) -> dict:
    return {
        "session_id": session_id,
        "round_no": round_no,
        "ok": ok,
        "status": status,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "error": error,
    }


async def request_once(
    client, *, base_url: str, session_id: int, round_no: int, timeout_s: float
) -> dict:
    """跑一轮 SSE：记录首 token / 总时长 / 错误；只读事件，不调用 LLM。"""
    started = time.perf_counter()
    first_token_ms: int | None = None
    error: str | None = None
    done = False
    status: int | None = None
    try:
        async with asyncio.timeout(timeout_s):
            async with client.stream(
                "POST",
                f"{base_url}/api/v1/chat/stream",
                json={"session_id": session_id, "message": f"压测第 {round_no} 轮"},
            ) as resp:
                status = resp.status_code
                if status != 200:
                    await resp.aread()
                    return _result(
                        ok=False, status=status, first_token_ms=None,
                        total_ms=int((time.perf_counter() - started) * 1000),
                        error=f"HTTP {status}",
                        session_id=session_id, round_no=round_no,
                    )
                parser = SseParser()
                async for line in resp.aiter_lines():
                    frame = parser.feed(line)
                    if frame is None:
                        continue
                    event, data = frame
                    if event == "token" and first_token_ms is None:
                        first_token_ms = int((time.perf_counter() - started) * 1000)
                    elif event == "error":
                        error = str(data.get("detail") or "sse error")
                    elif event == "done":
                        done = True
                    if done or error:
                        break
    except TimeoutError:
        error = f"timeout {timeout_s}s"
    except Exception as exc:  # noqa: BLE001 - 单请求失败计数，不中断整轮压测
        error = f"{type(exc).__name__}: {exc}"
    total_ms = int((time.perf_counter() - started) * 1000)
    return _result(
        ok=status == 200 and done and error is None,
        status=status,
        first_token_ms=first_token_ms,
        total_ms=total_ms,
        error=error,
        session_id=session_id,
        round_no=round_no,
    )


async def run_worker(
    client, *, base_url: str, session_id: int, rounds: int, timeout_s: float
) -> list[dict]:
    results: list[dict] = []
    for round_no in range(1, max(0, rounds) + 1):
        results.append(
            await request_once(
                client,
                base_url=base_url,
                session_id=session_id,
                round_no=round_no,
                timeout_s=timeout_s,
            )
        )
    return results


# ===== 聚合 =====


def summarize(
    results: list[dict],
    *,
    run_id: str,
    base_url: str,
    llm_provider: str,
    concurrency: int,
    rounds: int,
    wall_ms: int | None,
    fake_calls: int | None = None,
) -> dict:
    """把逐请求结果聚合成 §8.2 报告主体（纯函数，便于单测）。

    `fake_calls` 必须来自 server `/health` 的 `fake_llm_calls` 前后差值（R9）；
    拿不到时为 null，**不得**用“成功请求数”冒充 fake 调用次数。
    """
    requests = len(results)
    errors = sum(1 for r in results if not r.get("ok"))
    ttft = [float(r["first_token_ms"]) for r in results if r.get("first_token_ms") is not None]
    totals = [float(r["total_ms"]) for r in results if r.get("total_ms") is not None]
    throughput = None
    if wall_ms and wall_ms > 0:
        throughput = round(requests / (wall_ms / 1000.0), 3)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_url": base_url,
        "llm_provider": llm_provider,
        "concurrency": concurrency,
        "rounds": rounds,
        "requests": requests,
        "errors": errors,
        "error_rate": _rate(errors, requests),
        "p50_first_token_ms": _int_pct(ttft, 50),
        "p95_first_token_ms": _int_pct(ttft, 95),
        "p50_total_ms": _int_pct(totals, 50),
        "p95_total_ms": _int_pct(totals, 95),
        "throughput_req_s": throughput,
        # R9：真实 fake 调用次数 = /health 计数器前后差值；未测得为 null
        "fake_llm_calls": fake_calls,
        "fake_llm_calls_source": "health_delta" if fake_calls is not None else "unavailable",
        "usage": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        "budget_section": "4.6#4",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# SSE 压测报告（fake LLM）",
        "",
        f"- run_id：`{report['run_id']}`",
        f"- base_url：`{report['base_url']}`（llm_provider=`{report['llm_provider']}`）",
        f"- 并发 / 轮数：{report['concurrency']} × {report['rounds']}",
        f"- 请求数 / 错误数：{report['requests']} / {report['errors']}"
        f"（error_rate={report['error_rate']}）",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| p50 / p95 首 token（ms） | {report['p50_first_token_ms']} / "
        f"{report['p95_first_token_ms']} |",
        f"| p50 / p95 总时长（ms） | {report['p50_total_ms']} / {report['p95_total_ms']} |",
        f"| 吞吐（req/s） | {report['throughput_req_s']} |",
        f"| fake LLM 调用（health 差值） | {report['fake_llm_calls']}"
        f"（{report['fake_llm_calls_source']}） |",
        "",
        f"usage：llm_calls=0、prompt_tokens=0（本脚本只压 fake，budget_section="
        f"{report['budget_section']}）",
        "",
    ]
    return "\n".join(lines)


def _database_name(url: str | None = None) -> str | None:
    url = os.environ.get("DATABASE_URL", "") if url is None else url
    if not url or "/" not in url:
        return None
    return url.rsplit("/", 1)[-1].split("?", 1)[0] or None


def _trace_sha256(trace_dir: str | None) -> str | None:
    """压测 server 的 trace 目录内容哈希（同一台机器时可读；否则 null）。"""
    if not trace_dir:
        return None
    path = Path(trace_dir)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    files = sorted(path.glob("*.jsonl"))
    if not files:
        return None
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def _build_manifest(
    *, run_id: str, command: str, started_at: str, finished_at: str,
    report_sha: str | None, llm_provider: str, skipped: bool,
    health: dict | None = None, session_ids: list[int] | None = None,
    cases_sha: str | None = None,
) -> dict:
    """§8.3 manifest：压测只允许 fake，usage 恒为 0（R8/R9）。"""
    health = health or {}
    trace_dir = health.get("trace_dir")
    return {
        "run_id": run_id,
        "command": command,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "started_at": started_at,
        "finished_at": finished_at,
        "skipped": skipped,
        "env": {
            "python": sys.version.split()[0],
            # R8/R9：server 真实库与嵌入/世界书/trace 配置由 /health 提供
            "database": health.get("database") or _database_name(),
            "llm_model": llm_provider,
            "embedding_provider": health.get("embedding_provider"),
            "embedding_model": health.get("embedding_model"),
            "rerank_model": health.get("rerank_model"),
            "lore_enabled": health.get("lore_enabled"),
            "lore_static_mode": health.get("lore_static_mode"),
            # R1：图开关必须进 manifest，否则两次 20×20 不是同负载、无法复现
            "memory_extract_enabled": health.get("memory_extract_enabled"),
            "memory_profile_enabled": health.get("memory_profile_enabled"),
            "memory_l2_enabled": health.get("memory_l2_enabled"),
            "memory_weighted_recall": health.get("memory_weighted_recall"),
            "trace_enabled": health.get("trace_enabled"),
            "trace_dir": trace_dir,
            "trace_sample_rate": health.get("trace_sample_rate"),
        },
        "inputs": {
            "trace_sha256": _trace_sha256(trace_dir),
            "sessions": list(session_ids or []),
            "seed_script": "scripts/seed_loadtest.py",
        },
        "outputs": {"report_sha256": report_sha, "cases_sha256": cases_sha},
        "usage": {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "judge_cache_hits": 0,
            "judge_cache_misses": 0,
            "budget_prompt_tokens": 0,
            "within_budget": True,
        },
        "budget_section": "4.6#4",
    }


def _write_outputs(
    *, out_path: Path, report: dict, manifest: dict, skipped: bool,
    cases: list[dict] | None = None,
) -> tuple[Path, Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    # R9：逐请求原始数据，审查方可独立重算 p95/吞吐
    cases_path = out_path.with_suffix(".cases.jsonl")
    if cases is not None:
        cases_path.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + ("\n" if cases else ""),
            encoding="utf-8",
        )
        manifest["outputs"]["cases_sha256"] = _sha256_file(cases_path)
    manifest["outputs"]["report_sha256"] = _sha256_file(out_path)
    manifest["skipped"] = skipped
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, manifest_path


def _parse_session_ids(raw: str) -> list[int]:
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in ids:
            ids.append(value)
    return ids


# ===== 入口 =====


async def _run(args, session_ids: list[int]) -> int:
    import httpx

    base_url = args.base_url.rstrip("/")
    timeout = httpx.Timeout(args.timeout, connect=HEALTH_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(f"{base_url}/health")
            resp.raise_for_status()
            health = resp.json()
            provider = str(health.get("llm_provider") or "")
        except Exception as exc:  # noqa: BLE001 - 健康检查失败：不能确认 fake，退出 2
            print(f"[load_test] 无法确认 fake LLM（/health 失败）：{exc}", file=sys.stderr)
            return 2
        if provider != "fake":
            print(
                f"[load_test] 拒绝压测：llm_provider={provider!r}（只允许 fake，C4）",
                file=sys.stderr,
            )
            return 2

        fake_before = health.get("fake_llm_calls")
        wall_started = time.perf_counter()
        batches = await asyncio.gather(
            *[
                run_worker(
                    client,
                    base_url=base_url,
                    session_id=sid,
                    rounds=args.rounds,
                    timeout_s=args.timeout,
                )
                for sid in session_ids
            ]
        )
        wall_ms = int((time.perf_counter() - wall_started) * 1000)
        results = [r for batch in batches for r in batch]

        # R9：真实 fake 调用次数 = 计数器前后差值（拿不到就 null，不冒充）
        fake_calls: int | None = None
        try:
            after = (await client.get(f"{base_url}/health")).json()
            fake_after = after.get("fake_llm_calls")
            if isinstance(fake_before, int) and isinstance(fake_after, int):
                fake_calls = max(0, fake_after - fake_before)
            health = {**health, **after}
        except Exception as exc:  # noqa: BLE001 - 后置健康检查失败不影响压测结果
            print(f"[load_test] 后置 /health 失败，fake_llm_calls=null：{exc}", file=sys.stderr)

    report = summarize(
        results,
        run_id=args.run_id,
        base_url=base_url,
        llm_provider=provider,
        concurrency=len(session_ids),
        rounds=args.rounds,
        wall_ms=wall_ms,
        fake_calls=fake_calls,
    )
    return _emit(report, args=args, provider=provider, health=health,
                 session_ids=session_ids, cases=results)


def _emit(report: dict, *, args, provider: str, health: dict | None = None,
          session_ids: list[int] | None = None, cases: list[dict] | None = None) -> int:
    started_at = args.started_at
    manifest = _build_manifest(
        run_id=args.run_id,
        command="uv run python scripts/load_test.py " + " ".join(args.argv),
        started_at=started_at,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        report_sha=None,
        llm_provider=provider,
        skipped=False,
        health=health,
        session_ids=session_ids,
    )
    md_path, manifest_path = _write_outputs(
        out_path=args.out_path, report=report, manifest=manifest, skipped=False,
        cases=cases,
    )
    print("---- load test（fake LLM）----")
    print(f"requests      : {report['requests']}  errors: {report['errors']}"
          f"  error_rate: {report['error_rate']}")
    print(f"p50/p95 ttft  : {report['p50_first_token_ms']} / "
          f"{report['p95_first_token_ms']} ms")
    print(f"p50/p95 total : {report['p50_total_ms']} / {report['p95_total_ms']} ms")
    print(f"throughput    : {report['throughput_req_s']} req/s")
    print(f"报告          : {args.out_path}")
    print(f"markdown      : {md_path}")
    print(f"manifest      : {manifest_path}")
    return 0


def _emit_skipped(args, reason: str) -> int:
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "llm_provider": None,
        "concurrency": args.concurrency,
        "rounds": args.rounds,
        "requests": 0,
        "errors": 0,
        "error_rate": 0.0,
        "p50_first_token_ms": None,
        "p95_first_token_ms": None,
        "p50_total_ms": None,
        "p95_total_ms": None,
        "throughput_req_s": None,
        "fake_llm_calls": 0,
        "usage": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        "skipped": True,
        "reason": reason,
        "budget_section": "4.6#4",
    }
    manifest = _build_manifest(
        run_id=args.run_id,
        command="uv run python scripts/load_test.py " + " ".join(args.argv),
        started_at=args.started_at,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        report_sha=None,
        llm_provider="unknown",
        skipped=True,
    )
    _write_outputs(out_path=args.out_path, report=report, manifest=manifest, skipped=True)
    print(f"[load_test] skipped：{reason}", file=sys.stderr)
    print(f"报告          : {args.out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="SSE 压测（只允许 fake LLM，C4）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--session-ids", required=True, help="逗号分隔的会话 id（每会话一条并发）")
    parser.add_argument("--timeout", type=float, default=60.0, help="单请求超时（秒）")
    parser.add_argument("--out", help="输出 JSON 路径（默认 eval/report/loadtest-<run_id>.json）")
    args = parser.parse_args(argv)

    args.argv = argv
    args.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    session_ids = _parse_session_ids(args.session_ids)
    if args.concurrency < 1 or args.rounds < 1:
        print("[load_test] --concurrency/--rounds 必须 ≥1", file=sys.stderr)
        return 2
    if not session_ids:
        print("[load_test] --session-ids 不能为空", file=sys.stderr)
        return 2
    if len(session_ids) < args.concurrency:
        print(
            f"[load_test] session 数不足：需要 {args.concurrency} 个不同会话，"
            f"实际 {len(session_ids)} 个（同会话会被 B-02 串行）",
            file=sys.stderr,
        )
        return 2

    args.out_path = (
        Path(args.out) if args.out
        else DEFAULT_OUT_DIR / f"loadtest-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}.json"
    )
    if not args.out_path.is_absolute():
        args.out_path = ROOT / args.out_path
    stem = args.out_path.stem
    args.run_id = stem[len("loadtest-"):] if stem.startswith("loadtest-") else stem

    try:
        import httpx  # noqa: F401
    except ImportError:
        return _emit_skipped(args, "httpx 未安装，跳过压测（不得用真实 LLM 替代）")

    return asyncio.run(_run(args, session_ids))


if __name__ == "__main__":
    raise SystemExit(main())
