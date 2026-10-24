"""trace 聚合报告（V2.4 §2.5）。

从 `data/traces/{YYYY-MM-DD}.jsonl` 读 JSONL，输出
`eval/report/trace-<run_id>.{json,md,manifest.json}`，**不调用 LLM**。

用法：
    uv run python scripts/trace_report.py --date 2026-10-21 --out eval/report/trace-20261021.json
    uv run python scripts/trace_report.py --since 2026-10-20 --until 2026-10-21

空输入 → `traces=0`、各项指标 null，退出 0（不得伪造数据）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "data" / "traces"
DEFAULT_OUT_DIR = ROOT / "eval" / "report"
SCHEMA_VERSION = "1.0"


# ===== 文件与 manifest =====


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - 非 git 环境也要能出报告
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def _sha256_bytes(chunks: list[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_files(*, trace_dir: Path, date: str | None, since: str | None, until: str | None) -> list[Path]:
    """按 --date 或 --since/--until 选择 JSONL；按文件名（日期）升序。"""
    if not trace_dir.exists():
        return []
    files = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.name)
    if date:
        return [p for p in files if p.stem == date]
    out: list[Path] = []
    for p in files:
        day = p.stem
        if since and day < since:
            continue
        if until and day > until:
            continue
        out.append(p)
    return out


def load_records(files: list[Path]) -> tuple[list[dict], int]:
    """读全部记录；损坏行跳过并计数（不因个别坏行失败）。"""
    records: list[dict] = []
    malformed = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            malformed += 1
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                malformed += 1
    return records, malformed


# ===== 统计 =====


def percentile(values: list[float], p: float) -> float | None:
    """nearest-rank 分位数；空列表返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[min(k, len(ordered) - 1)]


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rate(count: int, total: int) -> float | None:
    return None if total == 0 else round(count / total, 6)


def _ms_p95(values: list[float]) -> int | None:
    value = percentile(values, 95)
    return None if value is None else int(round(value))


def _is_fake(record: dict) -> bool:
    """R3：fake 压测 trace（model=fake / usage_source=fake）不计入成本与延迟。"""
    llm = record.get("llm") or {}
    return str(llm.get("usage_source") or "").lower() == "fake" or (
        str(llm.get("model") or "").lower() == "fake"
    )


def aggregate(records: list[dict]) -> dict:
    """把逐条 trace 聚合成报告主体（纯函数，便于单测）。

    R3：fake trace 单独计数（`fake_traces/fake_calls`），不进入成本与耗时聚合。
    """
    all_records = records
    fake_records = [r for r in all_records if _is_fake(r)]
    records = [r for r in all_records if not _is_fake(r)]
    total = len(records)
    errors = sum(1 for r in records if r.get("error"))
    degraded = sum(1 for r in records if r.get("degraded"))

    total_ms = [_num(r.get("total_ms")) for r in records if r.get("total_ms") is not None]
    first_token_ms = [
        _num(r.get("first_token_ms")) for r in records if r.get("first_token_ms") is not None
    ]

    # 节点：按名字聚合耗时与失败率
    node_values: dict[str, list[float]] = {}
    node_fail: dict[str, int] = {}
    for record in records:
        for node in record.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            name = str(node.get("name") or "unknown")
            node_values.setdefault(name, []).append(_num(node.get("ms")))
            if not node.get("ok", True):
                node_fail[name] = node_fail.get(name, 0) + 1
    nodes = {
        name: {
            "count": len(values),
            "failures": node_fail.get(name, 0),
            "fail_rate": _rate(node_fail.get(name, 0), len(values)),
            "p50_ms": None if percentile(values, 50) is None else int(round(percentile(values, 50))),
            "p95_ms": _ms_p95(values),
        }
        for name, values in sorted(node_values.items())
    }

    # 工具：调用次数 / 失败率 / 耗时
    tool_values: dict[str, list[float]] = {}
    tool_fail: dict[str, int] = {}
    for record in records:
        for tool in record.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "unknown")
            tool_values.setdefault(name, []).append(_num(tool.get("ms")))
            if not tool.get("ok", True):
                tool_fail[name] = tool_fail.get(name, 0) + 1
    tools = {
        name: {
            "calls": len(values),
            "failures": tool_fail.get(name, 0),
            "failure_rate": _rate(tool_fail.get(name, 0), len(values)),
            "p95_ms": _ms_p95(values),
        }
        for name, values in sorted(tool_values.items())
    }

    # 成本/usage：按 agent 与 session 聚合
    def _bucket(key: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for record in records:
            llm = record.get("llm") or {}
            if key == "agent":
                raw = record.get("agent_id")
                name = "unknown" if raw is None else str(raw)
            else:
                raw_session = record.get("session_id")
                raw_group = record.get("group_session_id")
                if raw_session is not None:
                    name = str(raw_session)
                elif raw_group is not None:
                    name = f"g{raw_group}"
                else:
                    name = "unknown"
            row = out.setdefault(
                name,
                {"traces": 0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_cny": 0.0},
            )
            row["traces"] += 1
            row["calls"] += int(_num(llm.get("calls")))
            row["prompt_tokens"] += int(_num(llm.get("prompt_tokens")))
            row["completion_tokens"] += int(_num(llm.get("completion_tokens")))
            row["cost_cny"] += _num(llm.get("cost_cny"))
        for row in out.values():
            row["cost_cny"] = round(row["cost_cny"], 6)
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    by_agent = _bucket("agent")
    by_session = _bucket("session")

    totals = {
        "calls": sum(int(_num((r.get("llm") or {}).get("calls"))) for r in records),
        "prompt_tokens": sum(
            int(_num((r.get("llm") or {}).get("prompt_tokens"))) for r in records
        ),
        "completion_tokens": sum(
            int(_num((r.get("llm") or {}).get("completion_tokens"))) for r in records
        ),
        "cost_cny": round(
            sum(_num((r.get("llm") or {}).get("cost_cny")) for r in records), 6
        ),
    }

    # token 段占比（仅在 trace 带 budget 时）
    ratios = [
        _num(r["budget"].get("estimated_prompt_tokens")) / _num(r["budget"].get("prompt_token_budget"))
        for r in records
        if isinstance(r.get("budget"), dict) and _num(r["budget"].get("prompt_token_budget")) > 0
    ]
    over = sum(
        1
        for r in records
        if isinstance(r.get("budget"), dict) and r["budget"].get("over_budget")
    )
    token_segments = {
        "traces_with_budget": len(ratios),
        "mean_prompt_ratio": round(sum(ratios) / len(ratios), 6) if ratios else None,
        "max_prompt_ratio": round(max(ratios), 6) if ratios else None,
        "over_budget_traces": over,
        "over_budget_rate": _rate(over, len(ratios)),
    }

    return {
        "traces": total,
        "records_total": len(all_records),
        "fake_traces": len(fake_records),
        "fake_calls": sum(
            int(_num((r.get("llm") or {}).get("calls"))) for r in fake_records
        ),
        "errors": errors,
        "degraded": degraded,
        "error_rate": _rate(errors, total),
        "degraded_rate": _rate(degraded, total),
        "p50_total_ms": None if percentile(total_ms, 50) is None else int(round(percentile(total_ms, 50))),
        "p95_total_ms": _ms_p95(total_ms),
        "p50_first_token_ms": (
            None if percentile(first_token_ms, 50) is None else int(round(percentile(first_token_ms, 50)))
        ),
        "p95_first_token_ms": _ms_p95(first_token_ms),
        "nodes": nodes,
        "tools": tools,
        "by_agent": by_agent,
        "by_session": by_session,
        "totals": totals,
        "token_segments": token_segments,
    }


# ===== 输出 =====


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(report: dict) -> str:
    """Markdown 摘要（与 JSON 同源，不另算）。"""
    lines = [
        "# trace 聚合报告",
        "",
        f"- run_id：`{report['run_id']}`",
        f"- 生成时间：{report['generated_at']}",
        f"- git_sha：`{report['git_sha']}`（dirty={report['git_dirty']}）",
        f"- 过滤：{report['filters']}",
        f"- 输入文件数：{report['inputs']['files']}（损坏行 {report['inputs']['malformed_lines']}）",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    for key in (
        "traces",
        "error_rate",
        "degraded_rate",
        "p50_total_ms",
        "p95_total_ms",
        "p50_first_token_ms",
        "p95_first_token_ms",
    ):
        lines.append(f"| {key} | {_fmt(report['metrics'][key])} |")
    totals = report["metrics"]["totals"]
    lines.append(
        f"| llm calls / prompt / completion / cost_cny | "
        f"{totals['calls']} / {totals['prompt_tokens']} / "
        f"{totals['completion_tokens']} / {_fmt(totals['cost_cny'])} |"
    )
    lines.append(
        f"| fake trace（已从成本/耗时排除） | {report['metrics']['fake_traces']} 条 / "
        f"{report['metrics']['fake_calls']} 次调用 |"
    )
    segments = report["metrics"]["token_segments"]
    lines.append(
        f"| prompt 预算占比（mean / max） | "
        f"{_fmt(segments['mean_prompt_ratio'])} / {_fmt(segments['max_prompt_ratio'])} |"
    )

    lines += ["", "## 节点", "", "| 节点 | count | fail_rate | p50_ms | p95_ms |", "|---|---|---|---|---|"]
    for name, row in report["metrics"]["nodes"].items():
        lines.append(
            f"| {name} | {row['count']} | {_fmt(row['fail_rate'])} | "
            f"{_fmt(row['p50_ms'])} | {_fmt(row['p95_ms'])} |"
        )
    if not report["metrics"]["nodes"]:
        lines.append("| （无） | 0 | null | null | null |")

    lines += ["", "## 工具", "", "| 工具 | calls | failure_rate | p95_ms |", "|---|---|---|---|"]
    for name, row in report["metrics"]["tools"].items():
        lines.append(
            f"| {name} | {row['calls']} | {_fmt(row['failure_rate'])} | {_fmt(row['p95_ms'])} |"
        )
    if not report["metrics"]["tools"]:
        lines.append("| （无） | 0 | null | null |")

    lines += [
        "",
        "## 按 agent",
        "",
        "| agent_id | traces | calls | prompt | completion | cost_cny |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in report["metrics"]["by_agent"].items():
        lines.append(
            f"| {name} | {row['traces']} | {row['calls']} | {row['prompt_tokens']} | "
            f"{row['completion_tokens']} | {_fmt(row['cost_cny'])} |"
        )
    lines += ["", f"usage：llm_calls=0（本脚本不调用 LLM），budget_section={report['budget_section']}", ""]
    return "\n".join(lines)


def build_report(records: list[dict], *, run_id: str, filters: dict, files: int, malformed: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "filters": filters,
        "inputs": {"files": files, "records": len(records), "malformed_lines": malformed},
        "metrics": aggregate(records),
        "budget_section": "4.6#4",
    }


def _sample(records: list[dict], n: int | None) -> list[dict]:
    """等距抽样（确定性），n 大于总数时原样返回。"""
    if not n or n >= len(records):
        return records
    if n <= 0:
        return []
    step = len(records) / n
    return [records[min(len(records) - 1, int(i * step))] for i in range(n)]


def _build_manifest(
    *, run_id: str, command: str, started_at: str, finished_at: str,
    trace_sha: str, report_sha: str | None, files: list[str] | None = None,
) -> dict:
    """§8.3 manifest：本脚本不调 LLM，usage 固定为 0。"""
    usage = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "judge_cache_hits": 0,
        "judge_cache_misses": 0,
        "budget_prompt_tokens": 0,
        "within_budget": True,  # 0 <= 0，由脚本自身比较得出
    }
    return {
        "run_id": run_id,
        "command": command,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "started_at": started_at,
        "finished_at": finished_at,
        "env": {
            "python": sys.version.split()[0],
            "database": None,
            "llm_model": None,
            "embedding_model": None,
            "rerank_model": None,
        },
        # R8：输入文件清单 + 拼接哈希，审查方可在仓库内找到同一份 JSONL
        "inputs": {"trace_sha256": trace_sha, "files": list(files or [])},
        "outputs": {"report_sha256": report_sha, "cases_sha256": None},
        "usage": usage,
        "budget_section": "4.6#4",
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="从 trace JSONL 聚合可观测报告（不调 LLM）")
    parser.add_argument("--date", help="单日 YYYY-MM-DD")
    parser.add_argument("--since", help="起始日期（含）")
    parser.add_argument("--until", help="结束日期（含）")
    parser.add_argument("--out", help="输出 JSON 路径（默认 eval/report/trace-<run_id>.json）")
    parser.add_argument("--sample", type=int, default=None, help="等距抽样条数")
    parser.add_argument("--limit", type=int, default=None, help="最多读取条数（按时间序）")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    args = parser.parse_args(argv)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    trace_dir = Path(args.trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = ROOT / trace_dir

    files = select_files(trace_dir=trace_dir, date=args.date, since=args.since, until=args.until)
    records, malformed = load_records(files)
    # 按开始时间稳定排序后套用 --limit / --sample
    records.sort(key=lambda r: str(r.get("started_at") or ""))
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    records = _sample(records, args.sample)
    trace_sha = _sha256_bytes([p.read_bytes() for p in files if p.exists()])

    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"trace-{run_id}.json"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(
        records,
        run_id=run_id,
        filters={"date": args.date, "since": args.since, "until": args.until,
                 "sample": args.sample, "limit": args.limit},
        files=len(files),
        malformed=malformed,
    )
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    manifest = _build_manifest(
        run_id=run_id,
        command="uv run python scripts/trace_report.py " + " ".join(argv),
        started_at=started_at,
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        trace_sha=trace_sha,
        report_sha=_sha256_file(out_path),
        files=[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in files],
    )
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    m = report["metrics"]
    print("---- trace 报告 ----")
    print(f"traces        : {m['traces']}")
    print(f"error_rate    : {_fmt(m['error_rate'])}  degraded_rate: {_fmt(m['degraded_rate'])}")
    print(f"p50/p95 total : {_fmt(m['p50_total_ms'])} / {_fmt(m['p95_total_ms'])} ms")
    print(f"p50/p95 ttft  : {_fmt(m['p50_first_token_ms'])} / {_fmt(m['p95_first_token_ms'])} ms")
    print(f"llm cost_cny  : {_fmt(m['totals']['cost_cny'])}")
    print(f"报告          : {out_path}")
    print(f"markdown      : {md_path}")
    print(f"manifest      : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
