"""trace_report 单测（V2.4 §2.5）：聚合口径 / 空输入 / 分位数 / 文件选择 / manifest。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import trace_report  # noqa: E402


def _record(**overrides) -> dict:
    base = {
        "trace_version": "1.0",
        "trace_id": "t",
        "parent_trace_id": None,
        "mode": "private",
        "session_id": 1,
        "group_session_id": None,
        "agent_id": 1,
        "turn": 1,
        "speaker_index": None,
        "started_at": "2026-10-21T21:00:00.000+08:00",
        "total_ms": 100,
        "first_token_ms": 50,
        "nodes": [{"name": "generate", "ms": 80, "ok": True}],
        "retrieval": {"lore_ids": [], "memory_count": 0, "rerank_top": None,
                      "rewrite": {"used": False, "ms": 0}},
        "tools": [],
        "llm": {"model": "m", "calls": 1, "prompt_tokens": 1000,
                "completion_tokens": 10, "cost_cny": 0.00208, "usage_source": "estimator"},
        "budget": {"prompt_token_budget": 8000, "estimated_prompt_tokens": 4000,
                   "over_budget": False},
        "memory_stats": None,
        "error": None,
        "degraded": False,
    }
    base.update(overrides)
    return base


def test_空输入输出零且指标为null():
    metrics = trace_report.aggregate([])
    assert metrics["traces"] == 0
    assert metrics["error_rate"] is None
    assert metrics["p50_total_ms"] is None
    assert metrics["p95_first_token_ms"] is None
    assert metrics["nodes"] == {}
    assert metrics["totals"]["cost_cny"] == 0.0


def test_分位数采用最近秩():
    assert trace_report.percentile([], 50) is None
    assert trace_report.percentile([10], 95) == 10
    assert trace_report.percentile([1, 2, 3, 4], 50) == 2
    assert trace_report.percentile([1, 2, 3, 4], 95) == 4


def test_聚合错误率降级率与成本():
    records = [
        _record(),
        _record(error={"type": "timeout", "message": "x"}, degraded=True,
                llm={"model": "m", "calls": 2, "prompt_tokens": 500,
                     "completion_tokens": 5, "cost_cny": 0.00104}),
    ]
    metrics = trace_report.aggregate(records)
    assert metrics["traces"] == 2
    assert metrics["error_rate"] == 0.5
    assert metrics["degraded_rate"] == 0.5
    assert metrics["totals"]["calls"] == 3
    assert metrics["totals"]["prompt_tokens"] == 1500
    assert metrics["totals"]["cost_cny"] == 0.00312


def test_节点与工具失败率():
    records = [
        _record(nodes=[{"name": "generate", "ms": 10, "ok": True},
                       {"name": "generate", "ms": 30, "ok": False}]),
        _record(nodes=[{"name": "compose_prompt", "ms": 5, "ok": True}],
                tools=[{"name": "get_current_time", "ok": False, "ms": 4,
                        "error_type": "tool_error"}]),
    ]
    metrics = trace_report.aggregate(records)
    assert metrics["nodes"]["generate"]["count"] == 2
    assert metrics["nodes"]["generate"]["failures"] == 1
    assert metrics["nodes"]["generate"]["fail_rate"] == 0.5
    assert metrics["tools"]["get_current_time"]["calls"] == 1
    assert metrics["tools"]["get_current_time"]["failure_rate"] == 1.0


def test_按agent与会话聚合且群聊用group键():
    records = [
        _record(agent_id=1, session_id=3),
        _record(agent_id=2, session_id=None, group_session_id=9, mode="group"),
    ]
    metrics = trace_report.aggregate(records)
    assert set(metrics["by_agent"]) == {"1", "2"}
    assert metrics["by_agent"]["2"]["calls"] == 1
    assert set(metrics["by_session"]) == {"3", "g9"}


def test_token段占比与超预算():
    records = [
        _record(budget={"prompt_token_budget": 1000, "estimated_prompt_tokens": 500,
                        "over_budget": False}),
        _record(budget={"prompt_token_budget": 1000, "estimated_prompt_tokens": 1500,
                        "over_budget": True}),
        _record(budget=None),
    ]
    segments = trace_report.aggregate(records)["token_segments"]
    assert segments["traces_with_budget"] == 2
    assert segments["mean_prompt_ratio"] == 1.0
    assert segments["over_budget_traces"] == 1
    assert segments["over_budget_rate"] == 0.5


def test_文件选择按日期与区间(tmp_path):
    for day in ("2026-10-20", "2026-10-21", "2026-10-22"):
        (tmp_path / f"{day}.jsonl").write_text("", encoding="utf-8")
    only = trace_report.select_files(trace_dir=tmp_path, date="2026-10-21",
                                      since=None, until=None)
    assert [p.stem for p in only] == ["2026-10-21"]
    ranged = trace_report.select_files(trace_dir=tmp_path, date=None,
                                       since="2026-10-20", until="2026-10-21")
    assert [p.stem for p in ranged] == ["2026-10-20", "2026-10-21"]


def test_损坏行被跳过并计数(tmp_path):
    path = tmp_path / "2026-10-21.jsonl"
    path.write_text(
        json.dumps(_record(), ensure_ascii=False) + "\n{bad json\n\n", encoding="utf-8"
    )
    records, malformed = trace_report.load_records([path])
    assert len(records) == 1
    assert malformed == 1


def test_等距抽样确定性():
    records = [{"i": i} for i in range(10)]
    sampled = trace_report._sample(records, 4)
    assert [r["i"] for r in sampled] == [0, 2, 5, 7]
    assert trace_report._sample(records, 20) == records


def test_main空目录写出零报告与manifest(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    out = tmp_path / "trace.json"
    code = trace_report.main(["--trace-dir", str(trace_dir), "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["metrics"]["traces"] == 0
    assert report["metrics"]["p95_total_ms"] is None
    assert out.with_suffix(".md").exists()

    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["usage"]["llm_calls"] == 0
    assert manifest["usage"]["prompt_tokens"] == 0
    assert manifest["usage"]["budget_prompt_tokens"] == 0
    assert manifest["usage"]["within_budget"] is True
    assert manifest["inputs"]["trace_sha256"]
    assert manifest["outputs"]["report_sha256"]
