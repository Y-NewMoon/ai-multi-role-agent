"""T1：trace core 单测（schema / 采样 / fail-open / 并发隔离 / 截断）。"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.observability import trace as trace_mod
from app.observability.trace import (
    TraceStore,
    configure_trace_store,
    current_trace,
    is_degraded,
    mark_degraded,
    reset_trace,
    start_trace,
)


@pytest.fixture
def trace_dir(tmp_path):
    return tmp_path / "traces"


@pytest.fixture
def store(trace_dir):
    store = TraceStore(directory=str(trace_dir), enabled=True, sample_rate=1.0)
    configure_trace_store(store)
    yield store
    configure_trace_store(None)


def _read_records(trace_dir) -> list[dict]:
    records: list[dict] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def test_trace_schema_fields(store, trace_dir):
    recorder, token = start_trace(
        mode="private", session_id=12, agent_id=1, turn=7
    )
    try:
        assert current_trace() is recorder
        recorder.node("retrieve_memory", 42)
        recorder.node("generate", 1800)
        recorder.retrieval(
            lore_ids=["12:3"],
            memory_count=4,
            rerank_top=3,
            rewrite_used=True,
            rewrite_ms=300,
        )
        recorder.tool("get_current_time", True, 80)
        recorder.llm(
            model="deepseek-chat",
            prompt_tokens=3100,
            completion_tokens=210,
            cost_cny=0.00788,
        )
        recorder.set_budget(prompt_token_budget=8000, estimated_prompt_tokens=3100)
        recorder.first_token()
        recorder.finish()
    finally:
        reset_trace(token)

    records = _read_records(trace_dir)
    assert len(records) == 1
    rec = records[0]

    assert rec["trace_version"] == "1.0"
    assert rec["prompt_version"]
    assert rec["git_sha"]
    assert rec["mode"] == "private"
    assert rec["session_id"] == 12
    assert rec["group_session_id"] is None
    assert rec["agent_id"] == 1
    assert rec["turn"] == 7
    assert rec["speaker_index"] is None
    assert rec["parent_trace_id"] is None
    assert rec["first_token_ms"] is not None
    assert rec["total_ms"] >= 0
    assert rec["nodes"] == [
        {"name": "retrieve_memory", "ms": 42, "ok": True},
        {"name": "generate", "ms": 1800, "ok": True},
    ]
    assert rec["retrieval"] == {
        "lore_ids": ["12:3"],
        "memory_count": 4,
        "rerank_top": 3,
        "rewrite": {"used": True, "ms": 300},
    }
    assert rec["tools"] == [
        {"name": "get_current_time", "ok": True, "ms": 80, "error_type": None}
    ]
    assert rec["llm"]["calls"] == 1
    assert rec["llm"]["prompt_tokens"] == 3100
    assert rec["llm"]["completion_tokens"] == 210
    assert rec["llm"]["cost_cny"] == 0.00788
    assert rec["llm"]["usage_source"] == "estimator"
    assert rec["budget"] == {
        "prompt_token_budget": 8000,
        "estimated_prompt_tokens": 3100,
        "over_budget": False,
    }
    assert rec["memory_stats"] is None
    assert rec["error"] is None
    assert rec["degraded"] is False
    assert rec["started_at"].endswith("+08:00")


def test_noop_recorder_when_sample_rate_zero(trace_dir):
    configure_trace_store(
        TraceStore(directory=str(trace_dir), enabled=True, sample_rate=0.0)
    )
    try:
        recorder, token = start_trace(
            mode="private", session_id=1, agent_id=1, turn=1
        )
        try:
            recorder.node("generate", 1)
            recorder.llm(
                model="m", prompt_tokens=1, completion_tokens=1, cost_cny=0.0
            )
            recorder.finish()
        finally:
            reset_trace(token)
    finally:
        configure_trace_store(None)
    assert _read_records(trace_dir) == []


def test_noop_recorder_when_disabled(trace_dir):
    configure_trace_store(
        TraceStore(directory=str(trace_dir), enabled=False, sample_rate=1.0)
    )
    try:
        recorder, token = start_trace(
            mode="idle", session_id=2, agent_id=1, turn=None
        )
        recorder.finish()
        reset_trace(token)
    finally:
        configure_trace_store(None)
    assert _read_records(trace_dir) == []


def test_write_failure_is_fail_open(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("x", encoding="utf-8")  # 同名路径是文件，mkdir 必失败
    configure_trace_store(
        TraceStore(directory=str(blocker / "sub"), enabled=True, sample_rate=1.0)
    )
    try:
        recorder, token = start_trace(
            mode="private", session_id=1, agent_id=1, turn=1
        )
        recorder.node("generate", 10)
        recorder.finish()  # 不得抛异常
        reset_trace(token)
    finally:
        configure_trace_store(None)


def test_error_message_truncated(store, trace_dir):
    recorder, token = start_trace(
        mode="private", session_id=1, agent_id=1, turn=1
    )
    try:
        recorder.finish(error={"type": "timeout", "message": "x" * 500}, degraded=True)
    finally:
        reset_trace(token)
    rec = _read_records(trace_dir)[0]
    assert rec["error"]["type"] == "timeout"
    assert len(rec["error"]["message"]) == 200
    assert rec["degraded"] is True


def test_llm_calls_aggregated_and_cost_rounded(store, trace_dir):
    recorder, token = start_trace(
        mode="private", session_id=1, agent_id=1, turn=1
    )
    try:
        recorder.llm(
            model="deepseek-chat", prompt_tokens=1000, completion_tokens=10, cost_cny=0.0001
        )
        recorder.llm(
            model="deepseek-chat", prompt_tokens=2000, completion_tokens=20, cost_cny=0.0002
        )
        recorder.llm_wait(3)
        recorder.finish()
    finally:
        reset_trace(token)
    llm = _read_records(trace_dir)[0]["llm"]
    assert llm["calls"] == 2
    assert llm["prompt_tokens"] == 3000
    assert llm["completion_tokens"] == 30
    assert llm["cost_cny"] == 0.0003
    assert llm["wait_ms"] == 3


def test_finish_is_idempotent(store, trace_dir):
    recorder, token = start_trace(
        mode="private", session_id=1, agent_id=1, turn=1
    )
    try:
        recorder.finish()
        recorder.finish()
    finally:
        reset_trace(token)
    assert len(_read_records(trace_dir)) == 1


def test_trace_ids_unique(store, trace_dir):
    ids = []
    for _ in range(3):
        recorder, token = start_trace(
            mode="private", session_id=1, agent_id=1, turn=1
        )
        ids.append(recorder.trace_id)
        recorder.finish()
        reset_trace(token)
    assert len(set(ids)) == 3
    assert len({r["trace_id"] for r in _read_records(trace_dir)}) == 3


@pytest.mark.asyncio
async def test_concurrent_traces_do_not_leak(store, trace_dir):
    async def one(agent_id: int) -> None:
        recorder, token = start_trace(
            mode="private", session_id=agent_id * 10, agent_id=agent_id, turn=1
        )
        try:
            await asyncio.sleep(0)
            recorder.node(f"node_{agent_id}", agent_id)
            recorder.finish()
        finally:
            reset_trace(token)

    await asyncio.gather(*(one(i) for i in range(1, 21)))

    records = _read_records(trace_dir)
    assert len(records) == 20
    assert len({r["trace_id"] for r in records}) == 20
    for rec in records:
        assert rec["session_id"] == rec["agent_id"] * 10
        assert rec["nodes"] == [
            {"name": f"node_{rec['agent_id']}", "ms": rec["agent_id"], "ok": True}
        ]


def test_degraded_marker_lifecycle(store):
    recorder, token = start_trace(mode="private", session_id=1, agent_id=1, turn=1)
    try:
        assert is_degraded() is False
        mark_degraded()
        assert is_degraded() is True
    finally:
        reset_trace(token)
    assert is_degraded() is False


def test_current_trace_none_outside(store):
    recorder, token = start_trace(mode="private", session_id=1, agent_id=1, turn=1)
    reset_trace(token)
    assert current_trace() is None
    assert trace_mod.TRACE_VERSION == "1.0"
