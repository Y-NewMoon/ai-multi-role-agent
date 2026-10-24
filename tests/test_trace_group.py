"""群聊 trace：每个发言人一条、共享 parent_trace_id（V2.4 §2.3 / D2）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.supervisor import GroupMember, Supervisor
from app.observability.trace import TraceStore, configure_trace_store


class _StubGraph:
    async def astream(self, state, *, config, stream_mode):
        yield ("values", {"replies": ["回复"], "reply": "回复"})


class _FixedDecisionLLM:
    """决策恒返回 1：验证"连续同人强制换人"下的发言人顺序。"""

    def __init__(self, decision: str = "1") -> None:
        self._decision = decision
        self.decisions = 0

    async def generate(self, messages):
        self.decisions += 1
        return self._decision

    async def astream(self, messages):  # 不被调用
        yield ""


def _members(*ids: int) -> list[GroupMember]:
    return [
        GroupMember(agent_id=i, name=f"角色{i}", persona_prompt="人设", background="")
        for i in ids
    ]


def _read_all(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.fixture
def trace_dir(tmp_path):
    store = TraceStore(directory=str(tmp_path), enabled=True, sample_rate=1.0)
    configure_trace_store(store)
    try:
        yield tmp_path
    finally:
        configure_trace_store(None)


async def test_每个发言人一条trace且共享parent(trace_dir):
    sup = Supervisor(
        llm=_FixedDecisionLLM("1"),
        graph_factory=lambda: _StubGraph(),
        members=_members(1, 2, 3),
        max_turns=3,
    )
    events = [
        event
        async for event in sup.run(
            group_session_id=9, user_message="hi", parent_trace_id="p" * 32, turn=7
        )
    ]
    assert [d["agent_id"] for e, d in events if e == "speaker"] == [1, 2, 3]

    records = _read_all(trace_dir)
    assert len(records) == 3  # 每个发言人一条，无 group 汇总条
    assert [r["agent_id"] for r in records] == [1, 2, 3]
    assert [r["speaker_index"] for r in records] == [1, 2, 3]
    assert {r["mode"] for r in records} == {"group"}
    assert {r["parent_trace_id"] for r in records} == {"p" * 32}
    assert {r["group_session_id"] for r in records} == {9}
    assert {r["session_id"] for r in records} == {None}
    assert {r["turn"] for r in records} == {7}
    assert len({r["trace_id"] for r in records}) == 3


async def test_群聊自主闲聊mode与轮次(trace_dir):
    sup = Supervisor(
        llm=_FixedDecisionLLM(),
        graph_factory=lambda: _StubGraph(),
        members=_members(1, 2),
        max_turns=2,
    )
    async for _ in sup.run_idle(
        group_session_id=5,
        max_total=4,
        turn_max_messages=2,
        parent_trace_id="q" * 32,
    ):
        pass

    records = _read_all(trace_dir)
    assert len(records) == 2
    assert {r["mode"] for r in records} == {"group_idle"}
    assert {r["turn"] for r in records} == {None}
    assert [r["speaker_index"] for r in records] == [1, 2]
    assert {r["parent_trace_id"] for r in records} == {"q" * 32}
