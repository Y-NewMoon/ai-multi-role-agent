"""V2.6 回归：跨轮工具状态隔离（FL-2026-10-24-13）。

`tool_rounds` 是普通 int 字段、`tool_results` 是 `operator.add` 累积字段，二者都会被
Postgres checkpointer 跨轮保留。若服务层不逐轮重置，会话累计到 `max_tool_rounds` 后
`forced=True` → `tool_choice="none"`，该会话将**永久失去工具能力**（时间答错、提醒建不了）。
"""

from app.agents.nodes.generate import _turn_tool_results
from app.services.chat_service import ChatContext, ChatService


def _context() -> ChatContext:
    return ChatContext(
        session_id=37,
        agent_id=3,
        agent_name="四季夏目",
        persona_prompt="茶艺师。",
        background="古镇",
        allowed_skills=("get_current_time", "set_reminder"),
    )


def test_state_in_resets_tool_state_every_turn():
    first = ChatService._state_in(_context(), "第一轮")
    second = ChatService._state_in(_context(), "第二轮")

    for state in (first, second):
        assert state["tool_rounds"] == 0
        assert state["tool_calls"] == []
        assert state["tool_turn"]

    # 每轮一个独立标记：上一轮遗留的 tool_results 不会被当成本轮结果
    assert first["tool_turn"] != second["tool_turn"]


def test_turn_tool_results_only_returns_current_turn():
    state = {
        "tool_turn": "turn-b",
        "tool_results": [
            {"name": "old_time", "tool_turn": "turn-a"},
            {"name": "new_reminder", "tool_turn": "turn-b"},
            {"name": "new_time", "tool_turn": "turn-b"},
        ],
    }

    assert [r["name"] for r in _turn_tool_results(state)] == ["new_reminder", "new_time"]
