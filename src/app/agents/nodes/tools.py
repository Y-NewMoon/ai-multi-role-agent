"""节点：执行工具调用并把结构化结果回灌（契约 §2.4）。

- 读取 `state.tool_calls`，对每个 call 调 `ToolExecutor.execute`；
- 单轮多个工具按顺序执行，一个失败不阻断其他（各自结构化返回）；
- 返回 `tool_results` + `tool_rounds + 1`，并清空 `tool_calls`。
"""

import logging
import uuid

from app.agents.state import RolePlayState
from app.skills.base import ToolCall, ToolContext
from app.skills.executor import ToolExecutor

logger = logging.getLogger(__name__)


def _session_id(state: RolePlayState) -> int:
    """群聊的 session_id 形如 `g{id}`（非 int），工具上下文回落 0（不误用业务表主键）。"""
    value = state.get("session_id")
    return value if isinstance(value, int) else 0


async def execute_tools(
    state: RolePlayState,
    executor: ToolExecutor,
    *,
    allowed_skills: list[str] | None = None,
) -> dict:
    calls = list(state.get("tool_calls") or [])
    rounds = int(state.get("tool_rounds") or 0)
    allowed = tuple(state.get("allowed_skills") or allowed_skills or [])
    ctx = ToolContext(
        agent_id=int(state.get("agent_id") or 0),
        session_id=_session_id(state),
        allowed_skills=allowed,
    )

    records: list[dict] = []
    for raw in calls:
        call = ToolCall(
            id=str(raw.get("id") or uuid.uuid4().hex),
            name=str(raw.get("name") or ""),
            arguments=dict(raw.get("arguments") or {}),
        )
        result = await executor.execute(call, ctx)  # 绝不抛业务异常
        records.append(
            {
                "tool_call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "ok": result.ok,
                "error_type": result.error_type,
                "message": result.message,
                "data": result.data,
                "latency_ms": result.latency_ms,
                "cached": result.cached,
                "tool_round": rounds + 1,
                # 标记属于哪一轮请求：tool_results 由 operator.add 累积且经 checkpointer
                # 跨轮保留，生成节点据此只认本轮记录，避免上一轮的工具结果永久占用轮数。
                "tool_turn": state.get("tool_turn") or "",
            }
        )
    return {"tool_results": records, "tool_calls": [], "tool_rounds": rounds + 1}
