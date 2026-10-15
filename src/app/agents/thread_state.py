"""撤回用：按「消息内容」读取/清理 checkpoint 工作记忆（不触碰业务表）。

记忆条目按内容成条存储，业务快照表只存内容、不存 checkpointer 的消息 id，
故这里以内容为锚点定位 thread 状态中的消息，并用 LangGraph `RemoveMessage` 原地删除。
只有仍在工作记忆窗口内的消息可被撤回；已归档进会话摘要的消息不在此窗口，无法定位。
"""

from typing import Any

from langchain_core.messages import RemoveMessage


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _messages_of(snapshot: Any) -> list:
    values = getattr(snapshot, "values", None) or {}
    return list(values.get("messages", []))


async def thread_message_contents(graph, thread_id: str) -> set[str]:
    """该 thread 当前工作记忆里所有消息的文本内容集合。"""
    snapshot = await graph.aget_state(_config(thread_id))
    return {str(getattr(m, "content", "")) for m in _messages_of(snapshot)}


async def remove_thread_messages(graph, thread_id: str, targets: set[str]) -> int:
    """从 thread 工作记忆删除内容命中 `targets` 的消息；返回实际删除条数。"""
    snapshot = await graph.aget_state(_config(thread_id))
    removes = [
        RemoveMessage(id=m.id)
        for m in _messages_of(snapshot)
        if getattr(m, "id", None) and str(getattr(m, "content", "")) in targets
    ]
    if removes:
        await graph.aupdate_state(_config(thread_id), {"messages": removes})
    return len(removes)


async def remove_thread_message_by_id(graph, thread_id: str, message_id: str) -> int:
    """按消息 id 精确删除工作记忆中的一条消息；返回删除条数（0 表示未找到）。

    与 `remove_thread_messages`（按内容命中，可能误删同内容的其它消息）不同，
    这里锚定注入时预置的消息 id，用于「流式请求中途失败」时回滚本轮用户消息。
    """
    snapshot = await graph.aget_state(_config(thread_id))
    ids = {getattr(m, "id", None) for m in _messages_of(snapshot)}
    if message_id not in ids:
        return 0
    await graph.aupdate_state(_config(thread_id), {"messages": [RemoveMessage(id=message_id)]})
    return 1
