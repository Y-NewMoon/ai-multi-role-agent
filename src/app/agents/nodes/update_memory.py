"""节点：写入记忆 + 触发滚动摘要（M2 / M5）。

接缝 4（方案 3.11A#4）：以 (agent_id, content) 为单位写入，不以 session 为单位。
接缝 2：写入该 agent 自己的 namespace（经注入的 Store，图内不碰业务表）。
方案 3.10：工作记忆超过 trigger_rounds 轮时，把最旧 chunk_rounds 轮压缩为 1 条摘要
（落 key=summary:{session_id}），并用 RemoveMessage 裁剪 checkpointer 侧窗口；
`messages` 表仍保留全量用于展示（由服务层写，图内不写业务表）。

M5：角色一轮可连发多条（`state["replies"]`）。记忆按「条」写入（每条一条记忆）；
裁剪窗口按「轮」计数（一轮 = 一条用户消息 + 其后的全部角色回复），不再假设每轮恰 2 条。
"""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from app.agents.state import RolePlayState
from app.llm.base import LLMProvider
from app.memory.store import MemoryStore
from app.memory.summarizer import summarize


def _last_human(messages: list):
    """本轮用户消息（最后一条 HumanMessage）。"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return None


def _round_count(messages: list) -> int:
    """轮数 = 用户消息条数（每轮恰有一条用户消息）。"""
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _old_rounds(messages: list, chunk_rounds: int) -> list:
    """截取最旧 chunk_rounds 轮（含各轮全部角色回复），从第 chunk_rounds+1 条用户消息处断开。"""
    seen = 0
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            seen += 1
            if seen == chunk_rounds + 1:
                return messages[:i]
    return list(messages)


def _is_idle_message(m) -> bool:
    """是否为自主闲聊消息（generate 打的额外标记）。"""
    return bool((getattr(m, "additional_kwargs", None) or {}).get("idle"))


async def update_memory(
    state: RolePlayState,
    memory_store: MemoryStore,
    llm: LLMProvider,
    *,
    trigger_rounds: int = 10,
    chunk_rounds: int = 5,
) -> dict:
    agent_id = state["agent_id"]
    session_id = state["session_id"]
    messages = list(state.get("messages", []))
    is_idle = bool(state.get("idle_scope"))

    # 1) 写入本轮新记忆（粒度 = agent_id + content，同内容覆盖）
    #    M5：一轮可含多条角色回复，逐条写入；无 replies 时回退为最后一条 AI（兼容旧路径）。
    #    自主闲聊轮没有真实用户发言，跳过 human 写入，避免把历史用户消息当本轮重复入库。
    human = _last_human(messages)
    if not is_idle and human is not None and str(human.content).strip():
        await memory_store.put(agent_id, str(human.content).strip(), role="user")

    replies = [str(r).strip() for r in (state.get("replies") or [])]
    replies = [r for r in replies if r]
    if not replies:
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai is not None and str(last_ai.content).strip():
            replies = [str(last_ai.content).strip()]
    for content in replies:
        await memory_store.put(agent_id, content, role="assistant")

    # 2) 超阈值则滚动摘要：压缩最旧 chunk_rounds 轮，并裁剪工作记忆窗口
    if _round_count(messages) <= trigger_rounds:
        return {}

    old_chunk = _old_rounds(messages, chunk_rounds)
    # 阻止式：摘要输入剔除自主闲聊消息（闲聊永不被并入摘要），但裁剪窗口仍包含它们。
    summarize_input = [m for m in old_chunk if not _is_idle_message(m)]
    existing = await memory_store.get_summary(agent_id, session_id)
    summary_text = await summarize(llm, summarize_input, existing_summary=existing)
    if not summary_text:
        return {}

    await memory_store.put_summary(agent_id, session_id, summary_text)
    removals = [RemoveMessage(id=m.id) for m in old_chunk if m.id]
    return {"messages": removals} if removals else {}
