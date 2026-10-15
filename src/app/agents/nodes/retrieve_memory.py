"""节点：从该角色 namespace 召回语义记忆，并取出本会话滚动摘要。

M2：以本轮最后一条用户消息为 query 做 top-k 向量召回（方案 3.2 语义记忆）；
摘要（方案 3.10）单独取出，供 compose_prompt 高优先级注入。
"""

from langchain_core.messages import HumanMessage

from app.agents.state import RolePlayState
from app.memory.retriever import retrieve
from app.memory.store import MemoryStore


def _latest_user_query(messages: list) -> str | None:
    """取本轮最后一条用户消息作为检索 query；无则返回 None（退化为列表召回）。"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            text = str(m.content).strip()
            if text:
                return text
    return None


async def retrieve_memory(state: RolePlayState, memory_store: MemoryStore) -> dict:
    messages = state.get("messages", [])
    query = _latest_user_query(messages)
    memories = await retrieve(memory_store, state["agent_id"], query=query, limit=5)
    summary = await memory_store.get_summary(state["agent_id"], state["session_id"])
    return {"recalled_memories": memories, "summary": summary}
