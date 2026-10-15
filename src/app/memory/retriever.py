"""记忆召回（方案 3.2 语义记忆）。

M2 起：query 非空时经注入 Store 的向量索引做 top-k 语义召回；query 为空则退化为列表召回。
"""

from app.memory.store import MemoryStore


async def retrieve(
    memory_store: MemoryStore,
    agent_id: int | str,
    query: str | None = None,
    limit: int = 5,
) -> list[str]:
    return await memory_store.search(agent_id, query=query, limit=limit)
