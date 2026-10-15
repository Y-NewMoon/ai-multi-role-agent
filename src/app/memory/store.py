"""LangGraph Store 的薄封装——记忆读写的唯一入口。

接缝 5（方案 3.11A#5）：图内节点不访问业务表，记忆读写一律经注入的 Store。
会话隔离（接缝 2）：namespace 只由 agent_id 决定，不掺 session。

M2 起：
- 语义记忆：`search` 带 query 时走向量索引 top-k（见 main.py 注入的 index）；
- 情景记忆：滚动摘要以 key=`summary:{session_id}` 落在同一 namespace（方案 3.10），
  由 `put_summary` / `get_summary` 读写，并在 `search` 中剔除，避免重复占用 prompt 预算。
"""

import hashlib

from langgraph.store.base import BaseStore

from app.memory.namespace import memory_namespace

SUMMARY_KEY_PREFIX = "summary:"
# 语义召回时按 limit+overscan 取，再剔除摘要条目，保证过滤后仍尽量凑够 limit。
_SEARCH_OVERSCAN = 10


def _memory_key(agent_id: int | str, content: str) -> str:
    """以 (agent_id, content) 为单位生成稳定 key（接缝 4）。"""
    digest = hashlib.sha1(f"{agent_id}:{content}".encode("utf-8")).hexdigest()
    return digest[:32]


def _summary_key(session_id: int | str) -> str:
    """摘要 key 形如 summary:{session_id}（方案 3.10）：同会话重复摘要覆盖同一条。"""
    return f"{SUMMARY_KEY_PREFIX}{session_id}"


class MemoryStore:
    """封装 LangGraph Store，对外只暴露记忆语义。"""

    def __init__(self, store: BaseStore) -> None:
        self._store = store

    async def put(
        self, agent_id: int | str, content: str, *, kind: str = "memory", **metadata: object
    ) -> None:
        """写入一条记忆，粒度 = (agent_id, content)。"""
        ns = memory_namespace(agent_id)
        value = {"text": content, "kind": kind, **metadata}
        await self._store.aput(ns, _memory_key(agent_id, content), value)

    async def put_summary(self, agent_id: int | str, session_id: int | str, text: str) -> None:
        """写入/覆盖该会话的滚动摘要（同 namespace，摘要文本同样会被向量化）。"""
        ns = memory_namespace(agent_id)
        await self._store.aput(ns, _summary_key(session_id), {"text": text, "kind": "summary"})

    async def get_summary(self, agent_id: int | str, session_id: int | str) -> str:
        """读取该会话的滚动摘要，无则返回空串。"""
        ns = memory_namespace(agent_id)
        item = await self._store.aget(ns, _summary_key(session_id))
        if item is None:
            return ""
        return str(item.value.get("text", ""))

    async def search(
        self, agent_id: int | str, query: str | None = None, limit: int = 5
    ) -> list[str]:
        """召回该角色的语义记忆文本（摘要条目单独注入，故在此剔除）。"""
        ns = memory_namespace(agent_id)
        items = await self._store.asearch(ns, query=query, limit=limit + _SEARCH_OVERSCAN)
        out: list[str] = []
        for item in items:
            if item.value.get("kind") == "summary":
                continue
            text = item.value.get("text")
            if text:
                out.append(str(text))
            if len(out) >= limit:
                break
        return out

    async def delete(self, agent_id: int | str, content: str) -> None:
        """按内容精确删除一条语义记忆（撤回消息时用；不影响该角色其余记忆）。

        key 由 (agent_id, content) 决定，故只删属于该内容的条目。
        """
        ns = memory_namespace(agent_id)
        await self._store.adelete(ns, _memory_key(agent_id, content))

    async def delete_all(self, agent_id: int | str) -> None:
        """清空该角色的整个 namespace（语义记忆 + 全部会话摘要）。

        用于删除角色时的级联清理；向量随 store_vectors 的 ON DELETE CASCADE 一并删除。
        """
        ns = memory_namespace(agent_id)
        # 单次 asearch 有 limit 上限（默认最多 1000 条），必须分页循环删至空，
        # 否则条目超过上限时会残留（实测 1005 条 → 残留 5 条）。
        while True:
            items = await self._store.asearch(ns, limit=1000)
            if not items:
                break
            for item in items:
                await self._store.adelete(item.namespace, item.key)

    async def delete_summary(self, agent_id: int | str, session_id: int | str) -> None:
        """删除某会话的滚动摘要（删除会话时调用；语义记忆属角色，不在此删）。"""
        ns = memory_namespace(agent_id)
        await self._store.adelete(ns, _summary_key(session_id))
