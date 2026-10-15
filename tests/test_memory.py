"""记忆层单测：namespace 纯度、key 粒度、读写隔离（接缝 2 / 4）。"""

from langgraph.store.memory import InMemoryStore

from app.memory.namespace import memory_namespace
from app.memory.store import MemoryStore, _memory_key


def test_namespace_depends_only_on_agent_id():
    assert memory_namespace(1) == ("agent:1", "memories")


def test_namespace_has_no_session_leak():
    # 接缝 2：namespace 不得掺入 session / 群聊上下文
    ns = memory_namespace(7)
    assert all("session" not in part for part in ns)


def test_memory_key_is_stable_and_content_scoped():
    assert _memory_key(1, "你好") == _memory_key(1, "你好")
    assert _memory_key(1, "你好") != _memory_key(1, "再见")  # 内容不同 → key 不同
    assert _memory_key(1, "你好") != _memory_key(2, "你好")  # agent 不同 → key 不同
    assert len(_memory_key(1, "你好")) == 32


async def test_put_then_search_roundtrip():
    store = MemoryStore(InMemoryStore())
    await store.put(1, "客人叫阿飞", role="user")
    assert await store.search(1) == ["客人叫阿飞"]


async def test_search_isolated_per_agent():
    store = MemoryStore(InMemoryStore())
    await store.put(1, "agent1 的记忆")
    await store.put(2, "agent2 的记忆")
    assert await store.search(1) == ["agent1 的记忆"]
    assert await store.search(2) == ["agent2 的记忆"]


async def test_same_content_written_twice_is_single_entry():
    # 粒度 = (agent_id, content)：同内容重复写不产生新条目
    store = MemoryStore(InMemoryStore())
    await store.put(1, "同一句话")
    await store.put(1, "同一句话")
    assert await store.search(1) == ["同一句话"]


async def test_summary_roundtrip_is_scoped_by_session():
    store = MemoryStore(InMemoryStore())
    await store.put_summary(1, 7, "会话7的摘要")
    assert await store.get_summary(1, 7) == "会话7的摘要"
    assert await store.get_summary(1, 8) == ""


async def test_summary_is_excluded_from_memory_search():
    # 摘要单独注入，不应出现在语义记忆召回里
    store = MemoryStore(InMemoryStore())
    await store.put(1, "普通记忆")
    await store.put_summary(1, 1, "会话摘要")
    assert await store.search(1) == ["普通记忆"]

