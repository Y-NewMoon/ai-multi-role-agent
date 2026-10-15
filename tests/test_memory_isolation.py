"""真库记忆隔离测试（方案 3.8 · 接缝 1/2）：namespace 只由 agent_id 决定，thread 由 session 决定。

必须跑真库 `roleplay_test`（InMemoryStore 测不出真实 schema 与 pgvector 索引行为）。
每个用例用唯一 agent/session id，跑完清理 store namespace、checkpoint thread 与业务表行。
"""

import uuid

from langchain_core.messages import HumanMessage
from sqlalchemy import delete

from app.agents.graph import build_graph
from app.db.models import Agent, Message
from app.db.models import Session as ChatSession
from app.memory.namespace import memory_namespace
from app.memory.store import MemoryStore
from app.services.chat_service import ChatService


class _FakeLLM:
    """固定回复的 LLM 替身（仅 generate 节点会调用，M3 起走 astream）。"""

    def __init__(self, reply: str = "好的，客人。") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def generate(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply

    async def astream(self, messages: list[dict]):
        self.calls.append(messages)
        yield self.reply


def _agent_id() -> str:
    return f"it-{uuid.uuid4().hex[:8]}"


def _rand_int() -> int:
    """随机正整数，隔离本次用例的记忆/业务行，避免污染。"""
    return int(uuid.uuid4().hex[:8], 16)


def _state(message: str, *, agent_id: int, session_id: int) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "agent_id": agent_id,
        "session_id": session_id,
        "agent_name": "测试角色",
        "persona_prompt": "测试人设。",
        "background": "测试背景",
    }


async def _cleanup(store, agent_id) -> None:
    ns = memory_namespace(agent_id)
    for item in await store.asearch(ns, limit=1000):
        await store.adelete(item.namespace, item.key)


async def test_real_store_memories_isolated_between_agents(real_store):
    a, b = _agent_id(), _agent_id()
    ms = MemoryStore(real_store)
    try:
        await ms.put(a, "A 的专属记忆")
        await ms.put(b, "B 的专属记忆")
        assert await ms.search(a) == ["A 的专属记忆"]
        assert await ms.search(b) == ["B 的专属记忆"]
    finally:
        await _cleanup(real_store, a)
        await _cleanup(real_store, b)


async def test_real_store_semantic_search_hits_vector_index(real_store):
    a = _agent_id()
    ms = MemoryStore(real_store)
    try:
        await ms.put(a, "客人喜欢喝龙井")
        await ms.put(a, "客人生日在三月")
        hits = await ms.search(a, query="他平时爱喝什么茶")
        assert "客人喜欢喝龙井" in hits  # 真 pgvector 索引下 top-k 能取回语义记忆
    finally:
        await _cleanup(real_store, a)


async def test_real_store_memory_shared_across_sessions_of_same_agent(real_store):
    # 接缝 2：namespace 不掺 session，同一角色跨会话共享同一份记忆
    a = _agent_id()
    ms = MemoryStore(real_store)
    try:
        await ms.put(a, "跨会话共享的记忆")
        assert "跨会话共享的记忆" in await ms.search(a, query="随便问问")
    finally:
        await _cleanup(real_store, a)


async def test_real_store_summary_scoped_by_session(real_store):
    a = _agent_id()
    ms = MemoryStore(real_store)
    try:
        await ms.put_summary(a, 1, "会话1摘要")
        await ms.put_summary(a, 2, "会话2摘要")
        assert await ms.get_summary(a, 1) == "会话1摘要"
        assert await ms.get_summary(a, 2) == "会话2摘要"
        # 摘要单独注入，不参与语义记忆召回
        assert await ms.search(a, query="摘要") == []
    finally:
        await _cleanup(real_store, a)


async def test_real_checkpoint_threads_isolated_across_sessions(real_store, real_checkpointer):
    # 接缝 1：thread_id = f"{session_id}:{agent_id}"；同一角色的两 session 各持独立历史，互不覆盖
    agent_id = _rand_int()
    s1, s2 = agent_id + 1, agent_id + 2
    thread1, thread2 = f"{s1}:{agent_id}", f"{s2}:{agent_id}"
    graph = build_graph(
        llm=_FakeLLM("收到"),
        memory_store=MemoryStore(real_store),
        checkpointer=real_checkpointer,
        store=real_store,
    )
    cfg1 = {"configurable": {"thread_id": thread1}}
    cfg2 = {"configurable": {"thread_id": thread2}}
    try:
        await graph.ainvoke(_state("会话一的秘密", agent_id=agent_id, session_id=s1), config=cfg1)
        await graph.ainvoke(_state("会话二的话", agent_id=agent_id, session_id=s2), config=cfg2)

        m1 = [m.content for m in (await graph.aget_state(cfg1)).values["messages"]]
        m2 = [m.content for m in (await graph.aget_state(cfg2)).values["messages"]]
        assert "会话一的秘密" in m1
        assert "会话二的话" not in m1  # 线程间不串历史
        assert "会话二的话" in m2
        assert "会话一的秘密" not in m2
    finally:
        await real_checkpointer.adelete_thread(thread1)
        await real_checkpointer.adelete_thread(thread2)
        await _cleanup(real_store, agent_id)


async def test_real_service_uses_session_agent_and_keeps_memory_separate(
    real_store, real_checkpointer, real_session_factory
):
    # 接缝 2 / 5：ChatService 以 session.agent_id 为准；A 的记忆对 B 的会话不可读、也不被串写
    llm = _FakeLLM("回复")
    ms = MemoryStore(real_store)
    service = ChatService(
        graph=build_graph(
            llm=llm,
            memory_store=ms,
            checkpointer=real_checkpointer,
            store=real_store,
        ),
        session_factory=real_session_factory,
        memory_store=ms,
    )

    # 造两个角色，仅 B 拥有一个 session
    async with real_session_factory() as db:
        a = Agent(name="角色A", persona_prompt="A 的人设", background="A 的背景")
        b = Agent(name="角色B", persona_prompt="B 的人设", background="B 的背景")
        db.add_all([a, b])
        await db.flush()  # 取回自增 id
        b_session = ChatSession(agent_id=b.id)
        db.add(b_session)
        await db.commit()
        a_id, b_id, b_sid = a.id, b.id, b_session.id

    await ms.put(a_id, "A 的机密：口令 123")
    thread = f"{b_sid}:{b_id}"
    try:
        resp = await service.chat(session_id=b_sid, message="你还记得我吗")
        assert resp.agent_id == b_id  # 以 session.agent_id 为准，而非任何外部传入

        # 本轮记忆落到 B 的 namespace，未串写到 A
        assert any("你还记得我吗" in h for h in await ms.search(b_id, query="你还记得我吗"))
        assert all("你还记得我吗" not in h for h in await ms.search(a_id, query="你还记得我吗"))
        # A 的机密没有出现在 B 收到的 prompt 里（未被跨角色召回）
        assert all("A 的机密" not in m["content"] for m in llm.calls[-1])
    finally:
        async with real_session_factory() as db:
            await db.execute(delete(Message).where(Message.session_id == b_sid))
            await db.execute(delete(ChatSession).where(ChatSession.id == b_sid))
            await db.execute(delete(Agent).where(Agent.id.in_([a_id, b_id])))
            await db.commit()
        await real_checkpointer.adelete_thread(thread)
        await _cleanup(real_store, a_id)
        await _cleanup(real_store, b_id)
