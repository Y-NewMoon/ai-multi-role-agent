"""M3 CRUD 测试：角色/会话接口（路由）与级联删除（真库）。

路由层用替身服务验证状态码与透传；级联语义用真库 `roleplay_test` 验证
（业务表 + Store 记忆 + checkpoint thread 一并清理）。
"""

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from sqlalchemy import delete, select

from app.agents.graph import build_graph
from app.api.v1 import api_router
from app.core.error_handlers import register_exception_handlers
from app.core.errors import NotFoundError
from app.db.models import Agent, Message
from app.db.models import Session as ChatSession
from app.memory.namespace import memory_namespace
from app.memory.store import MemoryStore
from app.schemas.agent import AgentCreate, AgentUpdate
from app.schemas.session import SessionCreate, SessionUpdate
from app.services.agent_service import AgentService
from app.services.session_service import SessionService

# ============================ 路由层（替身服务） ============================


class _FakeAgentService:
    def __init__(self) -> None:
        self.deleted: list[int] = []

    async def list(self):
        return [{"id": 1, "name": "A", "persona_prompt": "p", "created_at": "2026-01-01T00:00:00Z"}]

    async def create(self, data):
        return {"id": 9, "name": data.name, "persona_prompt": data.persona_prompt,
                "created_at": "2026-01-01T00:00:00Z"}

    async def get(self, agent_id: int):
        raise NotFoundError(f"agent {agent_id} 不存在")

    async def update(self, agent_id: int, data):
        return {"id": agent_id, "name": "B", "persona_prompt": "p2",
                "created_at": "2026-01-01T00:00:00Z"}

    async def delete(self, agent_id: int):
        self.deleted.append(agent_id)


class _FakeSessionService(_FakeAgentService):
    async def list(self, *, agent_id=None):
        return [{"id": 3, "agent_id": 1, "created_at": "2026-01-01T00:00:00Z"}]

    async def create(self, data):
        return {"id": 3, "agent_id": data.agent_id, "created_at": "2026-01-01T00:00:00Z"}

    async def update(self, session_id: int, data):
        return {"id": session_id, "agent_id": 1, "title": data.title,
                "created_at": "2026-01-01T00:00:00Z"}


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    register_exception_handlers(app)
    app.state.agent_service = _FakeAgentService()
    app.state.session_service = _FakeSessionService()
    return app


async def _request(method: str, url: str, payload: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, json=payload)


async def test_agent_routes_status_codes():
    assert (await _request("GET", "/api/v1/agents")).status_code == 200
    created = await _request("POST", "/api/v1/agents", {"name": "A", "persona_prompt": "p"})
    assert created.status_code == 201 and created.json()["name"] == "A"
    assert (await _request("GET", "/api/v1/agents/404")).status_code == 404
    assert (await _request("PATCH", "/api/v1/agents/1", {"name": "B"})).status_code == 200
    assert (await _request("DELETE", "/api/v1/agents/1")).status_code == 204


async def test_agent_create_requires_name_and_persona():
    resp = await _request("POST", "/api/v1/agents", {"name": "A"})
    assert resp.status_code == 422  # 缺 persona_prompt


async def test_session_routes_status_codes():
    assert (await _request("GET", "/api/v1/sessions?agent_id=1")).status_code == 200
    created = await _request("POST", "/api/v1/sessions", {"agent_id": 1})
    assert created.status_code == 201 and created.json()["agent_id"] == 1
    assert (await _request("GET", "/api/v1/sessions/404")).status_code == 404
    assert (await _request("PATCH", "/api/v1/sessions/1", {"title": "t"})).status_code == 200
    assert (await _request("DELETE", "/api/v1/sessions/1")).status_code == 204


# ======================== 服务层 / 级联删除（真库） ========================


class _FakeLLM:
    """固定回复的 LLM 替身（M3 起 generate 节点走 astream）。"""

    async def generate(self, messages):
        return "好的"

    async def astream(self, messages):
        yield "好的"


async def test_agent_service_crud(real_session_factory, real_store, real_checkpointer):
    service = AgentService(
        session_factory=real_session_factory,
        memory_store=MemoryStore(real_store),
        checkpointer=real_checkpointer,
    )
    agent = await service.create(AgentCreate(name="四季夏目", persona_prompt="茶艺师", background="古镇"))
    try:
        assert await service.get(agent.id) is not None
        updated = await service.update(agent.id, AgentUpdate(name="改名"))
        assert updated.name == "改名"
        assert any(a.id == agent.id for a in await service.list())
        with pytest.raises(NotFoundError):
            await service.get(10**9)
    finally:
        async with real_session_factory() as db:
            await db.execute(delete(Agent).where(Agent.id == agent.id))
            await db.commit()


async def test_session_create_requires_existing_agent(real_session_factory, real_store, real_checkpointer):
    service = SessionService(
        session_factory=real_session_factory,
        memory_store=MemoryStore(real_store),
        checkpointer=real_checkpointer,
    )
    with pytest.raises(NotFoundError):
        await service.create(SessionCreate(agent_id=10**9))


async def test_agent_delete_cascades_business_store_and_checkpoint(
    real_session_factory, real_store, real_checkpointer
):
    ms = MemoryStore(real_store)
    agent_service = AgentService(
        session_factory=real_session_factory, memory_store=ms, checkpointer=real_checkpointer
    )
    session_service = SessionService(
        session_factory=real_session_factory, memory_store=ms, checkpointer=real_checkpointer
    )

    agent = await agent_service.create(AgentCreate(name="带会话的角色", persona_prompt="p"))
    session = await session_service.create(SessionCreate(agent_id=agent.id))

    # 铺数据：消息快照 / 角色级记忆 / 会话摘要 / checkpoint thread
    async with real_session_factory() as db:
        db.add(Message(session_id=session.id, role="user", content="你好"))
        await db.commit()
    await ms.put(agent.id, "角色级记忆")
    await ms.put_summary(agent.id, session.id, "会话摘要")
    thread = f"{session.id}:{agent.id}"
    graph = build_graph(
        llm=_FakeLLM(), memory_store=ms, checkpointer=real_checkpointer, store=real_store
    )
    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="你好")],
            "agent_id": agent.id,
            "session_id": session.id,
            "agent_name": "带会话的角色",
            "persona_prompt": "p",
            "background": "",
        },
        config={"configurable": {"thread_id": thread}},
    )

    await agent_service.delete(agent.id)

    async with real_session_factory() as db:
        assert await db.get(Agent, agent.id) is None
        assert await db.get(ChatSession, session.id) is None
        rows = (await db.execute(select(Message).where(Message.session_id == session.id))).scalars().all()
        assert rows == []
    assert await ms.search(agent.id) == []
    assert await ms.get_summary(agent.id, session.id) == ""
    state = await graph.aget_state({"configurable": {"thread_id": thread}})
    assert not state.values.get("messages")  # checkpoint thread 已清


async def test_session_delete_keeps_agent_and_semantic_memory(
    real_session_factory, real_store, real_checkpointer
):
    ms = MemoryStore(real_store)
    agent_service = AgentService(
        session_factory=real_session_factory, memory_store=ms, checkpointer=real_checkpointer
    )
    session_service = SessionService(
        session_factory=real_session_factory, memory_store=ms, checkpointer=real_checkpointer
    )

    agent = await agent_service.create(AgentCreate(name="保留角色", persona_prompt="p"))
    session = await session_service.create(SessionCreate(agent_id=agent.id))
    thread = f"{session.id}:{agent.id}"
    graph = build_graph(
        llm=_FakeLLM(), memory_store=ms, checkpointer=real_checkpointer, store=real_store
    )
    try:
        async with real_session_factory() as db:
            db.add(Message(session_id=session.id, role="user", content="你好"))
            await db.commit()
        await ms.put(agent.id, "角色级语义记忆")
        await ms.put_summary(agent.id, session.id, "会话摘要")
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="你好")],
                "agent_id": agent.id,
                "session_id": session.id,
                "agent_name": "保留角色",
                "persona_prompt": "p",
                "background": "",
            },
            config={"configurable": {"thread_id": thread}},
        )

        await session_service.delete(session.id)

        async with real_session_factory() as db:
            assert await db.get(ChatSession, session.id) is None
            assert await db.get(Agent, agent.id) is not None  # 角色保留
            rows = (await db.execute(select(Message).where(Message.session_id == session.id))).scalars().all()
            assert rows == []
        assert await ms.get_summary(agent.id, session.id) == ""  # 会话摘要删除
        assert "角色级语义记忆" in await ms.search(agent.id)  # 角色级记忆保留
        state = await graph.aget_state({"configurable": {"thread_id": thread}})
        assert not state.values.get("messages")
    finally:
        async with real_session_factory() as db:
            await db.execute(delete(Message).where(Message.session_id == session.id))
            await db.execute(delete(ChatSession).where(ChatSession.id == session.id))
            await db.execute(delete(Agent).where(Agent.id == agent.id))
            await db.commit()
        await real_checkpointer.adelete_thread(thread)
        for item in await real_store.asearch(memory_namespace(agent.id), limit=1000):
            await real_store.adelete(item.namespace, item.key)
