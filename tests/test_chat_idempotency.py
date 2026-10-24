"""非流式 /chat 幂等（V2.4 D8 §2.9）：同 key 不双写、不同 key 正常、空 key 不缓存。"""

import asyncio

import httpx
from fastapi import FastAPI

from app.api.v1 import api_router
from app.core.error_handlers import register_exception_handlers
from app.db.models import Agent
from app.db.models import Session as ChatSession
from app.schemas.chat import ChatResponse
from app.services.chat_service import ChatService


class _Store:
    async def delete(self, agent_id, content: str) -> None:  # pragma: no cover - 不触发撤回
        pass


class _Session:
    def __init__(self, tables: dict) -> None:
        self._tables = tables
        self.added: list = []

    async def get(self, model, pk):
        return self._tables.get((model, pk))

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        pass

    async def refresh(self, obj) -> None:
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added)


class _Ctx:
    def __init__(self, session: _Session) -> None:
        self._session = session

    async def __aenter__(self) -> _Session:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


class _Factory:
    def __init__(self) -> None:
        agent = Agent(id=1, name="角色", persona_prompt="p")
        session = ChatSession(id=1, agent_id=1)
        self._tables = {(Agent, 1): agent, (ChatSession, 1): session}
        self.sessions: list[_Session] = []

    def __call__(self) -> _Ctx:
        session = _Session(self._tables)
        self.sessions.append(session)
        return _Ctx(session)

    @property
    def added(self) -> list:
        return [obj for session in self.sessions for obj in session.added]


class _Graph:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, state, config):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return {"reply": f"回复{self.calls}"}


def _service(*, delay: float = 0.0, **kwargs) -> tuple[ChatService, _Factory, _Graph]:
    factory = _Factory()
    graph = _Graph(delay=delay)
    return ChatService(graph=graph, session_factory=factory, memory_store=_Store(), **kwargs), factory, graph


async def test_same_client_msg_id_no_double_write():
    service, factory, graph = _service()
    first = await service.chat(session_id=1, message="在吗", client_msg_id="c1")
    second = await service.chat(session_id=1, message="在吗", client_msg_id="c1")

    assert graph.calls == 1  # 第二次未调图
    assert [m.role for m in factory.added] == ["user", "assistant"]  # 只写 1 组 messages
    assert first.idempotent is False
    assert second.idempotent is True
    assert second.reply == first.reply == "回复1"


async def test_different_key_or_empty_key_runs_again():
    service, factory, graph = _service()
    await service.chat(session_id=1, message="a", client_msg_id="c1")
    await service.chat(session_id=1, message="b", client_msg_id="c2")
    await service.chat(session_id=1, message="c")  # 空 key 不缓存
    await service.chat(session_id=1, message="c")  # 再来一次仍要执行
    assert graph.calls == 4


async def test_same_key_different_session_not_shared():
    service, factory, graph = _service()
    # 两个会话共享同一 client_msg_id 不得串号（key 含 session_id）
    factory._tables[(ChatSession, 2)] = ChatSession(id=2, agent_id=1)
    await service.chat(session_id=1, message="a", client_msg_id="c1")
    await service.chat(session_id=2, message="a", client_msg_id="c1")
    assert graph.calls == 2


async def test_concurrent_same_key_executes_once():
    service, factory, graph = _service(delay=0.02)
    results = await asyncio.gather(
        service.chat(session_id=1, message="在吗", client_msg_id="c1"),
        service.chat(session_id=1, message="在吗", client_msg_id="c1"),
    )
    assert graph.calls == 1
    assert sorted(r.idempotent for r in results) == [False, True]
    assert len([m for m in factory.added if m.role == "user"]) == 1


async def test_ttl_expiry_allows_rerun():
    service, factory, graph = _service(idempotency_ttl_s=0.01)
    await service.chat(session_id=1, message="a", client_msg_id="c1")
    await asyncio.sleep(0.03)
    again = await service.chat(session_id=1, message="a", client_msg_id="c1")
    assert graph.calls == 2
    assert again.idempotent is False


async def test_capacity_evicts_oldest():
    service, factory, graph = _service(idempotency_max_entries=2)
    for i in range(3):
        await service.chat(session_id=1, message=f"m{i}", client_msg_id=f"c{i}")
    assert len(service._idempotency) == 2
    # 最早的 c0 已被淘汰 → 再请求会重新执行
    await service.chat(session_id=1, message="m0", client_msg_id="c0")
    assert graph.calls == 4


# ===== 路由层：字段透传与校验 =====


class _FakeChatService:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    async def chat(self, *, session_id: int, message: str, client_msg_id=None):
        self.kwargs = {"session_id": session_id, "message": message,
                       "client_msg_id": client_msg_id}
        return ChatResponse(session_id=session_id, agent_id=1, reply="ok",
                            idempotent=client_msg_id == "dup")


async def _post(payload: dict) -> tuple[httpx.Response, _FakeChatService]:
    service = _FakeChatService()
    app = FastAPI()
    app.include_router(api_router)
    register_exception_handlers(app)
    app.state.chat_service = service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/chat", json=payload)
    return resp, service


async def test_chat_route_passes_client_msg_id_and_returns_idempotent():
    resp, service = await _post({"session_id": 1, "message": "在吗", "client_msg_id": "dup"})
    assert resp.status_code == 200
    assert service.kwargs["client_msg_id"] == "dup"
    assert resp.json()["idempotent"] is True


async def test_chat_route_rejects_overlong_client_msg_id():
    resp, _ = await _post({"session_id": 1, "message": "在吗", "client_msg_id": "x" * 65})
    assert resp.status_code == 422
