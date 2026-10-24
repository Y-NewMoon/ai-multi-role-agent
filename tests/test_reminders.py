"""提醒 API + `allowed_skills` CRUD（V2.2 契约 §2.8）。

路由层用替身服务验证状态码/校验/404；提醒幂等与确认/取消用真库 `roleplay_test` 验证。
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import delete, select

from app.api.v1 import api_router
from app.core.error_handlers import register_exception_handlers
from app.core.errors import NotFoundError
from app.db.models import Agent, Reminder
from app.db.models import Session as ChatSession
from app.memory.store import MemoryStore
from app.schemas.agent import AgentCreate, AgentUpdate
from app.schemas.session import SessionCreate
from app.services.agent_service import AgentService
from app.services.reminder_service import ReminderService, ReminderView, create_pending
from app.services.session_service import SessionService
from app.skills.builtin import build_default_registry


# ============================ 路由层（替身服务） ============================


def _view(reminder_id: int, *, status: str, session_id: int = 3) -> ReminderView:
    return ReminderView(
        reminder_id=reminder_id,
        agent_id=1,
        session_id=session_id,
        content="喝水",
        remind_at=datetime.now(timezone.utc) + timedelta(hours=1),
        status=status,
    )


class _FakeReminderService:
    async def list_reminders(self, *, agent_id: int, session_id: int) -> list[ReminderView]:
        if session_id == 404:
            raise NotFoundError(f"session {session_id} 不存在")
        return [_view(1, status="pending", session_id=session_id)]

    async def confirm_reminder(self, reminder_id: int) -> ReminderView:
        if reminder_id == 404:
            raise NotFoundError(f"reminder {reminder_id} 不存在")
        return _view(reminder_id, status="active")

    async def cancel_reminder(self, reminder_id: int) -> ReminderView:
        if reminder_id == 404:
            raise NotFoundError(f"reminder {reminder_id} 不存在")
        return _view(reminder_id, status="cancelled")


class _FakeAgentService:
    async def list(self):
        return [
            {
                "id": 1,
                "name": "A",
                "persona_prompt": "p",
                "allowed_skills": [],
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]

    async def create(self, data):
        return {
            "id": 9,
            "name": data.name,
            "persona_prompt": data.persona_prompt,
            "allowed_skills": list(data.allowed_skills),
            "created_at": "2026-01-01T00:00:00Z",
        }

    async def get(self, agent_id: int):
        raise NotFoundError(f"agent {agent_id} 不存在")

    async def update(self, agent_id: int, data):
        return {
            "id": agent_id,
            "name": "B",
            "persona_prompt": "p2",
            "allowed_skills": list(data.allowed_skills or []),
            "created_at": "2026-01-01T00:00:00Z",
        }

    async def delete(self, agent_id: int):  # pragma: no cover - 本文件不测删除
        return None


def _app(*, with_registry: bool = False) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    register_exception_handlers(app)
    app.state.reminder_service = _FakeReminderService()
    app.state.agent_service = _FakeAgentService()
    if with_registry:
        app.state.skill_registry = build_default_registry()
    return app


async def _request(
    method: str, url: str, payload: dict | None = None, *, with_registry: bool = False
) -> httpx.Response:
    transport = httpx.ASGITransport(app=_app(with_registry=with_registry))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, json=payload)


async def test_reminder_routes_status_codes():
    listed = await _request("GET", "/api/v1/reminders?agent_id=1&session_id=3")
    assert listed.status_code == 200
    assert listed.json()[0]["status"] == "pending"
    assert listed.json()[0]["id"] == 1

    # agent_id / session_id 不存在 → 404（由服务层抛 NotFoundError）
    assert (await _request("GET", "/api/v1/reminders?agent_id=1&session_id=404")).status_code == 404
    # 缺查询参数 → 422
    assert (await _request("GET", "/api/v1/reminders")).status_code == 422

    confirmed = await _request("POST", "/api/v1/reminders/1/confirm")
    assert confirmed.status_code == 200 and confirmed.json()["status"] == "active"
    assert (await _request("POST", "/api/v1/reminders/404/confirm")).status_code == 404

    assert (await _request("DELETE", "/api/v1/reminders/1")).status_code == 204
    assert (await _request("DELETE", "/api/v1/reminders/404")).status_code == 404


async def test_agent_allowed_skills_validation_422():
    # 未知工具名 → 422（校验用 registry）
    bad = await _request("PATCH", "/api/v1/agents/1", {"allowed_skills": ["no_such_tool"]}, with_registry=True)
    assert bad.status_code == 422
    # 空数组合法
    empty = await _request("PATCH", "/api/v1/agents/1", {"allowed_skills": []}, with_registry=True)
    assert empty.status_code == 200 and empty.json()["allowed_skills"] == []
    # 已注册工具名合法
    good = await _request(
        "PATCH", "/api/v1/agents/1", {"allowed_skills": ["get_current_time"]}, with_registry=True
    )
    assert good.status_code == 200
    # 创建时同样校验
    assert (
        await _request(
            "POST", "/api/v1/agents", {"name": "A", "persona_prompt": "p", "allowed_skills": ["nope"]},
            with_registry=True,
        )
    ).status_code == 422


# ======================== 服务层（真库 roleplay_test） ========================


async def test_agent_allowed_skills_round_trip(real_session_factory, real_store, real_checkpointer):
    service = AgentService(
        session_factory=real_session_factory,
        memory_store=MemoryStore(real_store),
        checkpointer=real_checkpointer,
    )
    agent = await service.create(
        AgentCreate(name="工具角色", persona_prompt="p", allowed_skills=["get_current_time"])
    )
    try:
        assert (await service.get(agent.id)).allowed_skills == ["get_current_time"]
        updated = await service.update(agent.id, AgentUpdate(allowed_skills=["roll_dice", "read_lore"]))
        assert updated.allowed_skills == ["roll_dice", "read_lore"]
        cleared = await service.update(agent.id, AgentUpdate(allowed_skills=[]))
        assert cleared.allowed_skills == []
    finally:
        async with real_session_factory() as db:
            await db.execute(delete(Agent).where(Agent.id == agent.id))
            await db.commit()


async def test_reminder_service_list_confirm_cancel_idempotent(
    real_session_factory, real_store, real_checkpointer
):
    agent_service = AgentService(
        session_factory=real_session_factory,
        memory_store=MemoryStore(real_store),
        checkpointer=real_checkpointer,
    )
    session_service = SessionService(
        session_factory=real_session_factory,
        memory_store=MemoryStore(real_store),
        checkpointer=real_checkpointer,
    )
    service = ReminderService(session_factory=real_session_factory)

    agent = await agent_service.create(AgentCreate(name="提醒角色", persona_prompt="p"))
    session = await session_service.create(SessionCreate(agent_id=agent.id))
    remind_at = datetime.now(timezone.utc) + timedelta(hours=2)
    try:
        assert await service.list_reminders(agent_id=agent.id, session_id=session.id) == []

        first = await create_pending(
            real_session_factory,
            agent_id=agent.id,
            session_id=session.id,
            content="喝水",
            remind_at=remind_at,
        )
        assert first.status == "pending" and first.idempotent is False
        # 同参数再调：唯一约束兜底，只落一条（§2.5-4 幂等）
        second = await create_pending(
            real_session_factory,
            agent_id=agent.id,
            session_id=session.id,
            content="喝水",
            remind_at=remind_at,
        )
        assert second.reminder_id == first.reminder_id and second.idempotent is True
        async with real_session_factory() as db:
            rows = (await db.execute(select(Reminder).where(Reminder.session_id == session.id))).scalars().all()
        assert len(rows) == 1 and rows[0].status == "pending"  # 未确认不激活

        # 列表含 pending；confirm 幂等 → active
        listed = await service.list_reminders(agent_id=agent.id, session_id=session.id)
        assert [v.status for v in listed] == ["pending"]
        assert (await service.confirm_reminder(first.reminder_id)).status == "active"
        assert (await service.confirm_reminder(first.reminder_id)).status == "active"

        # cancel 幂等；默认列表不含 cancelled
        assert (await service.cancel_reminder(first.reminder_id)).status == "cancelled"
        assert (await service.cancel_reminder(first.reminder_id)).status == "cancelled"
        assert await service.list_reminders(agent_id=agent.id, session_id=session.id) == []

        # 不存在的 agent/session/reminder → NotFoundError
        with pytest.raises(NotFoundError):
            await service.list_reminders(agent_id=10**9, session_id=session.id)
        with pytest.raises(NotFoundError):
            await service.list_reminders(agent_id=agent.id, session_id=10**9)
        with pytest.raises(NotFoundError):
            await service.confirm_reminder(10**9)
        with pytest.raises(NotFoundError):
            await service.cancel_reminder(10**9)
    finally:
        async with real_session_factory() as db:
            await db.execute(delete(Reminder).where(Reminder.session_id == session.id))
            await db.execute(delete(ChatSession).where(ChatSession.id == session.id))
            await db.execute(delete(Agent).where(Agent.id == agent.id))
            await db.commit()
