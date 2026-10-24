"""B-03 提醒 TTL 与确认话术（V2.4 契约 §2.8）。"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, update

from app.db.models import Agent, Reminder
from app.db.models import Session as ChatSession
from app.services.reminder_service import (
    ReminderService,
    confirm,
    create_pending,
    list_reminders,
    pending_ttl_hours,
)
from app.skills.builtin.reminder import make_set_reminder_skill


def test_set_reminder_description_requires_confirmation():
    """工具描述/使用时机必须含“请用户确认”，且禁止“已记下/已设置”话术。"""
    skill = make_set_reminder_skill()
    for text in (skill.description, skill.when_to_use):
        assert "请用户确认" in text
        assert "不得说" in text and "已记下/已设置" in text


async def _seed(real_session_factory) -> tuple[Agent, ChatSession]:
    async with real_session_factory() as db:
        agent = Agent(name="TTL角色", persona_prompt="p")
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
        session = ChatSession(agent_id=agent.id)
        db.add(session)
        await db.commit()
        await db.refresh(session)
    return agent, session


async def _cleanup(real_session_factory, agent: Agent, session: ChatSession) -> None:
    async with real_session_factory() as db:
        await db.execute(delete(Reminder).where(Reminder.session_id == session.id))
        await db.execute(delete(ChatSession).where(ChatSession.id == session.id))
        await db.execute(delete(Agent).where(Agent.id == agent.id))
        await db.commit()


async def _age_reminder(real_session_factory, reminder_id: int, hours: int) -> None:
    """把 created_at 调到 TTL 之前（模拟 pending 已过期）。"""
    old = datetime.now(timezone.utc) - timedelta(hours=hours)
    async with real_session_factory() as db:
        await db.execute(
            update(Reminder).where(Reminder.id == reminder_id).values(created_at=old)
        )
        await db.commit()


async def _status(real_session_factory, reminder_id: int) -> str:
    async with real_session_factory() as db:
        row = (
            await db.execute(select(Reminder).where(Reminder.id == reminder_id))
        ).scalar_one()
        return row.status


async def test_expired_pending_hidden_and_confirm_cancelled(real_session_factory):
    agent, session = await _seed(real_session_factory)
    service = ReminderService(session_factory=real_session_factory)
    remind_at = datetime.now(timezone.utc) + timedelta(hours=2)
    try:
        view = await create_pending(
            real_session_factory,
            agent_id=agent.id,
            session_id=session.id,
            content="喝水",
            remind_at=remind_at,
        )
        assert view.status == "pending"
        assert [v.reminder_id for v in await service.list_reminders(
            agent_id=agent.id, session_id=session.id
        )] == [view.reminder_id]

        # 过期 → 默认列表不含（惰性清理已把它置 cancelled）
        await _age_reminder(real_session_factory, view.reminder_id, pending_ttl_hours() + 1)
        assert await service.list_reminders(agent_id=agent.id, session_id=session.id) == []
        assert await _status(real_session_factory, view.reminder_id) == "cancelled"

        # confirm 对已过期 pending：200 语义、status=cancelled，不得变 active
        confirmed = await service.confirm_reminder(view.reminder_id)
        assert confirmed.status == "cancelled"
        assert (await service.confirm_reminder(view.reminder_id)).status == "cancelled"
        assert await _status(real_session_factory, view.reminder_id) == "cancelled"
    finally:
        await _cleanup(real_session_factory, agent, session)


async def test_expired_same_key_returns_cancelled_without_recreate(real_session_factory):
    agent, session = await _seed(real_session_factory)
    remind_at = datetime.now(timezone.utc) + timedelta(hours=2)
    try:
        first = await create_pending(
            real_session_factory,
            agent_id=agent.id,
            session_id=session.id,
            content="喝水",
            remind_at=remind_at,
        )
        await _age_reminder(real_session_factory, first.reminder_id, pending_ttl_hours() + 1)

        # 同键再次请求：返回既有 cancelled（幂等），不新建
        again = await create_pending(
            real_session_factory,
            agent_id=agent.id,
            session_id=session.id,
            content="喝水",
            remind_at=remind_at,
        )
        assert again.reminder_id == first.reminder_id
        assert again.status == "cancelled"
        assert again.idempotent is True
        async with real_session_factory() as db:
            rows = (
                await db.execute(select(Reminder).where(Reminder.session_id == session.id))
            ).scalars().all()
        assert len(rows) == 1
    finally:
        await _cleanup(real_session_factory, agent, session)


async def test_fresh_pending_still_confirms_active(real_session_factory):
    agent, session = await _seed(real_session_factory)
    remind_at = datetime.now(timezone.utc) + timedelta(hours=2)
    try:
        view = await create_pending(
            real_session_factory,
            agent_id=agent.id,
            session_id=session.id,
            content="喝水",
            remind_at=remind_at,
        )
        assert (await confirm(real_session_factory, view.reminder_id)).status == "active"

        # 已是 active 的行不受 TTL 清理影响，仍在列表中
        await _age_reminder(real_session_factory, view.reminder_id, pending_ttl_hours() + 1)
        listed = await list_reminders(
            real_session_factory, agent_id=agent.id, session_id=session.id
        )
        assert [(v.reminder_id, v.status) for v in listed] == [(view.reminder_id, "active")]
    finally:
        await _cleanup(real_session_factory, agent, session)
