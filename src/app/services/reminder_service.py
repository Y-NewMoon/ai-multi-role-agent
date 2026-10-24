"""提醒服务（契约 §2.8）：pending 落库 + 幂等；确认/取消/列表。V2.2 不做定时投递。"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.core.errors import NotFoundError
from app.db.models import Agent, Reminder
from app.db.models import Session as ChatSession

PENDING = "pending"
CANCELLED = "cancelled"
DEFAULT_TTL_HOURS = 24


def pending_ttl_hours() -> int:
    """pending 的有效期（小时）；配置非法/读取失败时按 24 兜底（§2.11 只告警不抛）。"""
    try:
        ttl = int(get_settings().reminder_pending_ttl_hours)
    except Exception:  # noqa: BLE001 - 配置读取失败不得影响提醒主流程
        return DEFAULT_TTL_HOURS
    return ttl if ttl > 0 else DEFAULT_TTL_HOURS


async def expire_pending(db: AsyncSession, *, now: datetime | None = None) -> int:
    """惰性清理（契约 §2.8）：把超过 TTL 的 pending 置 cancelled，返回清理条数。

    只由 `list_reminders` / `confirm` 入口调用；无后台任务、无迁移。
    """
    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(hours=pending_ttl_hours())
    result = await db.execute(
        update(Reminder)
        .where(Reminder.status == PENDING, Reminder.created_at < cutoff)
        .values(status=CANCELLED, updated_at=moment)
    )
    await db.commit()
    return result.rowcount or 0


def idempotency_key(agent_id: int, session_id: int, content: str, remind_at_iso: str) -> str:
    """契约 §2.5-4：sha256(f"{agent_id}|{session_id}|{content}|{remind_at_iso}")。"""
    raw = f"{agent_id}|{session_id}|{content}|{remind_at_iso}"
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class ReminderView:
    reminder_id: int
    agent_id: int
    session_id: int
    content: str
    remind_at: datetime
    status: str
    idempotent: bool = False


def _view(row: Reminder, *, idempotent: bool = False) -> ReminderView:
    return ReminderView(
        reminder_id=row.id,
        agent_id=row.agent_id,
        session_id=row.session_id,
        content=row.content,
        remind_at=row.remind_at,
        status=row.status,
        idempotent=idempotent,
    )


async def create_pending(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    agent_id: int,
    session_id: int,
    content: str,
    remind_at: datetime,
) -> ReminderView:
    """落 pending；同幂等键已存在则返回既有记录（idempotent=True）。"""
    key = idempotency_key(agent_id, session_id, content, remind_at.isoformat())
    async with session_factory() as db:
        stmt = (
            pg_insert(Reminder)
            .values(
                agent_id=agent_id,
                session_id=session_id,
                content=content,
                remind_at=remind_at,
                status="pending",
                idempotency_key=key,
            )
            .on_conflict_do_nothing(index_elements=["idempotency_key"])
            .returning(Reminder.id)
        )
        new_id = (await db.execute(stmt)).scalar_one_or_none()
        await db.commit()
        row = (
            await db.execute(select(Reminder).where(Reminder.idempotency_key == key))
        ).scalar_one()
        # 同键命中既有 pending 且已过期：置 cancelled 后返回，不新建（契约 §2.8）
        if row.status == PENDING and row.created_at < datetime.now(timezone.utc) - timedelta(
            hours=pending_ttl_hours()
        ):
            row.status = CANCELLED
            row.updated_at = datetime.now(timezone.utc)
            await db.commit()
        return _view(row, idempotent=new_id is None)


async def list_reminders(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    agent_id: int | None = None,
    session_id: int | None = None,
    include_cancelled: bool = False,
) -> list[ReminderView]:
    async with session_factory() as db:
        # 先惰性清理过期 pending；清理后的行是 cancelled，自然不会出现在默认列表里
        await expire_pending(db)
        stmt = select(Reminder).order_by(Reminder.remind_at.asc(), Reminder.id.asc())
        if agent_id is not None:
            stmt = stmt.where(Reminder.agent_id == agent_id)
        if session_id is not None:
            stmt = stmt.where(Reminder.session_id == session_id)
        if not include_cancelled:
            stmt = stmt.where(Reminder.status != CANCELLED)
        rows = (await db.execute(stmt)).scalars().all()
        return [_view(row) for row in rows]


async def confirm(
    session_factory: async_sessionmaker[AsyncSession], reminder_id: int
) -> ReminderView | None:
    """pending → active（幂等：已是 active 直接返回；已过期 pending 置 cancelled 不激活）。"""
    async with session_factory() as db:
        # 先惰性清理：过期 pending 已变 cancelled，此处不会把它激活成 active
        await expire_pending(db)
        row = await db.get(Reminder, reminder_id)
        if row is None:
            return None
        if row.status == PENDING:
            row.status = "active"
            await db.commit()
        return _view(row)


async def cancel(
    session_factory: async_sessionmaker[AsyncSession], reminder_id: int
) -> ReminderView | None:
    """pending/active → cancelled（幂等：已 cancelled 直接返回）。"""
    async with session_factory() as db:
        row = await db.get(Reminder, reminder_id)
        if row is None:
            return None
        if row.status in ("pending", "active"):
            row.status = CANCELLED
            await db.commit()
        return _view(row)


class ReminderService:
    """API 层用的提醒服务（契约 §2.8）：列表/确认/取消 + agent/session 存在性 404。

    工具侧（`set_reminder`）直接调模块函数 `create_pending`；本类只服务 HTTP 端点，
    故把「agent_id / session_id 不存在 → 404」的校验放在这里。
    """

    def __init__(self, *, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list_reminders(self, *, agent_id: int, session_id: int) -> list[ReminderView]:
        async with self._session_factory() as db:
            if await db.get(Agent, agent_id) is None:
                raise NotFoundError(f"agent {agent_id} 不存在")
            if await db.get(ChatSession, session_id) is None:
                raise NotFoundError(f"session {session_id} 不存在")
        return await list_reminders(
            self._session_factory, agent_id=agent_id, session_id=session_id
        )

    async def confirm_reminder(self, reminder_id: int) -> ReminderView:
        view = await confirm(self._session_factory, reminder_id)
        if view is None:
            raise NotFoundError(f"reminder {reminder_id} 不存在")
        return view

    async def cancel_reminder(self, reminder_id: int) -> ReminderView:
        view = await cancel(self._session_factory, reminder_id)
        if view is None:
            raise NotFoundError(f"reminder {reminder_id} 不存在")
        return view
