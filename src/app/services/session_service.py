"""会话服务：CRUD + 会话级删除。

删除语义（与用户确认）：仅删会话级数据——该会话 messages + checkpoint thread +
滚动摘要 `summary:{session_id}`；保留角色（Agent）与角色级语义记忆（跨会话共享）。

接缝 1：checkpoint thread_id = f"{session_id}:{agent_id}"。
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import NotFoundError
from app.db.models import Agent, Message
from app.db.models import Session as ChatSession
from app.memory.store import MemoryStore
from app.schemas.session import SessionCreate, SessionUpdate


class SessionService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        memory_store: MemoryStore,
        checkpointer,
    ) -> None:
        self._session_factory = session_factory
        self._memory_store = memory_store
        self._checkpointer = checkpointer

    async def list(self, *, agent_id: int | None = None) -> list[ChatSession]:
        async with self._session_factory() as db:
            stmt = select(ChatSession).order_by(ChatSession.id)
            if agent_id is not None:
                stmt = stmt.where(ChatSession.agent_id == agent_id)
            result = await db.execute(stmt)
            return list(result.scalars().all())

    async def get(self, session_id: int) -> ChatSession:
        async with self._session_factory() as db:
            session = await db.get(ChatSession, session_id)
            if session is None:
                raise NotFoundError(f"session {session_id} 不存在")
            return session

    async def list_messages(self, session_id: int) -> list[Message]:
        """按时间序返回该会话的展示用消息快照（历史回显）。"""
        async with self._session_factory() as db:
            if await db.get(ChatSession, session_id) is None:
                raise NotFoundError(f"session {session_id} 不存在")
            result = await db.execute(
                select(Message).where(Message.session_id == session_id).order_by(Message.id)
            )
            return list(result.scalars().all())

    async def create(self, data: SessionCreate) -> ChatSession:
        async with self._session_factory() as db:
            if await db.get(Agent, data.agent_id) is None:
                raise NotFoundError(f"agent {data.agent_id} 不存在")
            session = ChatSession(agent_id=data.agent_id, title=data.title)
            db.add(session)
            await db.commit()
            await db.refresh(session)
            return session

    async def update(self, session_id: int, data: SessionUpdate) -> ChatSession:
        async with self._session_factory() as db:
            session = await db.get(ChatSession, session_id)
            if session is None:
                raise NotFoundError(f"session {session_id} 不存在")
            for field, value in data.model_dump(exclude_unset=True).items():
                setattr(session, field, value)
            await db.commit()
            await db.refresh(session)
            return session

    async def delete(self, session_id: int) -> None:
        """删除会话：删该会话 messages + checkpoint thread + 滚动摘要（保留角色与语义记忆）。"""
        async with self._session_factory() as db:
            session = await db.get(ChatSession, session_id)
            if session is None:
                raise NotFoundError(f"session {session_id} 不存在")
            agent_id = session.agent_id

            await db.execute(delete(Message).where(Message.session_id == session_id))
            await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
            await db.commit()

        # 业务表已落库；以下为尽力清理
        await self._checkpointer.adelete_thread(f"{session_id}:{agent_id}")
        await self._memory_store.delete_summary(agent_id, session_id)
