"""角色服务：CRUD + 级联删除（删除角色时连带的会话/消息/记忆/checkpoint）。

接缝 1（方案 3.11A#1）：会话的 thread_id = f"{session_id}:{agent_id}"，删除角色时
按此规则逐会话清理 checkpoint，禁用裸 session_id。

跨系统非原子（业务表事务 vs Store/checkpoint autocommit）：顺序固定为
"先业务表（同一事务）→ 再尽力删记忆/checkpoint"。若后者失败会留下孤儿数据，
危害低（不可达），后续可由对账任务清理（记入技术债）。
"""

from sqlalchemy import delete, exists, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import NotFoundError
from app.db.models import Agent, GroupMember, GroupMessage, GroupSession, Message
from app.db.models import Session as ChatSession
from app.memory.store import MemoryStore
from app.schemas.agent import AgentCreate, AgentUpdate


class AgentService:
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

    async def list(self) -> list[Agent]:
        async with self._session_factory() as db:
            result = await db.execute(select(Agent).order_by(Agent.id))
            return list(result.scalars().all())

    async def get(self, agent_id: int) -> Agent:
        async with self._session_factory() as db:
            agent = await db.get(Agent, agent_id)
            if agent is None:
                raise NotFoundError(f"agent {agent_id} 不存在")
            return agent

    async def create(self, data: AgentCreate) -> Agent:
        async with self._session_factory() as db:
            agent = Agent(
                name=data.name,
                persona_prompt=data.persona_prompt,
                background=data.background,
                avatar=data.avatar,
            )
            db.add(agent)
            await db.commit()
            await db.refresh(agent)
            return agent

    async def update(self, agent_id: int, data: AgentUpdate) -> Agent:
        async with self._session_factory() as db:
            agent = await db.get(Agent, agent_id)
            if agent is None:
                raise NotFoundError(f"agent {agent_id} 不存在")
            # exclude_unset：只更新请求中显式给出的字段（可把 background 置 null）
            for field, value in data.model_dump(exclude_unset=True).items():
                setattr(agent, field, value)
            await db.commit()
            await db.refresh(agent)
            return agent

    async def delete(self, agent_id: int) -> None:
        """删除角色：级联删该角色的全部会话与消息，并清理记忆与 checkpoint。

        另含**空群自动清理**：角色退出后若某群已无成员，则连同该群的展示消息一并删除，
        避免残留「无成员的空壳群」。
        """
        async with self._session_factory() as db:
            agent = await db.get(Agent, agent_id)
            if agent is None:
                raise NotFoundError(f"agent {agent_id} 不存在")

            session_ids = list(
                (await db.execute(select(ChatSession.id).where(ChatSession.agent_id == agent_id)))
                .scalars()
                .all()
            )
            # 删除前记录该角色所在的群：删除后成员行不再指向它，需据此清理群聊 checkpoint
            # thread（接缝 1 群聊形态 = g{group_id}:{agent_id}）。
            group_ids = list(
                (
                    await db.execute(
                        select(GroupMember.group_session_id).where(GroupMember.agent_id == agent_id)
                    )
                )
                .scalars()
                .all()
            )
            # FK 顺序：messages → sessions → agents（同一事务，原子提交）。
            # 群成员行/群内发言显式删除，不依赖 DB 级联，保证后续空群判定在事务内可见。
            if session_ids:
                await db.execute(delete(Message).where(Message.session_id.in_(session_ids)))
            await db.execute(delete(ChatSession).where(ChatSession.agent_id == agent_id))
            await db.execute(delete(GroupMessage).where(GroupMessage.agent_id == agent_id))
            await db.execute(delete(GroupMember).where(GroupMember.agent_id == agent_id))
            await db.execute(delete(Agent).where(Agent.id == agent_id))

            # 空群自动清理：删除所有已无成员的群及其全部展示消息。
            empty_group_ids = list(
                (
                    await db.execute(
                        select(GroupSession.id).where(
                            ~exists(select(GroupMember.group_session_id).where(
                                GroupMember.group_session_id == GroupSession.id
                            ))
                        )
                    )
                )
                .scalars()
                .all()
            )
            if empty_group_ids:
                await db.execute(
                    delete(GroupMessage).where(GroupMessage.group_session_id.in_(empty_group_ids))
                )
                await db.execute(delete(GroupSession).where(GroupSession.id.in_(empty_group_ids)))
            await db.commit()

        # 业务表已落库；以下为尽力清理（失败不回滚，仅留孤儿数据）
        await self._memory_store.delete_all(agent_id)
        for session_id in session_ids:
            await self._checkpointer.adelete_thread(f"{session_id}:{agent_id}")
        for group_id in group_ids:
            await self._checkpointer.adelete_thread(f"g{group_id}:{agent_id}")

    async def clear_memory(self, agent_id: int) -> None:
        """清除角色记忆：长期记忆（Store 全量）+ 全部对话历史（checkpoint 与展示消息）。

        范围（与用户确认）：长期记忆 + 全部对话历史。保留角色档案（Agent）与会话壳。
        - 长期记忆：Store namespace（语义记忆 + 各会话滚动摘要）整清；
        - 单聊：删该角色各会话的展示消息 + 各 checkpoint thread；
        - 群聊：删该角色在各群的 checkpoint thread（不动共享的 group_messages，避免影响其他成员）。
        """
        async with self._session_factory() as db:
            agent = await db.get(Agent, agent_id)
            if agent is None:
                raise NotFoundError(f"agent {agent_id} 不存在")

            session_ids = list(
                (await db.execute(select(ChatSession.id).where(ChatSession.agent_id == agent_id)))
                .scalars()
                .all()
            )
            group_ids = list(
                (
                    await db.execute(
                        select(GroupMember.group_session_id).where(GroupMember.agent_id == agent_id)
                    )
                )
                .scalars()
                .all()
            )
            if session_ids:
                await db.execute(delete(Message).where(Message.session_id.in_(session_ids)))
            await db.commit()

        # 业务表已落库；以下为尽力清理（失败不回滚，仅留孤儿数据）
        await self._memory_store.delete_all(agent_id)
        for session_id in session_ids:
            await self._checkpointer.adelete_thread(f"{session_id}:{agent_id}")
        for group_id in group_ids:
            await self._checkpointer.adelete_thread(f"g{group_id}:{agent_id}")
