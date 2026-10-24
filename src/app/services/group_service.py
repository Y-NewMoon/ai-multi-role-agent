"""群聊服务层（M4 · ADR-0004）。

职责：建群 / 成员校验 / 展示快照读写 / **共享信息扇出** / 整轮编排与落库。
接缝 5：业务表访问全部在本层；图内节点零 SQL。
接缝 2：扇出写入仍落在各成员**自己的** namespace（`agent:{id}`），不建公共 namespace。

SSE 时序（ADR-0003 延续到群聊）：`token` / `turn_done` 为角色级；
**最终 `done` 在整轮所有角色结束、`group_messages` 快照落库之后才推送**。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.nodes.generate import normalize_thinkings
from app.agents.supervisor import GroupMember, Supervisor
from app.agents.thread_state import remove_thread_messages, thread_message_contents
from app.core.errors import AppError, NotFoundError
from app.core.locks import KeyedLockPool
from app.db.models import Agent, GroupMessage
from app.db.models import GroupMember as GroupMemberRow
from app.db.models import GroupSession
from app.llm.base import LLMProvider
from app.memory.store import MemoryStore
from app.schemas.group import GroupCreate


@dataclass(frozen=True)
class GroupContext:
    """一次群聊轮次所需的全部上下文（角色档案由本层预取）。"""

    group_session_id: int
    name: str
    members: list[GroupMember]


class GroupService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        memory_store: MemoryStore,
        llm: LLMProvider,
        graph_factory: Callable[[], Any],
        checkpointer,
        max_members: int = 5,
        max_turns: int = 4,
        idle_max_messages: int = 10,
        idle_turn_max_messages: int = 2,
        tool_executor=None,
    ) -> None:
        self._session_factory = session_factory
        self._memory_store = memory_store
        self._llm = llm
        self._graph_factory = graph_factory
        self._checkpointer = checkpointer
        self._max_members = max_members
        self._max_turns = max_turns
        self._idle_max_messages = idle_max_messages
        self._idle_turn_max_messages = idle_turn_max_messages
        # V2.2：透传给 Supervisor，在每个角色每轮进图前清空工具缓存（§2.2-5）
        self._tool_executor = tool_executor
        # 同群并发串行化（V2.4 B-02：引用计数锁池，退出即释放；单进程语义，多实例需分布式锁）
        self._locks = KeyedLockPool()

    # ===== 建群 / 查询 =====

    async def create(self, data: GroupCreate) -> GroupSession:
        agent_ids = list(dict.fromkeys(data.agent_ids))  # 去重保序
        if len(agent_ids) > self._max_members:
            raise AppError(f"群内角色数不能超过 {self._max_members} 个")

        async with self._session_factory() as db:
            found = set(
                (await db.execute(select(Agent.id).where(Agent.id.in_(agent_ids))))
                .scalars()
                .all()
            )
            missing = [a for a in agent_ids if a not in found]
            if missing:
                raise NotFoundError(f"角色不存在：{missing}")

            group = GroupSession(name=data.name)
            db.add(group)
            await db.flush()
            for agent_id in agent_ids:
                db.add(GroupMemberRow(group_session_id=group.id, agent_id=agent_id))
            await db.commit()
            await db.refresh(group)
            return group

    async def list_groups(self) -> list[GroupSession]:
        async with self._session_factory() as db:
            result = await db.execute(select(GroupSession).order_by(GroupSession.id))
            return list(result.scalars().all())

    async def get(self, group_session_id: int) -> GroupSession:
        async with self._session_factory() as db:
            group = await db.get(GroupSession, group_session_id)
            if group is None:
                raise NotFoundError(f"群 {group_session_id} 不存在")
            return group

    async def list_messages(self, group_session_id: int) -> list[GroupMessage]:
        """按时间序返回该群的历史消息快照（历史回显）。"""
        async with self._session_factory() as db:
            if await db.get(GroupSession, group_session_id) is None:
                raise NotFoundError(f"群 {group_session_id} 不存在")
            result = await db.execute(
                select(GroupMessage)
                .where(GroupMessage.group_session_id == group_session_id)
                .order_by(GroupMessage.id)
            )
            return list(result.scalars().all())

    async def member_ids(self, group_session_id: int) -> list[int]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(GroupMemberRow.agent_id)
                .where(GroupMemberRow.group_session_id == group_session_id)
                .order_by(GroupMemberRow.agent_id)
            )
            return list(result.scalars().all())

    async def delete(self, group_session_id: int) -> None:
        """删除群：删该群全部展示消息 + 成员行 + 群本身，并清理各成员的群聊 checkpoint thread。

        已扇出的语义记忆仍留在各成员私有 namespace（接缝 2：记忆属角色、不属群），
        V1 不做回溯清除（与「删角色」以外场景一致）。
        """
        async with self._session_factory() as db:
            if await db.get(GroupSession, group_session_id) is None:
                raise NotFoundError(f"群 {group_session_id} 不存在")
            member_ids = list(
                (
                    await db.execute(
                        select(GroupMemberRow.agent_id).where(
                            GroupMemberRow.group_session_id == group_session_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            await db.execute(
                delete(GroupMessage).where(GroupMessage.group_session_id == group_session_id)
            )
            await db.execute(
                delete(GroupMemberRow).where(GroupMemberRow.group_session_id == group_session_id)
            )
            await db.execute(delete(GroupSession).where(GroupSession.id == group_session_id))
            await db.commit()

        # 业务表已落库；以下为尽力清理（失败不回滚，仅留孤儿数据）
        for agent_id in member_ids:
            await self._checkpointer.adelete_thread(f"g{group_session_id}:{agent_id}")

    async def remove_member(self, group_session_id: int, agent_id: int) -> None:
        """把一个角色移出群：删成员行 + 清理其群聊 checkpoint thread。

        若移出后群内已无成员，则连同该群及其全部展示消息一并删除（不留空壳群）。
        """
        async with self._session_factory() as db:
            if await db.get(GroupSession, group_session_id) is None:
                raise NotFoundError(f"群 {group_session_id} 不存在")
            if await db.get(GroupMemberRow, (group_session_id, agent_id)) is None:
                raise NotFoundError(f"角色 {agent_id} 不在群 {group_session_id} 中")
            await db.execute(
                delete(GroupMemberRow).where(
                    GroupMemberRow.group_session_id == group_session_id,
                    GroupMemberRow.agent_id == agent_id,
                )
            )
            remaining = list(
                (
                    await db.execute(
                        select(GroupMemberRow.agent_id).where(
                            GroupMemberRow.group_session_id == group_session_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not remaining:
                await db.execute(
                    delete(GroupMessage).where(GroupMessage.group_session_id == group_session_id)
                )
                await db.execute(delete(GroupSession).where(GroupSession.id == group_session_id))
            await db.commit()

        # 业务表已落库；以下为尽力清理（失败不回滚，仅留孤儿数据）
        await self._checkpointer.adelete_thread(f"g{group_session_id}:{agent_id}")

    async def load_context(self, group_session_id: int) -> GroupContext:
        """取群 + 成员档案（服务层预取，供图内只读使用）。"""
        async with self._session_factory() as db:
            group = await db.get(GroupSession, group_session_id)
            if group is None:
                raise NotFoundError(f"群 {group_session_id} 不存在")
            rows = (
                (
                    await db.execute(
                        select(Agent)
                        .join(GroupMemberRow, GroupMemberRow.agent_id == Agent.id)
                        .where(GroupMemberRow.group_session_id == group_session_id)
                        .order_by(GroupMemberRow.agent_id)
                    )
                )
                .scalars()
                .all()
            )
            if not rows:
                raise AppError(f"群 {group_session_id} 没有成员")
            members = [
                GroupMember(
                    agent_id=a.id,
                    name=a.name,
                    persona_prompt=a.persona_prompt,
                    background=a.background or "",
                    # V2.2：每个角色用自己的工具白名单（契约 §2.4：工具结果不跨角色共享）
                    allowed_skills=tuple(a.allowed_skills or ()),
                )
                for a in rows
            ]
            return GroupContext(group_session_id=group.id, name=group.name, members=members)

    # ===== 写快照 / 扇出 =====

    async def _append(
        self,
        group_session_id: int,
        *,
        agent_id: int | None,
        role: str,
        content: str,
        is_idle: bool = False,
        thinking: str | None = None,
    ) -> int:
        """落一条群聊展示快照，返回其 id。"""
        async with self._session_factory() as db:
            row = GroupMessage(
                group_session_id=group_session_id,
                agent_id=agent_id,
                role=role,
                content=content,
                is_idle=is_idle,
                thinking=thinking,
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return row.id

    async def _fanout(
        self,
        context: GroupContext,
        content: str,
        *,
        role: str,
        speaker_agent_id: int | None,
        exclude: frozenset[int] = frozenset(),
    ) -> None:
        """把一条公屏内容写进每个成员的私有 namespace（标 visibility=group）。

        隔离墙在"读"上：每个成员的 namespace 只装自己的副本，不建公共 namespace。

        他人发言落库时**带「【角色名】」前缀**（与工作记忆逐条注入的格式一致）：
        ① 否则别人的台词与本角色自己的记忆无从区分，注入 `【你记得的相关信息】`
        （compose_prompt）后模型会当成自己说过的话照抄；
        ② 记忆 key 只由 (agent_id, content) 决定，不加前缀时"两人说了同一句话"
        会命中同一个 key，把本人那条覆盖成他人的（实测已发生）。
        用户发言（speaker 为空）不属他人发言，保持原文。
        """
        name = next((m.name for m in context.members if m.agent_id == speaker_agent_id), None)
        text = f"【{name}】{content}" if speaker_agent_id is not None and name else content
        for member in context.members:
            if member.agent_id in exclude:
                continue
            await self._memory_store.put(
                member.agent_id,
                text,
                role=role,
                visibility="group",
                origin_group_session_id=context.group_session_id,
                speaker_agent_id=speaker_agent_id,
            )

    # ===== 一轮群聊 =====

    async def _next_turn(self, group_session_id: int) -> int | None:
        """该群用户消息序号（1-based，落本轮用户消息之前取数）；取不到为 None。"""
        try:
            async with self._session_factory() as db:
                count = (
                    await db.execute(
                        select(func.count())
                        .select_from(GroupMessage)
                        .where(
                            GroupMessage.group_session_id == group_session_id,
                            GroupMessage.role == "user",
                        )
                    )
                ).scalar_one()
            return int(count) + 1
        except Exception:  # noqa: BLE001 - trace 旁路失败不影响主流程
            return None

    async def stream(
        self, *, context: GroupContext, message: str
    ) -> AsyncIterator[tuple[str, dict]]:
        group_session_id = context.group_session_id
        async with self._locks.acquire(f"group:{group_session_id}"):
            # V2.4：同一轮群聊共享 parent_trace_id（每个发言人一条 trace，不写汇总条）
            parent_trace_id = uuid.uuid4().hex
            turn = await self._next_turn(group_session_id)
            # 1) 用户消息：落快照 + 扇出给全部成员（ADR-0004 §3.2）
            user_msg_id = await self._append(
                group_session_id, agent_id=None, role="user", content=message
            )
            await self._fanout(context, message, role="user", speaker_agent_id=None)

            # 2) 调度各角色发言（串行）；每个角色的公屏回复落快照 + 扇出给其余成员
            supervisor = Supervisor(
                llm=self._llm,
                graph_factory=self._graph_factory,
                members=context.members,
                max_turns=self._max_turns,
                tool_executor=self._tool_executor,
            )
            order: list[int] = []
            replies: list[dict] = []
            async for event, data in supervisor.run(
                group_session_id=group_session_id,
                user_message=message,
                parent_trace_id=parent_trace_id,
                turn=turn,
            ):
                if event == "turn_done":
                    agent_id = int(data["agent_id"])
                    # M5：一个角色一轮可连发多条，逐条落快照并扇出
                    parts = [str(p) for p in (data.get("replies") or [])] or [
                        str(data.get("reply", ""))
                    ]
                    thinkings = normalize_thinkings(
                        [str(t) for t in (data.get("thinkings") or [])], len(parts)
                    )
                    for i, part in enumerate(parts):
                        reply_id = await self._append(
                            group_session_id,
                            agent_id=agent_id,
                            role="assistant",
                            content=part,
                            thinking=thinkings[i] or None,
                        )
                        await self._fanout(
                            context,
                            part,
                            role="assistant",
                            speaker_agent_id=agent_id,
                            exclude=frozenset({agent_id}),  # 本人那份以私有记录为准
                        )
                        replies.append(
                            {
                                "agent_id": agent_id,
                                "message_id": reply_id,
                                "content": part,
                                "thinking": thinkings[i] or None,
                            }
                        )
                    order.append(agent_id)
                yield event, data

            # 3) 快照已落库，才推 done（ADR-0003 时序）
            yield "done", {
                "group_session_id": group_session_id,
                "message_id": user_msg_id,
                "order": order,
                "replies": replies,
            }

    async def _last_speaker(self, group_session_id: int) -> int | None:
        """最近一条角色发言的 agent_id（用于闲聊公平轮转的起点；无则 None）。"""
        async with self._session_factory() as db:
            return (
                await db.execute(
                    select(GroupMessage.agent_id)
                    .where(
                        GroupMessage.group_session_id == group_session_id,
                        GroupMessage.role == "assistant",
                        GroupMessage.agent_id.is_not(None),
                    )
                    .order_by(GroupMessage.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

    async def idle_stream(self, *, context: GroupContext) -> AsyncIterator[tuple[str, dict]]:
        """群聊自主闲聊：用户点击按钮触发，角色之间互相搭话（无用户消息）。

        事件与 `stream` 同构：`speaker` / `token` / `message_break` / `turn_done` / `done`。
        每个角色按轮转顺序各发一次、每次不超过 `idle_turn_max_messages` 条，
        累计不超过 `idle_max_messages`；落库行标记 `is_idle=True`，`done.message_id=0`。
        """
        group_session_id = context.group_session_id
        async with self._locks.acquire(f"group:{group_session_id}"):
            rotate_after = await self._last_speaker(group_session_id)
            # V2.4：群聊自主闲聊同样按发言人各一条 trace（mode="group_idle"）
            parent_trace_id = uuid.uuid4().hex
            supervisor = Supervisor(
                llm=self._llm,
                graph_factory=self._graph_factory,
                members=context.members,
                max_turns=self._max_turns,
                tool_executor=self._tool_executor,
            )
            order: list[int] = []
            replies: list[dict] = []
            async for event, data in supervisor.run_idle(
                group_session_id=group_session_id,
                max_total=self._idle_max_messages,
                turn_max_messages=self._idle_turn_max_messages,
                rotate_after=rotate_after,
                parent_trace_id=parent_trace_id,
            ):
                if event == "turn_done":
                    agent_id = int(data["agent_id"])
                    parts = [str(p) for p in (data.get("replies") or [])] or [
                        str(data.get("reply", ""))
                    ]
                    thinkings = normalize_thinkings(
                        [str(t) for t in (data.get("thinkings") or [])], len(parts)
                    )
                    for i, part in enumerate(parts):
                        reply_id = await self._append(
                            group_session_id,
                            agent_id=agent_id,
                            role="assistant",
                            content=part,
                            is_idle=True,
                            thinking=thinkings[i] or None,
                        )
                        await self._fanout(
                            context,
                            part,
                            role="assistant",
                            speaker_agent_id=agent_id,
                            exclude=frozenset({agent_id}),
                        )
                        replies.append(
                            {
                                "agent_id": agent_id,
                                "message_id": reply_id,
                                "content": part,
                                "thinking": thinkings[i] or None,
                            }
                        )
                    order.append(agent_id)
                yield event, data

            yield "done", {
                "group_session_id": group_session_id,
                "message_id": 0,
                "order": order,
                "replies": replies,
                "idle": True,
            }

    async def clear_idle(self, *, group_session_id: int) -> list[int]:
        """清空该群的全部自主闲聊：删快照行 + 删全体成员对应记忆 + 清各成员工作记忆窗口。

        返回被移除的快照消息 id 列表。
        """
        context = await self.load_context(group_session_id)
        async with self._locks.acquire(f"group:{group_session_id}"):
            async with self._session_factory() as db:
                rows = (
                    (
                        await db.execute(
                            select(GroupMessage)
                            .where(
                                GroupMessage.group_session_id == group_session_id,
                                GroupMessage.is_idle.is_(True),
                            )
                            .order_by(GroupMessage.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                removed = [r.id for r in rows]
                if removed:
                    await db.execute(delete(GroupMessage).where(GroupMessage.id.in_(removed)))
                    await db.commit()

            if not removed:
                return []

            # 业务表已落库；以下为尽力清理记忆与工作记忆（与 recall 同构）
            name_by_id = {m.agent_id: m.name for m in context.members}
            threads = {m.agent_id: f"g{group_session_id}:{m.agent_id}" for m in context.members}
            contents_all = {r.content for r in rows}
            prefixed: set[str] = set()
            for r in rows:
                name = name_by_id.get(r.agent_id if r.agent_id is not None else -1)
                if name:
                    prefixed.add(f"【{name}】{r.content}")
            targets = contents_all | prefixed
            graphs = {aid: self._graph_factory() for aid in threads}
            for agent_id, thread_id in threads.items():
                # 他人发言在本成员 namespace 里带「【角色名】」前缀 → raw 与带前缀两种都删
                for content in targets:
                    await self._memory_store.delete(agent_id, content)
                await remove_thread_messages(graphs[agent_id], thread_id, targets)
            return removed

    async def recall(self, *, group_session_id: int, message_id: int) -> list[int]:
        """撤回一条群聊用户消息：整轮（含本轮全部角色回复）从快照/记忆/工作记忆移除。

        - 记忆：该消息与本轮各回复已扇出至全体成员 namespace，逐个按内容精确删除；
        - 工作记忆：逐成员 thread 删除本轮消息（他人回复在该成员线程里带「【角色名】」前缀）；
        - 仅允许撤回仍在工作记忆窗口内的消息；已归档为会话摘要的拒绝撤回。
        返回被移除的消息 id 列表（用户消息 + 本轮全部回复）。
        """
        context = await self.load_context(group_session_id)
        async with self._locks.acquire(f"group:{group_session_id}"):
            async with self._session_factory() as db:
                target = await db.get(GroupMessage, message_id)
                if target is None or target.group_session_id != group_session_id:
                    raise NotFoundError(f"message {message_id} 不存在")
                if target.role != "user":
                    raise AppError("只能撤回自己发送的消息")

                # 本轮全部回复：id 更大的消息，直到下一条用户消息之前
                rows = (
                    (
                        await db.execute(
                            select(GroupMessage)
                            .where(
                                GroupMessage.group_session_id == group_session_id,
                                GroupMessage.id > message_id,
                            )
                            .order_by(GroupMessage.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                replies: list[GroupMessage] = []
                for row in rows:
                    if row.role == "user" or row.is_idle:
                        break  # 遇到下一条用户消息或自主闲聊 → 本轮回复结束
                    replies.append(row)

                name_by_id = {m.agent_id: m.name for m in context.members}
                threads = {m.agent_id: f"g{group_session_id}:{m.agent_id}" for m in context.members}

                # 可撤回性：用户消息须仍出现在至少一个成员的工作记忆窗口内。
                # 本轮未发言的成员不会把该消息写进自己的 thread（甚至没有 checkpoint），
                # 不应据此拒绝；若处处都不含该消息，则说明已归档为会话摘要 → 拒绝。
                graphs = {aid: self._graph_factory() for aid in threads}
                present_anywhere = False
                for agent_id, thread_id in threads.items():
                    if target.content in await thread_message_contents(graphs[agent_id], thread_id):
                        present_anywhere = True
                        break
                if not present_anywhere:
                    raise AppError("该消息已超出可撤回范围（可能已归档为会话摘要），无法撤回")

                removed = [target.id] + [r.id for r in replies]
                await db.execute(delete(GroupMessage).where(GroupMessage.id.in_(removed)))
                await db.commit()

            # 业务表已落库；以下为尽力清理记忆与工作记忆
            contents_all = {target.content}
            prefixed: set[str] = set()
            for r in replies:
                contents_all.add(r.content)
                name = name_by_id.get(r.agent_id if r.agent_id is not None else -1)
                if name:
                    prefixed.add(f"【{name}】{r.content}")
            targets = contents_all | prefixed
            for agent_id, thread_id in threads.items():
                # 他人发言在本成员 namespace 里带「【角色名】」前缀 → raw 与带前缀两种都删
                for content in targets:
                    await self._memory_store.delete(agent_id, content)
                await remove_thread_messages(graphs[agent_id], thread_id, targets)
            return removed
