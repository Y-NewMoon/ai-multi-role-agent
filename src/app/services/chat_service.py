"""对话服务：串联 LangGraph 图与业务库读写。

接缝 1（方案 3.11A#1）：thread_id = f"{session_id}:{agent_id}"，禁止裸 session_id。
接缝 5：业务库访问只在本服务层；persona 预取后注入图状态，图内零业务表访问。
时序（方案 3.7）：图执行完成（update_memory 已落 Store）后，才写 messages 快照；
SSE 流式下 `done` 事件必须在快照落库之后才推送。
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.nodes.generate import normalize_thinkings
from app.agents.thread_state import (
    remove_thread_message_by_id,
    remove_thread_messages,
    thread_message_contents,
)
from app.core.errors import AppError, NotFoundError
from app.db.models import Agent, Message
from app.db.models import Session as ChatSession
from app.memory.store import MemoryStore
from app.schemas.chat import ChatResponse


@dataclass
class ChatContext:
    """一次对话所需的服务层上下文（persona 预取结果）。"""

    session_id: int
    agent_id: int
    agent_name: str
    persona_prompt: str
    background: str

    @property
    def thread_id(self) -> str:
        """接缝 1：thread_id 必须含 agent_id，不得使用裸 session_id。"""
        return f"{self.session_id}:{self.agent_id}"


# 回滚任务的强引用集合：shield 后外层若被取消，任务仍在后台执行，
# 必须持有引用避免被 GC 提前回收（否则清理可能半途而废）。
_rollback_tasks: set[asyncio.Task] = set()


async def _rollback_input_message(graph, thread_id: str, message_id: str) -> None:
    """尽力从工作记忆删除本轮注入的用户消息；失败只吞异常，绝不掩盖原始错误。"""
    try:
        await remove_thread_message_by_id(graph, thread_id, message_id)
    except BaseException:  # noqa: BLE001 - 回滚是尽力而为，不得影响原始异常/取消传播
        pass


class ChatService:
    def __init__(
        self,
        *,
        graph,
        session_factory: async_sessionmaker[AsyncSession],
        memory_store: MemoryStore,
    ) -> None:
        self._graph = graph
        self._session_factory = session_factory
        self._memory_store = memory_store
        # 按 thread_id 串行化：同会话并发写 checkpointer 会造成分支覆盖（后写覆盖先写）。
        # 单进程内存锁足以覆盖当前单实例部署；多实例部署需换成分布式锁（记入技术债）。
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_id] = lock
        return lock

    async def load_context(self, *, session_id: int) -> ChatContext:
        """读业务库取会话与角色档案（仅服务层）。

        以 `session.agent_id` 为唯一依据（方案 3.1 / 3.11A），请求体 agent_id 一律忽略。
        session/agent 不存在时抛 NotFoundError（SSE 端点在开始推流前即可返回 404）。
        """
        async with self._session_factory() as db:
            session = await db.get(ChatSession, session_id)
            if session is None:
                raise NotFoundError(f"session {session_id} 不存在")
            agent = await db.get(Agent, session.agent_id)
            if agent is None:
                raise NotFoundError(f"agent {session.agent_id} 不存在")
            return ChatContext(
                session_id=session.id,
                agent_id=agent.id,
                agent_name=agent.name,
                persona_prompt=agent.persona_prompt,
                background=agent.background or "",
            )

    @staticmethod
    def _state_in(context: ChatContext, message: str, *, message_id: str | None = None) -> dict:
        """把预取的人设 + 本轮用户消息注入图状态（图内只读状态）。

        `idle_scope` 每次显式置空：状态被 checkpointer 持久化，若不重置，
        上一次闲聊轮残留的标记会污染本次正常回复。

        `message_id` 显式预置到 HumanMessage 上：流式请求中途失败时可凭此 id
        精确回滚本轮用户消息，避免在工作记忆里留下「界面不可见却影响后续每轮」的孤儿消息。
        """
        return {
            "messages": [HumanMessage(content=message, id=message_id or str(uuid.uuid4()))],
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "agent_name": context.agent_name,
            "persona_prompt": context.persona_prompt,
            "background": context.background,
            "idle_scope": "",
            "reply_max_messages": 0,
        }

    @staticmethod
    def _idle_state_in(context: ChatContext) -> dict:
        """自主闲聊的图状态输入：不注入用户消息，仅置 idle_scope=private。"""
        return {
            "messages": [],
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "agent_name": context.agent_name,
            "persona_prompt": context.persona_prompt,
            "background": context.background,
            "idle_scope": "private",
            "reply_max_messages": 0,  # 私聊闲聊用图默认上限（reply_max_messages=3）
        }

    async def _write_snapshot(
        self, session_id: int, message: str, replies: list[str], thinkings: list[str]
    ) -> tuple[int, list[int]]:
        """写展示用 messages 快照（真相源是 checkpointer，方案 3.7）。

        M5：角色一轮可连发多条，逐条落一行 assistant，返回 (用户消息 id, 回复 id 列表)。
        M6：每条回复带上对应的思考链（thinkings 已与 replies 等长对齐）。
        """
        async with self._session_factory() as db:
            user_row = Message(session_id=session_id, role="user", content=message)
            db.add(user_row)
            reply_rows = [
                Message(
                    session_id=session_id,
                    role="assistant",
                    content=r,
                    thinking=thinkings[i] or None,
                )
                for i, r in enumerate(replies)
            ]
            for row in reply_rows:
                db.add(row)
            await db.commit()
            await db.refresh(user_row)
            for row in reply_rows:
                await db.refresh(row)
            return user_row.id, [row.id for row in reply_rows]

    async def _write_idle_snapshot(
        self, session_id: int, replies: list[str], thinkings: list[str]
    ) -> list[int]:
        """写自主闲聊的展示快照：只有 assistant 行（无用户消息），且标记 is_idle=True。"""
        async with self._session_factory() as db:
            rows = [
                Message(
                    session_id=session_id,
                    role="assistant",
                    content=r,
                    is_idle=True,
                    thinking=thinkings[i] or None,
                )
                for i, r in enumerate(replies)
            ]
            for row in rows:
                db.add(row)
            await db.commit()
            for row in rows:
                await db.refresh(row)
            return [row.id for row in rows]

    async def chat(self, *, session_id: int, message: str) -> ChatResponse:
        """非流式对话（M1 保留，便于 curl / 集成测试）。"""
        context = await self.load_context(session_id=session_id)
        async with self._lock_for(context.thread_id):
            result = await self._graph.ainvoke(
                self._state_in(context, message),
                config={"configurable": {"thread_id": context.thread_id}},
            )
            replies = [str(r) for r in (result.get("replies") or [])] or [
                str(result.get("reply", ""))
            ]
            thinkings = normalize_thinkings(
                [str(t) for t in (result.get("thinkings") or [])], len(replies)
            )
            # 图执行完成（记忆已落 Store）后写展示用快照
            await self._write_snapshot(context.session_id, message, replies, thinkings)
        return ChatResponse(
            session_id=context.session_id,
            agent_id=context.agent_id,
            reply="\n".join(replies),
        )

    async def stream(
        self, *, context: ChatContext, message: str
    ) -> AsyncIterator[tuple[str, dict]]:
        """流式对话，产出结构化事件 `(event_name, payload)`。

        事件名：`token`（增量文本）/ `message_break`（角色连发的分条边界）/ `done`（本轮结束）。
        时序严格守方案 3.7：逐 token 产出 → 图执行完成（update_memory 已落 Store）
        → 服务层写 messages 快照 → 才产出 `done`。
        """
        replies: list[str] = []
        thinkings: list[str] = []
        reply_text = ""
        user_msg_id = 0
        reply_ids: list[int] = []
        input_message_id = str(uuid.uuid4())
        async with self._lock_for(context.thread_id):
            completed = False
            try:
                async for mode, chunk in self._graph.astream(
                    self._state_in(context, message, message_id=input_message_id),
                    config={"configurable": {"thread_id": context.thread_id}},
                    stream_mode=["custom", "values"],
                ):
                    if mode == "custom":
                        if isinstance(chunk, dict) and chunk.get("break"):
                            yield ("message_break", {})
                            continue
                        text = str(chunk.get("text", "")) if isinstance(chunk, dict) else ""
                        if text:
                            yield ("token", {"text": text})
                    elif mode == "values":
                        # 每步状态快照；generate 完成后 chunk["replies"] 即本轮全部回复
                        if chunk.get("replies"):
                            replies = [str(r) for r in chunk["replies"]]
                        if chunk.get("reply") is not None:
                            reply_text = str(chunk["reply"])
                        if chunk.get("thinkings") is not None:
                            thinkings = [str(t) for t in chunk["thinkings"]]

                if not replies:
                    replies = [reply_text]
                thinkings = normalize_thinkings(thinkings, len(replies))

                # 图已到 END：update_memory 节点已把记忆/摘要落 Store（3.7 第 3 步）
                user_msg_id, reply_ids = await self._write_snapshot(
                    context.session_id, message, replies, thinkings
                )
                completed = True
            finally:
                if not completed:
                    # 图在起步时就把本轮用户消息写进了工作记忆；若此处因异常/取消
                    # 未走到快照落库，则该消息会成为「界面不可见却永久影响后续每轮」的
                    # 孤儿消息 → 回滚删除它（尽力而为，不掩盖原始异常）。
                    await self._discard_input(context.thread_id, input_message_id)

        # 快照落库后才产出 done（3.7 第 4 步）
        yield "done", {
            "session_id": context.session_id,
            "agent_id": context.agent_id,
            "reply": "\n".join(replies),
            "replies": [
                {"content": content, "message_id": mid, "thinking": th or None}
                for content, mid, th in zip(replies, reply_ids, thinkings, strict=False)
            ],
            "user_message_id": user_msg_id,
        }

    async def _discard_input(self, thread_id: str, message_id: str) -> None:
        """回滚本轮注入的工作记忆用户消息（取消安全）。

        用独立 task + `shield` 承载清理：外层被取消时清理仍能完成，不会被半途打断；
        任务持有强引用避免被 GC 回收。清理内部吞掉一切异常，绝不掩盖原始错误。
        """
        task = asyncio.ensure_future(_rollback_input_message(self._graph, thread_id, message_id))
        _rollback_tasks.add(task)
        task.add_done_callback(_rollback_tasks.discard)
        try:
            await asyncio.shield(task)
        except BaseException:  # noqa: BLE001 - 外层取消/异常时不阻塞收尾
            pass

    async def idle_stream(self, *, context: ChatContext) -> AsyncIterator[tuple[str, dict]]:
        """自主闲聊（私聊）：用户点击按钮触发，产出与 `stream` 同构的事件。

        - 不注入用户消息（`messages=[]`），仅置 `idle_scope="private"`；
        - 落库的 assistant 快照标记 `is_idle=True`，无用户行；
        - `done` 载荷 `user_message_id=0`、`idle=True`。
        """
        replies: list[str] = []
        thinkings: list[str] = []
        reply_text = ""
        reply_ids: list[int] = []
        async with self._lock_for(context.thread_id):
            async for mode, chunk in self._graph.astream(
                self._idle_state_in(context),
                config={"configurable": {"thread_id": context.thread_id}},
                stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    if isinstance(chunk, dict) and chunk.get("break"):
                        yield ("message_break", {})
                        continue
                    text = str(chunk.get("text", "")) if isinstance(chunk, dict) else ""
                    if text:
                        yield ("token", {"text": text})
                elif mode == "values":
                    if chunk.get("replies"):
                        replies = [str(r) for r in chunk["replies"]]
                    if chunk.get("reply") is not None:
                        reply_text = str(chunk["reply"])
                    if chunk.get("thinkings") is not None:
                        thinkings = [str(t) for t in chunk["thinkings"]]

            if not replies:
                replies = [reply_text]
            thinkings = normalize_thinkings(thinkings, len(replies))

            reply_ids = await self._write_idle_snapshot(context.session_id, replies, thinkings)

        yield "done", {
            "session_id": context.session_id,
            "agent_id": context.agent_id,
            "reply": "\n".join(replies),
            "replies": [
                {"content": content, "message_id": mid, "thinking": th or None}
                for content, mid, th in zip(replies, reply_ids, thinkings, strict=False)
            ],
            "user_message_id": 0,
            "idle": True,
        }

    async def clear_idle(self, *, session_id: int) -> list[int]:
        """清空该会话的全部自主闲聊：删快照行 + 删对应语义记忆 + 清工作记忆窗口。

        返回被移除的快照消息 id 列表。
        """
        context = await self.load_context(session_id=session_id)
        async with self._lock_for(context.thread_id):
            async with self._session_factory() as db:
                rows = (
                    (
                        await db.execute(
                            select(Message)
                            .where(Message.session_id == session_id, Message.is_idle.is_(True))
                            .order_by(Message.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                removed = [r.id for r in rows]
                if removed:
                    await db.execute(delete(Message).where(Message.id.in_(removed)))
                    await db.commit()

            if not removed:
                return []

            # 业务表已落库；以下为尽力清理记忆与工作记忆（失败不回滚，仅留孤儿数据）
            targets = {r.content for r in rows}
            for content in targets:
                await self._memory_store.delete(context.agent_id, content)
            await remove_thread_messages(self._graph, context.thread_id, targets)
            return removed

    async def recall(self, *, session_id: int, message_id: int) -> list[int]:
        """撤回一条用户消息：整轮（含紧随的**全部**角色回复）从快照/记忆/工作记忆移除。

        仅允许撤回仍在 checkpointer 工作记忆窗口内的消息；已归档为会话摘要的拒绝撤回。
        返回被移除的消息 id 列表（用户消息 + 连带回复）。
        """
        context = await self.load_context(session_id=session_id)
        async with self._lock_for(context.thread_id):
            async with self._session_factory() as db:
                target = await db.get(Message, message_id)
                if target is None or target.session_id != session_id:
                    raise NotFoundError(f"message {message_id} 不存在")
                if target.role != "user":
                    raise AppError("只能撤回自己发送的消息")

                # 本轮全部回复：id 更大的消息，直到下一条用户消息之前（M5：可能多条）
                rows = (
                    (
                        await db.execute(
                            select(Message)
                            .where(Message.session_id == session_id, Message.id > message_id)
                            .order_by(Message.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                replies: list[Message] = []
                for row in rows:
                    if row.role != "assistant" or row.is_idle:
                        break  # 遇到下一条用户消息或自主闲聊 → 本轮回复结束
                    replies.append(row)

                # 可撤回性：用户消息须仍在工作记忆窗口内（否则已归档为摘要）
                contents = await thread_message_contents(self._graph, context.thread_id)
                if target.content not in contents:
                    raise AppError("该消息已超出可撤回范围（可能已归档为会话摘要），无法撤回")

                removed = [target.id] + [r.id for r in replies]
                await db.execute(delete(Message).where(Message.id.in_(removed)))
                await db.commit()

            # 业务表已落库；以下为尽力清理记忆与工作记忆（失败不回滚，仅留孤儿数据）
            targets = {target.content}
            await self._memory_store.delete(context.agent_id, target.content)
            for reply in replies:
                await self._memory_store.delete(context.agent_id, reply.content)
                targets.add(reply.content)
            await remove_thread_messages(self._graph, context.thread_id, targets)
            return removed
