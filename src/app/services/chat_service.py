"""对话服务：串联 LangGraph 图与业务库读写。

接缝 1（方案 3.11A#1）：thread_id = f"{session_id}:{agent_id}"，禁止裸 session_id。
接缝 5：业务库访问只在本服务层；persona 预取后注入图状态，图内零业务表访问。
时序（方案 3.7）：图执行完成（update_memory 已落 Store）后，才写 messages 快照；
SSE 流式下 `done` 事件必须在快照落库之后才推送。
"""

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.nodes.generate import normalize_thinkings
from app.agents.thread_state import (
    remove_thread_message_by_id,
    remove_thread_messages,
    thread_message_contents,
)
from app.config import get_settings
from app.core import clock
from app.core.errors import AppError, NotFoundError
from app.core.locks import KeyedLockPool
from app.db.models import Agent, Message
from app.db.models import Session as ChatSession
from app.memory.store import MemoryStore
from app.observability.trace import (
    current_error,
    is_degraded,
    reset_trace,
    start_trace,
)
from app.schemas.chat import ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class ChatContext:
    """一次对话所需的服务层上下文（persona 预取结果）。"""

    session_id: int
    agent_id: int
    agent_name: str
    persona_prompt: str
    background: str
    # V2.2：本角色工具白名单（Agent.allowed_skills）；空 = 不绑工具（契约 §2.4）
    allowed_skills: tuple[str, ...] = ()

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
        tool_executor=None,
        idempotency_ttl_s: float | None = None,
        idempotency_max_entries: int | None = None,
    ) -> None:
        self._graph = graph
        self._session_factory = session_factory
        self._memory_store = memory_store
        # V2.2：每次图运行前清空工具缓存（同请求缓存，契约 §2.2-5）
        self._tool_executor = tool_executor
        # 按 thread_id 串行化：同会话并发写 checkpointer 会造成分支覆盖（后写覆盖先写）。
        # V2.4 B-02：引用计数锁池，退出即释放条目（单进程语义；多实例需分布式锁，记技术债）。
        self._locks = KeyedLockPool()
        # V2.4 D8：非流式 /chat 的进程内幂等缓存（单进程语义；重启/多实例失效，记 §7）
        settings = get_settings()
        ttl = idempotency_ttl_s if idempotency_ttl_s is not None else settings.chat_idempotency_ttl_s
        limit = (
            idempotency_max_entries
            if idempotency_max_entries is not None
            else settings.chat_idempotency_max_entries
        )
        self._idempotency_ttl_s = max(0.0, float(ttl))
        self._idempotency_max_entries = max(1, int(limit))
        self._idempotency: OrderedDict[tuple[int, str], tuple[float, ChatResponse]] = OrderedDict()

    def _reset_tools(self) -> None:
        if self._tool_executor is not None:
            self._tool_executor.reset()

    # ===== 非流式幂等（D8）=====

    def _idempotent_lookup(
        self, session_id: int, client_msg_id: str | None
    ) -> ChatResponse | None:
        """命中返回 `idempotent=true` 的副本；过期条目顺手剔除；空 key 不缓存。"""
        if not client_msg_id:
            return None
        key = (session_id, client_msg_id)
        item = self._idempotency.get(key)
        if item is None:
            return None
        expires_at, response = item
        if expires_at <= time.monotonic():
            self._idempotency.pop(key, None)
            return None
        self._idempotency.move_to_end(key)
        return response.model_copy(update={"idempotent": True})

    def _idempotent_store(
        self, session_id: int, client_msg_id: str | None, response: ChatResponse
    ) -> None:
        if not client_msg_id:
            return
        key = (session_id, client_msg_id)
        now = time.monotonic()
        self._idempotency[key] = (now + self._idempotency_ttl_s, response)
        self._idempotency.move_to_end(key)
        self._idempotent_evict(now)

    def _idempotent_evict(self, now: float) -> None:
        """清过期 + 超容量时按 LRU 淘汰（TTL 恒定，过期条目天然在队首）。"""
        while self._idempotency:
            key, (expires_at, _) = next(iter(self._idempotency.items()))
            if expires_at > now and len(self._idempotency) <= self._idempotency_max_entries:
                break
            self._idempotency.popitem(last=False)

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
                allowed_skills=tuple(agent.allowed_skills or ()),
            )

    @staticmethod
    def _state_in(
        context: ChatContext,
        message: str,
        *,
        message_id: str | None = None,
        last_user_at: str | None = None,
    ) -> dict:
        """把预取的人设 + 本轮用户消息注入图状态（图内只读状态）。

        `idle_scope` 每次显式置空：状态被 checkpointer 持久化，若不重置，
        上一次闲聊轮残留的标记会污染本次正常回复。

        `message_id` 显式预置到 HumanMessage 上：流式请求中途失败时可凭此 id
        精确回滚本轮用户消息，避免在工作记忆里留下「界面不可见却影响后续每轮」的孤儿消息。

        V3 M1：本轮用户消息带 `at`（ISO 8601 带 +08:00），供 prompt 渲染时间前缀；
        `last_user_at` 为上一条 user 消息时间，供“你们上次说话是 …”一行。
        """
        return {
            "messages": [
                HumanMessage(
                    content=message,
                    id=message_id or str(uuid.uuid4()),
                    additional_kwargs={"at": clock.now_iso()},
                )
            ],
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "agent_name": context.agent_name,
            "persona_prompt": context.persona_prompt,
            "background": context.background,
            "last_user_at": last_user_at or "",
            "idle_scope": "",
            "reply_max_messages": 0,
            "allowed_skills": list(context.allowed_skills),
            # 工具状态按「每轮请求」重新开始：tool_rounds/tool_calls 是普通字段、
            # tool_results 是 operator.add 累积字段，二者都会被 checkpointer 跨轮保留。
            # 不重置会让上一轮的轮数一直累加，最终 `forced=True` 导致该会话永久无法调工具。
            "tool_rounds": 0,
            "tool_calls": [],
            "tool_turn": uuid.uuid4().hex,
        }

    @staticmethod
    def _idle_state_in(context: ChatContext, *, last_user_at: str | None = None) -> dict:
        """自主闲聊的图状态输入：不注入用户消息，仅置 idle_scope=private。

        V3 M1：闲聊轮没有用户消息（不加 `at`），但同样要传 `last_user_at`。
        """
        return {
            "messages": [],
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "agent_name": context.agent_name,
            "persona_prompt": context.persona_prompt,
            "background": context.background,
            "last_user_at": last_user_at or "",
            "idle_scope": "private",
            "reply_max_messages": 0,  # 私聊闲聊用图默认上限（reply_max_messages=3）
            "allowed_skills": list(context.allowed_skills),
            # 同 `_state_in`：工具状态逐轮重置，避免闲聊轮继承上一轮的工具轮数
            "tool_rounds": 0,
            "tool_calls": [],
            "tool_turn": uuid.uuid4().hex,
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

    async def _last_user_at(self, session_id: int) -> str | None:
        """上一条 user 消息的落库时间（ISO 8601 带 +08:00）；查不到/失败返回 None。

        V3 M1 D10：每轮一次索引友好的 SELECT；失败只 warning，不阻断本轮。
        查询发生在图启动前，此时上一轮快照已落库（私聊快照在 graph 结束后写）。
        """
        try:
            async with self._session_factory() as db:
                created = (
                    await db.execute(
                        select(Message.created_at)
                        .where(Message.session_id == session_id, Message.role == "user")
                        .order_by(Message.id.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
        except Exception as exc:  # noqa: BLE001 - 时间感是增强项，不能成为可用性风险
            logger.warning("取 last_user_at 失败（session=%s）：%s", session_id, exc)
            return None
        if created is None:
            return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=clock.TZ)
        return created.astimezone(clock.TZ).isoformat(timespec="seconds")

    async def _next_turn(self, session_id: int) -> int | None:
        """该会话的用户消息序号（1-based）；取不到时为 None（不伪造）。"""
        try:
            async with self._session_factory() as db:
                count = (
                    await db.execute(
                        select(func.count())
                        .select_from(Message)
                        .where(Message.session_id == session_id, Message.role == "user")
                    )
                ).scalar_one()
            return int(count) + 1
        except Exception:  # noqa: BLE001 - trace 旁路失败不影响主流程
            return None

    async def chat(
        self, *, session_id: int, message: str, client_msg_id: str | None = None
    ) -> ChatResponse:
        """非流式对话（M1 保留，便于 curl / 集成测试）。

        V2.4 D8：带 `client_msg_id` 时同 `(session_id, client_msg_id)` 在 TTL 内返回缓存，
        不写 messages、不调图；命中结果 `idempotent=true`。空 key 行为与 V2.3 一致。
        """
        cached = self._idempotent_lookup(session_id, client_msg_id)
        if cached is not None:
            return cached
        context = await self.load_context(session_id=session_id)
        # R6：turn 必须在会话锁内、上一轮快照落库之后计算，否则同会话并发的两轮
        # 都会在写快照前取到相同序号（B-02 实际串行，第二轮应为 N+2）。
        async with self._locks.acquire(context.thread_id):
            # 进锁后二次检查：并发同 key 时只有第一个真正执行（防双写）
            cached = self._idempotent_lookup(session_id, client_msg_id)
            if cached is not None:
                return cached
            last_user_at = await self._last_user_at(context.session_id)
            recorder, token = start_trace(
                mode="private",
                session_id=context.session_id,
                agent_id=context.agent_id,
                turn=await self._next_turn(context.session_id),
            )
            try:
                self._reset_tools()
                result = await self._graph.ainvoke(
                    self._state_in(context, message, last_user_at=last_user_at),
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
                response = ChatResponse(
                    session_id=context.session_id,
                    agent_id=context.agent_id,
                    reply="\n".join(replies),
                )
                # 成功后才写缓存；异常/取消不缓存，重试可重新执行
                self._idempotent_store(context.session_id, client_msg_id, response)
                recorder.finish(error=current_error(), degraded=is_degraded())
                return response
            except BaseException as exc:
                recorder.finish(
                    error=current_error()
                    or {"type": type(exc).__name__, "message": str(exc)},
                    degraded=is_degraded(),
                )
                raise
            finally:
                reset_trace(token)

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
        # V2.4：每轮一条 trace；首个 token 事件记录 first_token_ms，done 前收敛写入（§2.3）
        # R6：turn 在会话锁内计算（上一轮快照已落库），同会话并发才不会取到同一序号。
        finished = False
        async with self._locks.acquire(context.thread_id):
            last_user_at = await self._last_user_at(context.session_id)
            recorder, token = start_trace(
                mode="private",
                session_id=context.session_id,
                agent_id=context.agent_id,
                turn=await self._next_turn(context.session_id),
            )
            try:
                completed = False
                try:
                    self._reset_tools()
                    async for mode, chunk in self._graph.astream(
                        self._state_in(
                            context,
                            message,
                            message_id=input_message_id,
                            last_user_at=last_user_at,
                        ),
                        config={"configurable": {"thread_id": context.thread_id}},
                        stream_mode=["custom", "values"],
                    ):
                        if mode == "custom":
                            if isinstance(chunk, dict) and chunk.get("break"):
                                yield ("message_break", {})
                                continue
                            text = str(chunk.get("text", "")) if isinstance(chunk, dict) else ""
                            if text:
                                recorder.first_token()
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

                # 快照落库后收敛 trace（done 事件在锁外产出）
                recorder.finish(error=current_error(), degraded=is_degraded())
                finished = True
            except BaseException as exc:
                if not finished:
                    recorder.finish(
                        error=current_error()
                        or {"type": type(exc).__name__, "message": str(exc)},
                        degraded=is_degraded(),
                    )
                raise
            finally:
                reset_trace(token)

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
        # V2.4：自主闲聊同样一条 trace（mode="idle"，无用户轮次 → turn=None）
        recorder, token = start_trace(
            mode="idle",
            session_id=context.session_id,
            agent_id=context.agent_id,
            turn=None,
        )
        finished = False
        try:
            async with self._locks.acquire(context.thread_id):
                last_user_at = await self._last_user_at(context.session_id)
                self._reset_tools()
                async for mode, chunk in self._graph.astream(
                    self._idle_state_in(context, last_user_at=last_user_at),
                    config={"configurable": {"thread_id": context.thread_id}},
                    stream_mode=["custom", "values"],
                ):
                    if mode == "custom":
                        if isinstance(chunk, dict) and chunk.get("break"):
                            yield ("message_break", {})
                            continue
                        text = str(chunk.get("text", "")) if isinstance(chunk, dict) else ""
                        if text:
                            recorder.first_token()
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

            recorder.finish(error=current_error(), degraded=is_degraded())
            finished = True
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
        except BaseException as exc:
            if not finished:
                recorder.finish(
                    error=current_error()
                    or {"type": type(exc).__name__, "message": str(exc)},
                    degraded=is_degraded(),
                )
            raise
        finally:
            reset_trace(token)

    async def clear_idle(self, *, session_id: int) -> list[int]:
        """清空该会话的全部自主闲聊：删快照行 + 删对应语义记忆 + 清工作记忆窗口。

        返回被移除的快照消息 id 列表。
        """
        context = await self.load_context(session_id=session_id)
        async with self._locks.acquire(context.thread_id):
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
        async with self._locks.acquire(context.thread_id):
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
