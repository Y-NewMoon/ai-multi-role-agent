"""ChatService 单测：快照写入、异常传播、同会话串行化（接缝 1 并发安全）。"""

import asyncio

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agents.graph import build_graph
from app.core.errors import NotFoundError, UpstreamError
from app.db.models import Agent
from app.db.models import Session as ChatSession
from app.memory.store import MemoryStore
from app.services.chat_service import ChatService


class _FakeMemoryStore:
    """满足 ChatService 构造依赖；这些用例不触发撤回，删除为 no-op。"""

    async def delete(self, agent_id, content: str) -> None:
        pass


_STORE = _FakeMemoryStore()


class FakeSession:
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
        # 模拟落库后回填主键（_write_snapshot 需要它返回消息 id）
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added)


class _Ctx:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> FakeSession:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


class FakeSessionFactory:
    def __init__(self, tables: dict) -> None:
        self._tables = tables
        self.sessions: list[FakeSession] = []

    def __call__(self) -> _Ctx:
        session = FakeSession(self._tables)
        self.sessions.append(session)
        return _Ctx(session)


class FakeGraph:
    def __init__(
        self,
        *,
        reply: str = "回复",
        exc: Exception | None = None,
        delay: float = 0.0,
        tokens: list[str] | None = None,
    ) -> None:
        self.reply = reply
        self.exc = exc
        self.delay = delay
        self.tokens = list(tokens) if tokens is not None else [reply]
        self.active = 0
        self.max_active = 0

    async def ainvoke(self, state, config):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.exc is not None:
                raise self.exc
            return {"reply": self.reply}
        finally:
            self.active -= 1

    async def astream(self, state, config, stream_mode=None):
        """模拟 LangGraph.astream：先逐 token 出 custom，再出最终 values。"""
        if self.exc is not None:
            raise self.exc
        for token in self.tokens:
            yield ("custom", {"text": token})
        yield ("values", {"reply": self.reply})


def _tables(*, session_id: int = 1, agent_id: int = 1, with_agent: bool = True) -> dict:
    agent = Agent(id=agent_id, name="四季夏目", persona_prompt="茶艺师", background="古镇")
    session = ChatSession(id=session_id, agent_id=agent_id)
    tables = {(ChatSession, session_id): session}
    if with_agent:
        tables[(Agent, agent_id)] = agent
    return tables


async def test_chat_writes_user_and_assistant_snapshot():
    factory = FakeSessionFactory(_tables())
    service = ChatService(
        graph=FakeGraph(reply="你好呀，客人。"), session_factory=factory, memory_store=_STORE
    )

    resp = await service.chat(session_id=1, message="在吗")

    assert resp.session_id == 1
    assert resp.agent_id == 1
    assert resp.reply == "你好呀，客人。"
    assert [m.role for m in factory.sessions[-1].added] == ["user", "assistant"]


async def test_chat_raises_not_found_for_missing_session():
    service = ChatService(
        graph=FakeGraph(), session_factory=FakeSessionFactory({}), memory_store=_STORE
    )
    with pytest.raises(NotFoundError):
        await service.chat(session_id=404, message="hi")


async def test_chat_raises_not_found_for_missing_agent():
    service = ChatService(
        graph=FakeGraph(),
        session_factory=FakeSessionFactory(_tables(with_agent=False)),
        memory_store=_STORE,
    )
    with pytest.raises(NotFoundError):
        await service.chat(session_id=1, message="hi")


async def test_chat_propagates_upstream_error():
    factory = FakeSessionFactory(_tables())
    service = ChatService(
        graph=FakeGraph(exc=UpstreamError("模型服务调用失败")),
        session_factory=factory,
        memory_store=_STORE,
    )
    with pytest.raises(UpstreamError):
        await service.chat(session_id=1, message="hi")


async def test_same_session_requests_are_serialized():
    graph = FakeGraph(delay=0.02)
    service = ChatService(
        graph=graph, session_factory=FakeSessionFactory(_tables()), memory_store=_STORE
    )

    await asyncio.gather(
        service.chat(session_id=1, message="a"),
        service.chat(session_id=1, message="b"),
        service.chat(session_id=1, message="c"),
    )

    assert graph.max_active == 1  # 同一 thread_id 未发生并发进入图


class _RecordingSession(FakeSession):
    """记录快照落库时点的 FakeSession。"""

    def __init__(self, tables: dict, timeline: list) -> None:
        super().__init__(tables)
        self._timeline = timeline

    async def commit(self) -> None:
        self._timeline.append("snapshot")


class _RecordingFactory(FakeSessionFactory):
    def __init__(self, tables: dict, timeline: list) -> None:
        super().__init__(tables)
        self.timeline = timeline

    def __call__(self) -> _Ctx:
        session = _RecordingSession(self._tables, self.timeline)
        self.sessions.append(session)
        return _Ctx(session)


async def test_stream_emits_tokens_then_done_after_snapshot():
    # 方案 3.7 时序：逐 token → 写 messages 快照 → done
    timeline: list[str] = []
    graph = FakeGraph(reply="你好", tokens=["你", "好"])
    service = ChatService(
        graph=graph, session_factory=_RecordingFactory(_tables(), timeline), memory_store=_STORE
    )

    context = await service.load_context(session_id=1)
    events = []
    async for name, data in service.stream(context=context, message="在吗"):
        events.append((name, data))
        timeline.append(f"event:{name}")

    assert [name for name, _ in events] == ["token", "token", "done"]
    assert events[0][1] == {"text": "你"}
    assert events[-1][1]["reply"] == "你好"
    # done 必须在快照落库之后
    assert timeline == ["event:token", "event:token", "snapshot", "event:done"]


class _MultiGraph(FakeGraph):
    """流式产出 2 条消息（中间一个 break），最终 values 带 replies（M5）。"""

    def __init__(self) -> None:
        super().__init__(reply="你好\n在的")

    async def astream(self, state, config, stream_mode=None):
        yield ("custom", {"text": "你好"})
        yield ("custom", {"break": True})
        yield ("custom", {"text": "在的"})
        yield ("values", {"reply": "你好\n在的", "replies": ["你好", "在的"]})


async def test_stream_multiple_messages_snapshot_and_done():
    # M5：角色连发多条 → 一个 user + 多条 assistant 快照；done 带各条 id 与内容
    factory = FakeSessionFactory(_tables())
    service = ChatService(
        graph=_MultiGraph(), session_factory=factory, memory_store=_STORE
    )
    context = await service.load_context(session_id=1)
    events = [e async for e in service.stream(context=context, message="在吗")]

    assert [n for n, _ in events] == ["token", "message_break", "token", "done"]
    payload = events[-1][1]
    assert [r["content"] for r in payload["replies"]] == ["你好", "在的"]
    assert all(r["message_id"] for r in payload["replies"])
    assert [m.role for m in factory.sessions[-1].added] == ["user", "assistant", "assistant"]


async def test_stream_uses_session_agent_in_thread_id():
    # 接缝 1：流式路径下 thread_id 仍含 agent_id
    captured: dict = {}

    class _CapturingGraph(FakeGraph):
        async def astream(self, state, config, stream_mode=None):
            captured["thread_id"] = config["configurable"]["thread_id"]
            captured["state_agent_id"] = state["agent_id"]
            async for item in super().astream(state, config, stream_mode):
                yield item

    service = ChatService(
        graph=_CapturingGraph(reply="ok"),
        session_factory=FakeSessionFactory(_tables(agent_id=7)),
        memory_store=_STORE,
    )
    context = await service.load_context(session_id=1)
    async for _ in service.stream(context=context, message="hi"):
        pass

    assert captured["thread_id"] == "1:7"
    assert captured["state_agent_id"] == 7


async def test_stream_propagates_upstream_error():
    service = ChatService(
        graph=FakeGraph(exc=UpstreamError("模型服务调用失败")),
        session_factory=FakeSessionFactory(_tables()),
        memory_store=_STORE,
    )
    context = await service.load_context(session_id=1)
    with pytest.raises(UpstreamError):
        async for _ in service.stream(context=context, message="hi"):
            pass


class _CapturingIdleGraph(FakeGraph):
    """流式产出 2 条消息并记录收到的图状态（用于验证闲聊轮的注入）。"""

    def __init__(self) -> None:
        super().__init__(reply="你好\n在的")
        self.captured: dict = {}

    async def astream(self, state, config, stream_mode=None):
        self.captured["state"] = state
        yield ("custom", {"text": "你好"})
        yield ("custom", {"break": True})
        yield ("custom", {"text": "在的"})
        yield ("values", {"reply": "你好\n在的", "replies": ["你好", "在的"]})


async def test_idle_stream_writes_idle_snapshot_without_user():
    # 自主闲聊：无用户消息、快照只有 assistant 行且标记 is_idle，done 带 idle=True
    factory = FakeSessionFactory(_tables())
    graph = _CapturingIdleGraph()
    service = ChatService(graph=graph, session_factory=factory, memory_store=_STORE)
    context = await service.load_context(session_id=1)

    events = [e async for e in service.idle_stream(context=context)]

    assert [n for n, _ in events] == ["token", "message_break", "token", "done"]
    payload = events[-1][1]
    assert payload["idle"] is True
    assert payload["user_message_id"] == 0
    assert [r["content"] for r in payload["replies"]] == ["你好", "在的"]
    assert all(r["message_id"] for r in payload["replies"])

    rows = factory.sessions[-1].added
    assert [m.role for m in rows] == ["assistant", "assistant"]
    assert all(m.is_idle for m in rows)

    # 图状态：不注入用户消息，仅置 idle_scope=private（不走正常轮）
    state = graph.captured["state"]
    assert state["messages"] == []
    assert state["idle_scope"] == "private"


async def test_normal_stream_state_resets_idle_scope():
    # 接缝：正常轮必须显式清空 idle_scope，避免上一次闲聊残留污染（状态被 checkpointer 持久化）
    graph = _CapturingIdleGraph()
    service = ChatService(
        graph=graph, session_factory=FakeSessionFactory(_tables()), memory_store=_STORE
    )
    context = await service.load_context(session_id=1)
    async for _ in service.stream(context=context, message="hi"):
        pass

    state = graph.captured["state"]
    assert state["idle_scope"] == ""
    assert len(state["messages"]) == 1  # 本轮用户消息


# ===== M6：思考链落库与 done 载荷 =====


class _ThinkingGraph(FakeGraph):
    """流式 2 条消息，并在 values 快照里带思考链。"""

    def __init__(self, replies: list[str], thinkings: list[str]) -> None:
        super().__init__(reply="\n".join(replies))
        self._replies = replies
        self._thinkings = thinkings

    async def astream(self, state, config, stream_mode=None):
        yield ("custom", {"text": self._replies[0]})
        yield ("custom", {"break": True})
        yield ("custom", {"text": self._replies[1]})
        yield (
            "values",
            {
                "reply": "\n".join(self._replies),
                "replies": list(self._replies),
                "thinkings": list(self._thinkings),
            },
        )


async def test_stream_persists_thinking_and_exposes_it_in_done():
    factory = FakeSessionFactory(_tables())
    service = ChatService(
        graph=_ThinkingGraph(["你好", "在的"], ["有点害羞", "其实很开心"]),
        session_factory=factory,
        memory_store=_STORE,
    )
    context = await service.load_context(session_id=1)
    events = [e async for e in service.stream(context=context, message="在吗")]

    assert [r["thinking"] for r in events[-1][1]["replies"]] == ["有点害羞", "其实很开心"]
    rows = factory.sessions[-1].added
    assert rows[0].role == "user" and rows[0].thinking is None  # 用户行无思考
    assistants = [m for m in rows if m.role == "assistant"]
    assert [m.thinking for m in assistants] == ["有点害羞", "其实很开心"]


async def test_stream_aligns_short_thinking_list_to_replies():
    # 思考不足时补齐为空（前端据此不显示按钮）
    factory = FakeSessionFactory(_tables())
    service = ChatService(
        graph=_ThinkingGraph(["甲", "乙"], ["仅一行"]),
        session_factory=factory,
        memory_store=_STORE,
    )
    context = await service.load_context(session_id=1)
    events = [e async for e in service.stream(context=context, message="hi")]

    assert [r["thinking"] for r in events[-1][1]["replies"]] == ["仅一行", None]


async def test_idle_stream_persists_thinking_on_idle_rows():
    factory = FakeSessionFactory(_tables())
    service = ChatService(
        graph=_ThinkingGraph(["自言自语", "又一句"], ["想起从前", "心情不错"]),
        session_factory=factory,
        memory_store=_STORE,
    )
    context = await service.load_context(session_id=1)
    events = [e async for e in service.idle_stream(context=context)]

    assert events[-1][1]["idle"] is True
    rows = factory.sessions[-1].added
    assert all(m.is_idle for m in rows)
    assert [m.thinking for m in rows] == ["想起从前", "心情不错"]


class _ThinkingInvokeGraph(FakeGraph):
    """非流式路径：ainvoke 直接返回 replies + thinkings。"""

    async def ainvoke(self, state, config):
        return {
            "reply": "你好\n在的",
            "replies": ["你好", "在的"],
            "thinkings": ["想你好", "想在的"],
        }


async def test_chat_non_stream_persists_thinking():
    factory = FakeSessionFactory(_tables())
    service = ChatService(
        graph=_ThinkingInvokeGraph(), session_factory=factory, memory_store=_STORE
    )
    await service.chat(session_id=1, message="在吗")

    assistants = [m for m in factory.sessions[-1].added if m.role == "assistant"]
    assert [m.thinking for m in assistants] == ["想你好", "想在的"]


# ===== 硬化：流式请求中断/异常时回滚工作记忆里的孤儿用户消息 =====

_DIMS = 1024


class _FixedEmbeddings(Embeddings):
    """固定向量，供 InMemoryStore 的向量索引使用（无需真模型）。"""

    def _vec(self) -> list[float]:
        return [0.1] * _DIMS

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec() for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec()


class _SuccessLLM:
    def __init__(self, reply: str = "你好呀，客人。") -> None:
        self.reply = reply

    async def generate(self, messages: list[dict]) -> str:
        return self.reply

    async def astream(self, messages: list[dict]):
        yield self.reply


class _FailingLLM:
    """流式生成即抛错，模拟请求中途失败（图此前已起步并写入工作记忆）。"""

    async def generate(self, messages: list[dict]) -> str:  # pragma: no cover - 仅走 astream
        raise UpstreamError("模型服务调用失败")

    async def astream(self, messages: list[dict]):
        raise UpstreamError("模型服务调用失败")
        yield  # pragma: no cover - 使函数成为异步生成器


def _real_graph(llm):
    """真图 + 内存 checkpointer/store：让工作记忆真实落盘，验证回滚确有效果。"""
    store = InMemoryStore(index={"dims": _DIMS, "embed": _FixedEmbeddings(), "fields": ["text"]})
    return build_graph(
        llm=llm, memory_store=MemoryStore(store), checkpointer=InMemorySaver(), store=store
    )


async def _thread_contents(graph, thread_id: str) -> list[str]:
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return [str(m.content) for m in (snapshot.values.get("messages") or [])]


async def test_stream_failure_rolls_back_working_memory_user_message():
    # 加固：请求中途失败 → 图已写入工作记忆的本轮用户消息必须被回滚，不留孤儿消息
    graph = _real_graph(_FailingLLM())
    factory = FakeSessionFactory(_tables())
    service = ChatService(graph=graph, session_factory=factory, memory_store=_STORE)
    context = await service.load_context(session_id=1)

    with pytest.raises(UpstreamError):
        async for _ in service.stream(context=context, message="幽灵消息"):
            pass

    assert "幽灵消息" not in await _thread_contents(graph, "1:1")
    assert all(s.added == [] for s in factory.sessions)  # 失败路径不写展示快照


async def test_stream_success_keeps_working_memory_user_message():
    # 正常完成不得误回滚：用户消息与角色回复都应留在工作记忆里
    graph = _real_graph(_SuccessLLM("你好呀，客人。"))
    service = ChatService(
        graph=graph, session_factory=FakeSessionFactory(_tables()), memory_store=_STORE
    )
    context = await service.load_context(session_id=1)

    events = [e async for e in service.stream(context=context, message="在吗")]
    assert [n for n, _ in events][-1] == "done"

    contents = await _thread_contents(graph, "1:1")
    assert "在吗" in contents
    assert "你好呀，客人。" in contents
