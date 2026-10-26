"""V2.4 返修项回归（R1/R2/R3/R4/R5/R6/R7）。

R1 的 20×20 门槛由脚本 + 重跑验证（见 §5）；这里测代码语义：
`PooledAsyncPostgresSaver` 在池化连接下不得再串行化 checkpoint I/O。
R8/R9（证据冻结）同样由脚本与 manifest 验证。
"""

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore

from app.agents.supervisor import GroupMember, Supervisor
from app.config import get_settings
from app.core.locks import KeyedLockPool
from app.db.checkpointer import PooledAsyncPostgresSaver
from app.llm.fake import FakeLLMProvider
from app.lore.indexer import source_ref
from app.memory.store import MemoryStore
from app.observability.llm import InstrumentedLLM
from app.observability.trace import (
    TZ,
    TraceRecorder,
    TraceStore,
    configure_trace_store,
    reset_trace,
    start_trace,
)
from app.services import chat_service as chat_service_module
from app.services.chat_service import ChatContext, ChatService

# trace 文件名按 Asia/Shanghai 日期切分，这里必须同源取日期，
# 否则 UTC 宿主（CI）的 16:00–24:00 窗口会去找“昨天”的文件。
TODAY = datetime.now(TZ).date().isoformat()


def _records(directory: Path) -> list[dict]:
    path = directory / f"{TODAY}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ===== R2：单测不得写默认 trace 目录 =====


def test_pytest_never_writes_default_trace_dir():
    path = Path(get_settings().trace_dir) / f"{TODAY}.jsonl"
    before = path.stat().st_size if path.exists() else None

    recorder, token = start_trace(mode="private", session_id=1, agent_id=1, turn=1)
    try:
        recorder.node("retrieve_memory", 1)
        recorder.finish()
    finally:
        reset_trace(token)

    after = path.stat().st_size if path.exists() else None
    # autouse fixture 注入 None → store 为空 → recorder 为 no-op，默认目录零写入
    assert recorder._noop is True
    assert after == before


# ===== R3：fake 模式不得标成真实模型/成本 =====


async def test_fake_trace_writes_model_fake_and_zero_cost(tmp_path):
    configure_trace_store(TraceStore(directory=str(tmp_path)))
    try:
        fake = FakeLLMProvider(delay_ms=0)
        llm = InstrumentedLLM(
            fake,
            model="fake",
            input_price_per_1m=0.0,
            output_price_per_1m=0.0,
            usage_source="fake",
        )
        recorder, token = start_trace(mode="private", session_id=1, agent_id=1, turn=1)
        try:
            await llm.generate([{"role": "user", "content": "你好"}])
            recorder.finish()
        finally:
            reset_trace(token)
    finally:
        configure_trace_store(None)

    record = _records(tmp_path)[0]
    assert record["llm"]["model"] == "fake"
    assert record["llm"]["cost_cny"] == 0.0
    assert record["llm"]["usage_source"] == "fake"
    assert record["llm"]["calls"] == 1


# ===== R4：TRACE_MEMORY_STATS 真正实现 =====


async def test_memory_store_stats_counts_namespace():
    store = MemoryStore(InMemoryStore())
    await store.put(1, "用户喜欢乌龙茶")
    await store.put_fact(1, subject="用户", predicate="喜欢", obj="乌龙茶")
    await store.put_profile(1, "称呼", "阿岚")

    assert await store.stats(1) == {
        "namespace_entries": 4,  # 1 memory + fact + fact_cur + profile
        "facts": 1,
        "profile_slots": 1,
    }


def test_memory_stats_default_off(tmp_path):
    assert get_settings().trace_memory_stats is False

    recorder = TraceRecorder(trace_id="t", git_sha="x", _store=None, _noop=False)
    recorder.memory_snapshot(namespace_entries=7, facts=2, profile_slots=1)
    assert recorder.memory_stats == {"namespace_entries": 7, "facts": 2, "profile_slots": 1}


# ===== R5：群聊所有决策调用都有归属 =====


class _StubGraph:
    """不调用 LLM 的角色图：只为 supervisor 提供一个可消费的 astream。"""

    async def astream(self, state, config=None, stream_mode=None):
        yield "values", {"replies": ["好的"]}


async def test_group_decision_calls_all_attributed(tmp_path):
    configure_trace_store(TraceStore(directory=str(tmp_path)))
    try:
        fake = FakeLLMProvider(delay_ms=0)
        llm = InstrumentedLLM(
            fake, model="fake", input_price_per_1m=0.0, output_price_per_1m=0.0,
            usage_source="fake",
        )
        members = [
            GroupMember(agent_id=i, name=f"角色{i}", persona_prompt="", background="")
            for i in (1, 2, 3)
        ]
        supervisor = Supervisor(
            llm=llm,
            graph_factory=lambda: _StubGraph(),
            members=members,
            max_turns=3,
        )
        async for _ in supervisor.run(group_session_id=9, user_message="聊聊", turn=1):
            pass
    finally:
        configure_trace_store(None)

    records = _records(tmp_path)
    assert len(records) == 3  # 三个发言人各一条 trace
    # 3 次选人决策 + 1 次终止决策（全部计入，含终止 END）
    assert fake.calls == 4
    assert sum(int(r["llm"]["calls"]) for r in records) == fake.calls


# ===== R6：turn 在会话锁内计算 =====


class _TrackedPool:
    """包一层 KeyedLockPool，暴露“当前是否在锁内”。"""

    def __init__(self) -> None:
        self.inner = KeyedLockPool()
        self.inside = False

    @asynccontextmanager
    async def acquire(self, key: str):
        async with self.inner.acquire(key):
            self.inside = True
            try:
                yield
            finally:
                self.inside = False


async def test_turn_computed_inside_session_lock(monkeypatch):
    service = object.__new__(ChatService)
    pool = _TrackedPool()
    service._locks = pool
    service._reset_tools = lambda: None
    service._idempotent_lookup = lambda *a, **k: None
    service._idempotent_store = lambda *a, **k: None

    context = ChatContext(
        session_id=7, agent_id=1, agent_name="A", persona_prompt="", background=""
    )

    async def load_context(*, session_id: int) -> ChatContext:
        return context

    service.load_context = load_context

    turns: list[int] = []
    completed = {"n": 0}

    async def next_turn(session_id: int) -> int:
        assert pool.inside is True, "turn 必须在本会话锁内计算"
        await asyncio.sleep(0.01)  # 让并发请求有机会插队（无锁时两者会取到同一 turn）
        turns.append(completed["n"] + 1)
        return completed["n"] + 1

    async def write_snapshot(*args, **kwargs):
        return [], []

    class _Graph:
        async def ainvoke(self, state, config=None):
            await asyncio.sleep(0.01)
            completed["n"] += 1
            return {"replies": ["ok"]}

    service._next_turn = next_turn
    service._write_snapshot = write_snapshot
    service._graph = _Graph()

    real_start_trace = chat_service_module.start_trace
    seen: list[int] = []

    def spy(**kwargs):
        seen.append(kwargs["turn"])
        return real_start_trace(**kwargs)

    monkeypatch.setattr(chat_service_module, "start_trace", spy)

    await asyncio.gather(
        service.chat(session_id=7, message="第一句"),
        service.chat(session_id=7, message="第二句"),
    )
    assert sorted(seen) == [1, 2]


# ===== R7：lore_ids 的 source_ref 口径 =====


def test_lore_source_ref_is_slug_seq():
    ref = source_ref("yunqi_tea", 10)
    assert ref == "yunqi_tea:10"
    assert re.fullmatch(r"[a-z0-9_]+:\d{2,}", ref)

    recorder = TraceRecorder(trace_id="t", git_sha="x", _store=None, _noop=False)
    recorder.retrieval(
        lore_ids=[ref], memory_count=0, rerank_top=None, rewrite_used=False, rewrite_ms=0
    )
    assert recorder.retrieval_stats["lore_ids"] == ["yunqi_tea:10"]


@pytest.mark.parametrize("bad", ["yunqi_tea", "yunqi_tea:", ":10", "yunqi tea:10"])
def test_lore_source_ref_negative_control(bad):
    assert re.fullmatch(r"[a-z0-9_]+:\d{2,}", bad) is None


# ===== R1：池化 saver 不得串行化 checkpoint I/O =====


async def test_pooled_saver_does_not_serialize_cursor(real_pool):
    """两个并发 `_cursor` 必须重叠执行（基类实例级全局锁会把它们排成 0.3s+0.3s）。"""
    saver = PooledAsyncPostgresSaver(real_pool)
    await saver.setup()

    async def hold() -> None:
        # `_cursor` 是 saver 的私有接缝；本类存在的目的就是改它的锁语义，故直接压测
        async with saver._cursor() as cur:
            await cur.execute("select pg_sleep(0.3)")

    started = time.perf_counter()
    await asyncio.gather(hold(), hold())
    elapsed = time.perf_counter() - started
    assert elapsed < 0.55, f"并发未生效（{elapsed:.2f}s ≈ 串行 0.6s）"


def test_pooled_saver_rejects_single_connection():
    with pytest.raises(TypeError):
        PooledAsyncPostgresSaver("postgresql://x")  # type: ignore[arg-type]


# ===== R13：群聊 trace 生命周期（turn_done 前冻结/断开与异常兜底）=====


class _FlakyDecisionLLM(FakeLLMProvider):
    """可控决策模型：返回固定编号；可指定第 N 次决策抛异常或延迟。"""

    def __init__(self, *, fail_on: int | None = None, delay_on: dict[int, int] | None = None):
        super().__init__(delay_ms=0)
        self._fail_on = fail_on
        self._delay_on = delay_on or {}

    async def generate(self, messages):
        self.calls += 1
        delay_ms = int(self._delay_on.get(self.calls, 0))
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000)
        if self._fail_on == self.calls:
            raise RuntimeError("decision boom")
        return "1" if self.calls < 3 else "END"


def _supervisor(llm: InstrumentedLLM, *, max_turns: int = 3) -> Supervisor:
    members = [
        GroupMember(agent_id=i, name=f"角色{i}", persona_prompt="", background="")
        for i in (1, 2, 3)
    ]
    return Supervisor(
        llm=llm, graph_factory=lambda: _StubGraph(), members=members, max_turns=max_turns
    )


def _instrumented(fake: FakeLLMProvider) -> InstrumentedLLM:
    return InstrumentedLLM(
        fake, model="fake", input_price_per_1m=0.0, output_price_per_1m=0.0,
        usage_source="fake",
    )


async def test_group_trace_kept_when_client_closes_after_turn_done(tmp_path):
    """R13：客户端在 turn_done 后断开（aclose）也必须留下该发言人的 trace。"""
    configure_trace_store(TraceStore(directory=str(tmp_path)))
    try:
        supervisor = _supervisor(_instrumented(_FlakyDecisionLLM()), max_turns=1)
        agen = supervisor.run(group_session_id=9, user_message="聊聊", turn=1)
        try:
            async for event, _ in agen:
                if event == "turn_done":
                    break
        finally:
            await agen.aclose()
    finally:
        configure_trace_store(None)

    records = _records(tmp_path)
    assert len(records) == 1
    assert records[0]["mode"] == "group"
    assert records[0]["llm"]["calls"] == 1  # 选人决策已归属


async def test_group_trace_kept_when_next_decision_raises(tmp_path):
    """R13：下一次 `_decide` 抛异常时，已完成的发言人 trace 不得丢失。"""
    configure_trace_store(TraceStore(directory=str(tmp_path)))
    try:
        llm = _instrumented(_FlakyDecisionLLM(fail_on=2))
        supervisor = _supervisor(llm)
        with pytest.raises(RuntimeError, match="decision boom"):
            async for _ in supervisor.run(group_session_id=9, user_message="聊聊", turn=1):
                pass
    finally:
        configure_trace_store(None)

    records = _records(tmp_path)
    assert len(records) == 1
    assert records[0]["agent_id"] == 1
    assert records[0]["error"] is None  # 本发言人的回合本身成功


async def test_group_trace_total_ms_excludes_next_decision(tmp_path):
    """R13：total_ms 必须在 turn_done 前冻结，不含下一次决策耗时。"""
    configure_trace_store(TraceStore(directory=str(tmp_path)))
    try:
        # 第 2 次决策（上一位 turn_done 之后）故意慢 200ms
        llm = _instrumented(_FlakyDecisionLLM(delay_on={2: 200}))
        supervisor = _supervisor(llm)
        async for _ in supervisor.run(group_session_id=9, user_message="聊聊", turn=1):
            pass
    finally:
        configure_trace_store(None)

    first = next(r for r in _records(tmp_path) if r["agent_id"] == 1)
    assert first["total_ms"] < 100, f"total_ms={first['total_ms']} 含了下一次决策"

