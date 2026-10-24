"""T2：InstrumentedLLM 超时兜底 / 信号量 / 降级轮不写记忆。"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.agents.nodes.generate import generate
from app.agents.nodes.update_memory import update_memory
from app.llm.base import LLMProvider, LLMReply
from app.observability.llm import InstrumentedLLM
from app.observability.trace import (
    TraceStore,
    configure_trace_store,
    current_error,
    is_degraded,
    reset_trace,
    start_trace,
)

FALLBACK = "（兜底测试文案）"


class SlowProvider(LLMProvider):
    async def generate(self, messages):
        await asyncio.sleep(1)
        return "真实回复"

    async def astream(self, messages):
        await asyncio.sleep(1)
        yield "真实"

    async def generate_with_tools(
        self, messages, tools, *, tool_choice="auto", temperature=0.0
    ):
        await asyncio.sleep(1)
        return LLMReply(content="真实工具回复")


class SlowTailProvider(LLMProvider):
    async def generate(self, messages):
        return "ok"

    async def astream(self, messages):
        yield "a"
        await asyncio.sleep(0.05)
        yield "b"


class TrackingProvider(LLMProvider):
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def generate(self, messages):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return "ok"

    async def astream(self, messages):
        yield "ok"


@pytest.fixture
def store(tmp_path):
    store = TraceStore(directory=str(tmp_path / "traces"), enabled=True, sample_rate=1.0)
    configure_trace_store(store)
    yield store
    configure_trace_store(None)


@pytest.mark.asyncio
async def test_generate_timeout_returns_fallback_and_marks_degraded(store, tmp_path):
    llm = InstrumentedLLM(SlowProvider(), model="m", timeout_s=0.01, fallback=FALLBACK)
    recorder, token = start_trace(mode="private", session_id=1, agent_id=1, turn=1)
    try:
        text = await llm.generate([{"role": "user", "content": "你好"}])
        assert text == FALLBACK
        assert is_degraded() is True
        assert current_error()["type"] == "timeout"
        recorder.finish(error=current_error(), degraded=is_degraded())
    finally:
        reset_trace(token)

    record = json.loads(
        (tmp_path / "traces" / f"{recorder.started_at.date().isoformat()}.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert record["degraded"] is True
    assert record["error"]["type"] == "timeout"
    assert record["llm"]["calls"] == 1
    assert record["llm"]["prompt_tokens"] > 0
    assert record["llm"]["completion_tokens"] > 0
    assert record["llm"]["usage_source"] == "estimator"


@pytest.mark.asyncio
async def test_astream_first_token_timeout_yields_only_fallback():
    llm = InstrumentedLLM(SlowProvider(), model="m", timeout_s=0.01, fallback=FALLBACK)
    chunks = [chunk async for chunk in llm.astream([{"role": "user", "content": "你好"}])]
    assert chunks == [FALLBACK]
    assert is_degraded() is True


@pytest.mark.asyncio
async def test_astream_not_killed_after_first_token():
    llm = InstrumentedLLM(SlowTailProvider(), model="m", timeout_s=0.01, fallback=FALLBACK)
    chunks = [chunk async for chunk in llm.astream([{"role": "user", "content": "你好"}])]
    assert chunks == ["a", "b"]
    assert is_degraded() is False


@pytest.mark.asyncio
async def test_generate_with_tools_timeout_returns_fallback_reply():
    llm = InstrumentedLLM(SlowProvider(), model="m", timeout_s=0.01, fallback=FALLBACK)
    reply = await llm.generate_with_tools(
        [{"role": "user", "content": "你好"}], [], temperature=0.0
    )
    assert reply.content == FALLBACK
    assert reply.tool_calls == []
    assert is_degraded() is True


@pytest.mark.asyncio
async def test_global_semaphore_limits_concurrency():
    provider = TrackingProvider()
    llm = InstrumentedLLM(provider, model="m", max_concurrency=1, timeout_s=5)
    results = await asyncio.gather(
        *(llm.generate([{"role": "user", "content": str(i)}]) for i in range(3))
    )
    assert results == ["ok", "ok", "ok"]
    assert provider.peak == 1


@pytest.mark.asyncio
async def test_generate_node_sets_degraded_flag():
    llm = InstrumentedLLM(SlowProvider(), model="m", timeout_s=0.01, fallback=FALLBACK)
    state = {
        "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "messages": [],
        "agent_id": 1,
        "session_id": 1,
    }
    out = await generate(state, llm, max_messages=3)
    assert out["degraded"] is True
    assert FALLBACK in out["replies"]


@pytest.mark.asyncio
async def test_update_memory_skips_all_writes_when_degraded():
    class ExplodingStore:
        def __getattr__(self, name):  # 任何写入/读取都视为失败
            raise AssertionError(f"降级轮不得访问 Store：{name}")

    state = {
        "agent_id": 1,
        "session_id": 1,
        "messages": [],
        "degraded": True,
    }
    out = await update_memory(
        state, ExplodingStore(), SlowProvider(), extract_enabled=True
    )
    assert out == {}
