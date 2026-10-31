"""I-01 回归：滚动摘要整条链路 fail-open（0 真实 LLM，InMemoryStore + fake LLM）。

背景（2026-10-27，QQ 桥连跑 6 天）：`/memory/turn` 连续 7 小时、41 轮 502。
根因是摘要失败后缓冲卡在阈值之上，每轮都在同一处重复失败；工作区的 fail-open
是临时修补，本文件把它钉成三条断言：

1. 摘要 LLM 抛异常 → `update_memory` 不抛、返回 `{}`、消息不被裁剪；
2. 同一批输入再来一轮、LLM 恢复正常 → 能成功摘要（不会永久卡住）；
3. 该场景下 `/memory/turn` 返回 200，响应里能看出本轮没有摘要（沿用现有字段）。

断言无法在现有实现下通过时应当停下来报告，不得改实现迁就测试。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from app.memory.store import MemoryStore
from app.agents.nodes.update_memory import update_memory
from app.api.v1 import api_router
from app.config import Settings
from app.core.error_handlers import register_exception_handlers
from app.llm.base import ChatMessage, LLMProvider
from app.prompts.memory_extract import EXTRACT_SYSTEM_PROMPT
from app.services.memory_api_buffer import MemoryApiBuffer
from app.services.memory_api_service import MemoryApiService

AGENT = "ext"
SESSION = "main"
_TRIGGER = 10
_CHUNK = 10


class _FlakyLLM(LLMProvider):
    """摘要调用按 `fail` 开关抛错；恢复后返回固定摘要文本。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.summary_calls = 0

    async def generate(self, messages: list[ChatMessage]) -> str:
        system = str(messages[0]["content"]) if messages else ""
        if system == EXTRACT_SYSTEM_PROMPT:
            return json.dumps({"facts": [], "profile": {}}, ensure_ascii=False)
        self.summary_calls += 1
        if self.fail:
            raise RuntimeError("模拟摘要模型抖动")
        return "用户前十轮反复说：今天有点累想早点休息"

    async def astream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # pragma: no cover


def _messages(rounds: int) -> list:
    """带 id 的 messages：摘要裁剪才有 RemoveMessage 可产出。"""
    out: list = []
    for i in range(1, rounds + 1):
        out.append(HumanMessage(id=f"u{i}", content=f"第{i}轮：今天有点累想早点休息"))
        out.append(AIMessage(id=f"a{i}", content=f"回应{i}"))
    return out


def _state(messages: list, replies: list[str]) -> dict:
    return {
        "messages": messages,
        "agent_id": AGENT,
        "session_id": SESSION,
        "replies": replies,
        "idle_scope": "",
        "degraded": False,
    }


async def test_摘要LLM异常时update_memory不抛且缓冲不裁剪():
    store = MemoryStore(InMemoryStore())
    llm = _FlakyLLM(fail=True)
    messages = _messages(_TRIGGER + 1)

    result = await update_memory(
        _state(messages, ["回应11"]),
        store,
        llm,
        trigger_rounds=_TRIGGER,
        chunk_rounds=_CHUNK,
    )

    assert result == {}  # 不带 RemoveMessage：缓冲一条都不裁
    assert llm.summary_calls == 1
    assert await store.get_summary(AGENT, SESSION) == ""


async def test_摘要恢复后下一轮能成功摘要():
    store = MemoryStore(InMemoryStore())
    llm = _FlakyLLM(fail=True)
    messages = _messages(_TRIGGER + 1)

    assert await update_memory(
        _state(messages, ["回应11"]),
        store,
        llm,
        trigger_rounds=_TRIGGER,
        chunk_rounds=_CHUNK,
    ) == {}

    llm.fail = False  # 模型恢复
    result = await update_memory(
        _state(messages, ["回应11"]),
        store,
        llm,
        trigger_rounds=_TRIGGER,
        chunk_rounds=_CHUNK,
    )

    assert llm.summary_calls == 2
    assert await store.get_summary(AGENT, SESSION) == "用户前十轮反复说：今天有点累想早点休息"
    removed_ids = [getattr(m, "id", None) for m in result["messages"]]
    assert removed_ids == [
        f"{kind}{i}" for i in range(1, _CHUNK + 1) for kind in ("u", "a")
    ]


def _settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql://roleplay:roleplay@localhost:5432/unused",
        memory_api_agents=AGENT,
        memory_api_buffer_path=str(tmp_path / "buffer.json"),
        memory_api_buffer_max_messages=200,
        memory_extract_enabled=False,  # 只压摘要链路，不触发抽取
    )


def _client(service: MemoryApiService) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(api_router)
    register_exception_handlers(app)
    app.state.memory_api_service = service
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def test_摘要失败时memory_turn返回200且能看出未摘要(tmp_path):
    settings = _settings(tmp_path)
    store = MemoryStore(InMemoryStore())
    llm = _FlakyLLM(fail=True)
    buffer = MemoryApiBuffer(
        settings.memory_api_buffer_path,
        max_messages=settings.memory_api_buffer_max_messages,
    )
    service = MemoryApiService(memory_store=store, llm=llm, buffer=buffer, settings=settings)

    async with _client(service) as client:
        for i in range(1, _TRIGGER + 2):
            resp = await client.post(
                "/api/v1/memory/turn",
                json={
                    "agent": AGENT,
                    "session": SESSION,
                    "user_text": f"第{i}轮：今天有点累想早点休息",
                    "replies": ["早点休息。"],
                    "at": f"2026-10-25T10:{i:02d}:00+08:00",
                },
            )
            assert resp.status_code == 200, resp.text

        body = resp.json()
        # 现有契约字段就能看出降级：本轮没有摘要、缓冲没有被裁剪
        assert body["rounds"] == _TRIGGER + 1
        assert body["summarized"] is False
        assert body["buffer_size"] == (_TRIGGER + 1) * 2
        assert await store.get_summary(AGENT, SESSION) == ""

        llm.fail = False  # 下一轮恢复
        recovered = await client.post(
            "/api/v1/memory/turn",
            json={
                "agent": AGENT,
                "session": SESSION,
                "user_text": f"第{_TRIGGER + 2}轮：今天有点累想早点休息",
                "replies": ["早点休息。"],
                "at": f"2026-10-25T10:{_TRIGGER + 2:02d}:00+08:00",
            },
        )

    assert recovered.status_code == 200, recovered.text
    recovered_body = recovered.json()
    assert recovered_body["summarized"] is True
    assert recovered_body["buffer_size"] == 4  # 24 条 → 摘要最旧 10 轮 → 剩 2 轮
    assert await store.get_summary(AGENT, SESSION) != ""
