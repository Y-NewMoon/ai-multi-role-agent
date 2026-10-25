"""V3 M2：记忆接口契约测试（0 真实 LLM；InMemoryStore + fake LLM）。

覆盖交接 §8.2 的最低清单：单轮 / 11 轮摘要 / parity / 重启 / 幂等 / 重复 id /
并发 / 溢出 / forget / state 分页 / 错误信封 / 白名单 / boundary 不存在。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.embeddings import Embeddings
from langgraph.store.memory import InMemoryStore

from app.api.v1 import api_router
from app.agents.nodes.update_memory import update_memory
from app.config import Settings
from app.core.error_handlers import register_exception_handlers
from app.llm.base import ChatMessage, LLMProvider
from app.memory.store import MemoryStore
from app.prompts.l2_summary import L2_SYSTEM_PROMPT
from app.prompts.memory_extract import EXTRACT_SYSTEM_PROMPT
from app.services.memory_api_buffer import BufferPersistError, MemoryApiBuffer
from app.services.memory_api_service import MemoryApiError, MemoryApiService

_DIMS = 4
AGENT = "ext"
SESSION = "main"
USER_TEXT = "今天有点累，想早点休息"


class _Emb(Embeddings):
    def __init__(self, dims: int = _DIMS) -> None:
        self._dims = dims

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        raw = (digest * ((self._dims // len(digest)) + 1))[: self._dims]
        return [b / 255.0 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class _FakeLLM(LLMProvider):
    """确定性 fake：摘要文本由输入决定（parity 才有意义），抽取返回固定 JSON。"""

    def __init__(self, *, extraction: dict | None = None) -> None:
        self.calls = 0
        self._extraction = extraction or {
            "facts": [{"subject": "用户", "predicate": "状态", "object": "有点累", "importance": 4}],
            "profile": {"称呼": "阿棠"},
        }

    async def generate(self, messages: list[ChatMessage]) -> str:
        self.calls += 1
        system = str(messages[0]["content"]) if messages else ""
        if system == EXTRACT_SYSTEM_PROMPT:
            return json.dumps(self._extraction, ensure_ascii=False)
        if system == L2_SYSTEM_PROMPT:
            return "L2 关系档案"
        # 摘要：截取待压缩对话（内容决定文本）
        return "摘要：" + str(messages[-1]["content"])[-80:]

    async def astream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # pragma: no cover


class _IncrementFactLLM(_FakeLLM):
    """抽取每次返回不同 predicate：能抓出“空 replies 却仍抽取”的回归。"""

    def __init__(self) -> None:
        super().__init__()
        self.extract_calls = 0

    async def generate(self, messages: list[ChatMessage]) -> str:
        system = str(messages[0]["content"]) if messages else ""
        if system == EXTRACT_SYSTEM_PROMPT:
            self.extract_calls += 1
            return json.dumps(
                {
                    "facts": [
                        {
                            "subject": "用户",
                            "predicate": f"状态{self.extract_calls}",
                            "object": "有点累",
                            "importance": 4,
                        }
                    ],
                    "profile": {},
                },
                ensure_ascii=False,
            )
        return await super().generate(messages)


def _store() -> MemoryStore:
    return MemoryStore(
        InMemoryStore(index={"dims": _DIMS, "embed": _Emb(), "fields": ["text"]})
    )


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "database_url": "postgresql://roleplay:roleplay@localhost:5432/unused",
        "memory_api_agents": AGENT,
        "memory_api_buffer_path": str(tmp_path / "buffer.json"),
        "memory_api_buffer_max_messages": 200,
    }
    values.update(overrides)
    return Settings(**values)


def _service(store: MemoryStore, llm: LLMProvider, settings: Settings, buffer=None):
    buffer = buffer or MemoryApiBuffer(
        settings.memory_api_buffer_path,
        max_messages=settings.memory_api_buffer_max_messages,
    )
    return MemoryApiService(memory_store=store, llm=llm, buffer=buffer, settings=settings), buffer


def _app(service: MemoryApiService) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    register_exception_handlers(app)
    app.state.memory_api_service = service
    return app


def _client(service: MemoryApiService) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(service)), base_url="http://test"
    )


async def _turn(client: httpx.AsyncClient, **payload) -> httpx.Response:
    body = {"agent": AGENT, "session": SESSION, "user_text": USER_TEXT, "replies": ["那就早点休息。"]}
    body.update(payload)
    return await client.post("/api/v1/memory/turn", json=body)


# ---------------------------------------------------------------------------
# 单轮 / 11 轮 / parity
# ---------------------------------------------------------------------------
async def test_turn_single_round(tmp_path):
    service, buffer = _service(_store(), _FakeLLM(), _settings(tmp_path))
    async with _client(service) as client:
        resp = await _turn(client, at="2026-10-25T10:35:00+08:00")
    body = resp.json()
    assert resp.status_code == 200
    assert body["ok"] is True and body["contract"] == "1"
    assert body["rounds"] == 1
    assert body["buffer_size"] == 2
    assert body["idempotent"] is False
    assert buffer.turn_count(AGENT, SESSION) == 1
    assert (tmp_path / "buffer.json").exists()


async def test_empty_replies_skip_extraction(tmp_path):
    """R1（§2.2）：空 replies 只写用户记忆、跳过助手写入与抽取。"""
    settings = _settings(tmp_path)
    store = _store()
    llm = _IncrementFactLLM()
    service, buffer = _service(store, llm, settings)
    first = await service.turn(
        agent=AGENT,
        session=SESSION,
        user_text=USER_TEXT,
        replies=["那就早点休息。"],
        at="2026-10-25T10:35:00+08:00",
        message_id="m-1",
        idempotency_key=None,
    )
    second = await service.turn(
        agent=AGENT,
        session=SESSION,
        user_text="再补一句：今晚不熬夜了真的",
        replies=[],
        at="2026-10-25T10:36:00+08:00",
        message_id="m-2",
        idempotency_key=None,
    )
    assert first["facts_added"] == 1
    assert llm.extract_calls == 1
    assert second["facts_added"] == 0
    assert llm.extract_calls == 1  # 空 replies 不再触发抽取
    assert second["buffer_size"] == 3  # 只追加用户消息，无助手消息
    assert [m["role"] for m in buffer.messages(AGENT, SESSION)] == [
        "user",
        "assistant",
        "user",
    ]
    assert (await service.state(agent=AGENT, session=SESSION))["facts"] == 1


async def test_eleven_rounds_summarized(tmp_path):
    settings = _settings(tmp_path, memory_extract_enabled=False)
    store = _store()
    service, buffer = _service(store, _FakeLLM(), settings)
    async with _client(service) as client:
        last = None
        for i in range(1, 12):
            last = await _turn(
                client,
                user_text=f"第{i}轮：今天有点累想早点休息",
                replies=["早点休息。"],
                at=f"2026-10-25T10:{i:02d}:00+08:00",
            )
            assert last.status_code == 200
        state = (await client.get("/api/v1/memory/state", params={"agent": AGENT, "session": SESSION})).json()

    body = last.json()
    assert body["rounds"] == 11
    assert body["summarized"] is True
    assert body["buffer_size"] == 2  # 22 条 → 摘要最旧 10 轮 → 剩本轮 2 条
    assert buffer.turn_count(AGENT, SESSION) == 11
    assert await store.get_summary(AGENT, SESSION) != ""
    assert state["buffer_size"] == 2
    assert state["last_turn_at"] == "2026-10-25T10:11:00+08:00"


async def test_summary_parity_with_node(tmp_path):
    settings = _settings(tmp_path, memory_extract_enabled=False)

    # service 路径
    service_store = _store()
    service, buffer = _service(service_store, _FakeLLM(), settings)
    for i in range(1, 12):
        await service.turn(
            agent=AGENT,
            session=SESSION,
            user_text=f"第{i}轮：今天有点累想早点休息",
            replies=["早点休息。"],
            at=f"2026-10-25T10:{i:02d}:00+08:00",
            message_id=None,
            idempotency_key=None,
        )
    service_summary = await service_store.get_summary(AGENT, SESSION)

    # 直接调 update_memory（同一组消息）
    node_store = _store()
    node_messages = MemoryApiService._to_langchain(buffer.messages(AGENT, SESSION))
    # 补回被摘要删掉的前 10 轮（与 service 首轮起完全一致）
    all_messages: list = []
    for i in range(1, 11):
        all_messages.extend(
            MemoryApiService._to_langchain(
                [
                    {
                        "id": f"m-{i}",
                        "role": "user",
                        "content": f"第{i}轮：今天有点累想早点休息",
                        "at": f"2026-10-25T10:{i:02d}:00+08:00",
                    },
                    {"id": f"m-{i}:a0", "role": "assistant", "content": "早点休息。"},
                ]
            )
        )
    all_messages.extend(node_messages)
    result = await update_memory(
        {
            "agent_id": AGENT,
            "session_id": SESSION,
            "messages": all_messages,
            "replies": ["早点休息。"],
            "idle_scope": "",
            "degraded": False,
        },
        node_store,
        _FakeLLM(),
        trigger_rounds=settings.summary_trigger_rounds,
        chunk_rounds=settings.summary_chunk_rounds,
        extract_enabled=settings.memory_extract_enabled,
    )
    assert result["messages"]
    node_summary = await node_store.get_summary(AGENT, SESSION)
    assert node_summary != ""
    assert service_summary == node_summary


# ---------------------------------------------------------------------------
# 重启 / 幂等 / 重复 id
# ---------------------------------------------------------------------------
async def test_restart_restores_buffer(tmp_path):
    settings = _settings(tmp_path)
    store = _store()
    service, buffer = _service(store, _FakeLLM(), settings)
    for i in range(1, 4):
        await service.turn(
            agent=AGENT,
            session=SESSION,
            user_text=f"第{i}轮：今天有点累想早点休息",
            replies=["早点休息。"],
            at=f"2026-10-25T10:0{i}:00+08:00",
            message_id=None,
            idempotency_key=None,
        )
    assert buffer.turn_count(AGENT, SESSION) == 3

    # 新实例读同一文件
    restarted_buffer = MemoryApiBuffer(
        settings.memory_api_buffer_path, max_messages=settings.memory_api_buffer_max_messages
    )
    await restarted_buffer.load()
    assert restarted_buffer.turn_count(AGENT, SESSION) == 3
    assert restarted_buffer.size(AGENT, SESSION) == 6
    assert [m["content"] for m in restarted_buffer.messages(AGENT, SESSION)][:1] == [
        "第1轮：今天有点累想早点休息"
    ]

    restarted, _ = _service(store, _FakeLLM(), settings, buffer=restarted_buffer)
    payload = await restarted.turn(
        agent=AGENT,
        session=SESSION,
        user_text="第4轮：今天有点累想早点休息",
        replies=["早点休息。"],
        at="2026-10-25T10:04:00+08:00",
        message_id=None,
        idempotency_key=None,
    )
    assert payload["rounds"] == 4
    assert payload["buffer_size"] == 8


async def test_idempotent_turn(tmp_path):
    settings = _settings(tmp_path)
    store = _store()
    service, buffer = _service(store, _FakeLLM(), settings)
    first = await service.turn(
        agent=AGENT,
        session=SESSION,
        user_text=USER_TEXT,
        replies=["那就早点休息。"],
        at="2026-10-25T10:35:00+08:00",
        message_id="m-1",
        idempotency_key="turn-1",
    )
    entries_after_first = await store.count_entries(AGENT)
    second = await service.turn(
        agent=AGENT,
        session=SESSION,
        user_text=USER_TEXT,
        replies=["那就早点休息。"],
        at="2026-10-25T10:35:00+08:00",
        message_id="m-1",
        idempotency_key="turn-1",
    )
    assert first["rounds"] == 1 and first["idempotent"] is False
    assert second["idempotent"] is True
    assert second["rounds"] == 1
    assert second["buffer_size"] == 2
    assert buffer.size(AGENT, SESSION) == 2
    assert await store.count_entries(AGENT) == entries_after_first


async def test_duplicate_message_id(tmp_path):
    service, buffer = _service(_store(), _FakeLLM(), _settings(tmp_path))
    async with _client(service) as client:
        assert (await _turn(client, message_id="m-1")).status_code == 200
        resp = await _turn(client, message_id="m-1", user_text="另一句完全不同的话")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "duplicate_message_id"
    assert buffer.size(AGENT, SESSION) == 2  # 未被追加


# ---------------------------------------------------------------------------
# 并发 / 溢出
# ---------------------------------------------------------------------------
async def test_concurrent_same_session(tmp_path):
    settings = _settings(tmp_path)
    service, buffer = _service(_store(), _FakeLLM(), settings)

    async def run(i: int):
        return await service.turn(
            agent=AGENT,
            session=SESSION,
            user_text=f"并发第{i}条：今天有点累想早点休息",
            replies=["早点休息。"],
            at=f"2026-10-25T10:0{i}:00+08:00",
            message_id=None,
            idempotency_key=None,
        )

    results = await asyncio.gather(run(1), run(2))
    assert all(r["ok"] for r in results)
    assert buffer.turn_count(AGENT, SESSION) == 2
    assert buffer.size(AGENT, SESSION) == 4
    assert sorted(r["rounds"] for r in results) == [1, 2]


async def test_buffer_overflow(tmp_path):
    settings = _settings(tmp_path, memory_api_buffer_max_messages=2)
    service, buffer = _service(_store(), _FakeLLM(), settings)
    async with _client(service) as client:
        assert (await _turn(client)).status_code == 200
        resp = await _turn(client, user_text="第二轮：今天有点累想早点休息", replies=["好。"])
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "buffer_overflow"
    assert buffer.turn_count(AGENT, SESSION) == 1
    assert buffer.size(AGENT, SESSION) == 2


async def test_idempotent_concurrent_same_key(tmp_path):
    """R2：同 key 并发只能写一次；后到者拿锁即命中幂等缓存。"""
    service, buffer = _service(_store(), _FakeLLM(), _settings(tmp_path))
    payload = {
        "agent": AGENT,
        "session": SESSION,
        "user_text": USER_TEXT,
        "replies": ["那就早点休息。"],
        "at": "2026-10-25T10:35:00+08:00",
        "message_id": "m-1",
        "idempotency_key": "turn-concurrent",
    }
    first, second = await asyncio.gather(
        service.turn(**payload), service.turn(**payload)
    )
    assert sorted([first["idempotent"], second["idempotent"]]) == [False, True]
    assert first["rounds"] == 1 and second["rounds"] == 1
    assert buffer.turn_count(AGENT, SESSION) == 1
    assert buffer.size(AGENT, SESSION) == 2


async def test_multi_reply_overflow(tmp_path):
    """R3：一轮多条回复时按本轮实际追加条数判上限，不得静默突破。"""
    settings = _settings(tmp_path, memory_api_buffer_max_messages=3)
    service, buffer = _service(_store(), _FakeLLM(), settings)
    async with _client(service) as client:
        first = await _turn(client, replies=["回复一。", "回复二。"])
        second = await _turn(
            client, user_text="第二轮：今天有点累想早点休息", replies=["回复三。", "回复四。"]
        )
    assert first.status_code == 200
    assert first.json()["buffer_size"] == 3
    assert second.status_code == 503
    assert second.json()["error"]["code"] == "buffer_overflow"
    assert buffer.turn_count(AGENT, SESSION) == 1
    assert buffer.size(AGENT, SESSION) == 3  # 第二轮一条都没追加


async def test_save_failure_rolls_back_slot(tmp_path, monkeypatch):
    """R4：写盘失败时内存槽必须回到本轮之前，重试不得重复计数/追加。"""
    settings = _settings(tmp_path)
    service, buffer = _service(_store(), _FakeLLM(), settings)
    original_save = buffer.save

    async def _boom() -> None:
        raise BufferPersistError("模拟写盘失败")

    async with _client(service) as client:
        monkeypatch.setattr(buffer, "save", _boom)
        failed = await _turn(client)
        assert failed.status_code == 503
        assert failed.json()["error"]["code"] == "buffer_persist_failed"
        assert buffer.turn_count(AGENT, SESSION) == 0
        assert buffer.size(AGENT, SESSION) == 0
        assert buffer.last_turn_at(AGENT, SESSION) is None

        monkeypatch.setattr(buffer, "save", original_save)
        retried = await _turn(client)
    assert retried.status_code == 200
    assert retried.json()["rounds"] == 1
    assert buffer.turn_count(AGENT, SESSION) == 1
    assert buffer.size(AGENT, SESSION) == 2


# ---------------------------------------------------------------------------
# forget / state
# ---------------------------------------------------------------------------
async def test_forget_deleted_missing_and_facts_kept(tmp_path):
    settings = _settings(tmp_path)
    store = _store()
    service, _ = _service(store, _FakeLLM(), settings)
    await service.turn(
        agent=AGENT,
        session=SESSION,
        user_text=USER_TEXT,
        replies=["那就早点休息。"],
        at="2026-10-25T10:35:00+08:00",
        message_id=None,
        idempotency_key=None,
    )
    before = await service.state(agent=AGENT, session=SESSION)
    assert before["facts"] == 1
    assert before["profile_slots"] == 1

    async with _client(service) as client:
        resp = await client.post(
            "/api/v1/memory/forget",
            json={"agent": AGENT, "contents": [USER_TEXT, "根本不存在的话", USER_TEXT]},
        )
    body = resp.json()
    assert resp.status_code == 200
    assert body["deleted"] == 1
    assert body["missing"] == ["根本不存在的话"]

    after = await service.state(agent=AGENT, session=SESSION)
    assert after["entries"] == before["entries"] - 1
    assert after["facts"] == 1  # facts 不被删
    assert after["profile_slots"] == 1  # profile 不被删
    assert USER_TEXT not in await store.search(AGENT, query=None, limit=20)


async def test_state_entries_paginated(tmp_path):
    settings = _settings(tmp_path)
    store = _store()
    service, _ = _service(store, _FakeLLM(), settings)
    for i in range(1005):
        await store.put(AGENT, f"批量记忆-{i}")
    state = await service.state(agent=AGENT, session=None)
    assert state["entries"] == 1005  # 分页统计，不受单次 1000 封顶


async def test_state_session_optional_aggregates(tmp_path):
    settings = _settings(tmp_path, memory_extract_enabled=False)
    service, buffer = _service(_store(), _FakeLLM(), settings)
    for session, text in [(SESSION, "主会话消息"), ("alt", "另一个会话")]:
        await service.turn(
            agent=AGENT,
            session=session,
            user_text=f"{text}：今天有点累想早点休息",
            replies=["好。"],
            at="2026-10-25T10:00:00+08:00",
            message_id=None,
            idempotency_key=None,
        )
    async with _client(service) as client:
        one = (await client.get("/api/v1/memory/state", params={"agent": AGENT, "session": SESSION})).json()
        both = (await client.get("/api/v1/memory/state", params={"agent": AGENT})).json()
    assert one["buffer_size"] == 2
    assert both["buffer_size"] == 4
    assert both["session"] is None
    assert both["last_turn_at"] == "2026-10-25T10:00:00+08:00"


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------
async def test_context_returns_profile_facts_memories(tmp_path):
    settings = _settings(tmp_path)
    store = _store()
    service, _ = _service(store, _FakeLLM(), settings)
    await service.turn(
        agent=AGENT,
        session=SESSION,
        user_text=USER_TEXT,
        replies=["那就早点休息。"],
        at="2026-10-25T10:35:00+08:00",
        message_id=None,
        idempotency_key=None,
    )
    async with _client(service) as client:
        resp = await client.post(
            "/api/v1/memory/context",
            json={"agent": AGENT, "session": SESSION, "query": "用户今天状态", "limit": 5},
        )
    body = resp.json()
    assert resp.status_code == 200
    assert body["ok"] is True and body["contract"] == "1"
    assert body["boundaries"] == []
    assert body["profile"] == {"称呼": "阿棠"}
    assert body["facts"][0]["object"] == "有点累"
    assert set(body["facts"][0]) == {
        "text",
        "subject",
        "predicate",
        "object",
        "importance",
        "updated_at",
    }
    assert any(m.startswith("用户说过：") for m in body["memories"])
    assert set(body["debug"]) == {"pool", "top"}
    assert body["debug"]["pool"] >= 0


async def test_context_extract_disabled(tmp_path):
    settings = _settings(tmp_path, memory_extract_enabled=False)
    store = _store()
    service, _ = _service(store, _FakeLLM(), settings)
    await service.turn(
        agent=AGENT,
        session=SESSION,
        user_text=USER_TEXT,
        replies=["那就早点休息。"],
        at="2026-10-25T10:35:00+08:00",
        message_id=None,
        idempotency_key=None,
    )
    async with _client(service) as client:
        body = (
            await client.post(
                "/api/v1/memory/context",
                json={"agent": AGENT, "session": SESSION, "query": ""},
            )
        ).json()
    assert body["profile"] == {}
    assert body["facts"] == []
    assert body["l2"] == ""
    assert body["debug"] == {"pool": 0, "top": []}
    assert body["memories"]  # 退化召回仍有记忆（无角色前缀）


# ---------------------------------------------------------------------------
# 错误信封 / 白名单 / boundary 不存在
# ---------------------------------------------------------------------------
async def test_error_envelope_validation(tmp_path):
    service, _ = _service(_store(), _FakeLLM(), _settings(tmp_path))
    async with _client(service) as client:
        missing = await client.post("/api/v1/memory/turn", json={"agent": AGENT, "session": SESSION})
        bad_state = await client.get("/api/v1/memory/state")
    for resp in (missing, bad_state):
        assert resp.status_code == 422
        body = resp.json()
        assert body["ok"] is False
        assert body["contract"] == "1"
        assert body["error"]["code"] == "invalid_request"
        assert body["error"]["message"]


async def test_agent_whitelist(tmp_path):
    service, _ = _service(_store(), _FakeLLM(), _settings(tmp_path))
    async with _client(service) as client:
        numeric = await _turn(client, agent="999")
        unknown = await _turn(client, agent="other")
    for resp in (numeric, unknown):
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "agent_not_allowed"


async def test_boundary_endpoint_absent(tmp_path):
    service, _ = _service(_store(), _FakeLLM(), _settings(tmp_path))
    async with _client(service) as client:
        resp = await client.post("/api/v1/memory/boundary", json={})
    assert resp.status_code in (404, 405)
