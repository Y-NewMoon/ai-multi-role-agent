"""Skill 注册表 / 执行器单测（契约 §2.2 七条硬要求）。"""

import asyncio

import pytest
from pydantic import BaseModel

from app.core.errors import UpstreamError
from app.skills.base import BaseSkill, ToolCall, ToolContext, ToolResult
from app.skills.executor import (
    CIRCUIT_COOLDOWN_S,
    MAX_RESULT_TOKENS,
    ToolExecutor,
    canonical_key,
)
from app.skills.registry import SkillRegistry


class _Args(BaseModel):
    timezone: str


def _ctx(**kw) -> ToolContext:
    return ToolContext(agent_id=1, session_id=2, **kw)


def _skill(handler, *, name="probe", timeout_s=5.0, max_retries=1) -> BaseSkill:
    return BaseSkill(
        name=name,
        description="probe",
        when_to_use="test",
        parameters=_Args,
        handler=handler,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )


def _noop_sleep():
    calls: list[float] = []

    async def _sleep(sec: float) -> None:
        calls.append(sec)

    return _sleep, calls


# ---- 注册表 ----

def test_registry_rejects_duplicate_name():
    s = _skill(lambda *a: None)
    with pytest.raises(ValueError):
        SkillRegistry([s, s])


def test_registry_ignores_unknown_and_keeps_order():
    a = _skill(lambda *a: None, name="aaa")
    b = _skill(lambda *a: None, name="bbb")
    reg = SkillRegistry([a, b])
    assert [x.name for x in reg.resolve(["bbb", "nope", "aaa"])] == ["bbb", "aaa"]
    assert reg.names() == ["aaa", "bbb"]
    assert [t["function"]["name"] for t in reg.to_openai_tools(["aaa"])] == ["aaa"]


def test_to_openai_tool_shape():
    tool = _skill(lambda *a: None).to_openai_tool()
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "probe"
    assert tool["function"]["parameters"]["additionalProperties"] is False


# ---- 执行器 硬要求 1：参数校验 ----

async def test_invalid_arguments_returns_fields():
    ex = ToolExecutor(SkillRegistry([_skill(lambda *a: None)]))
    res = await ex.execute(ToolCall(id="1", name="probe", arguments={}), _ctx())
    assert res.ok is False
    assert res.error_type == "invalid_arguments"
    assert "timezone" in res.data["fields"]


# ---- 硬要求 2：超时 + 重试 ----

async def test_timeout_retries_then_timeout():
    async def slow(ctx, args):
        await asyncio.sleep(0.05)
        return ToolResult(ok=True, data={})

    sleep, backoffs = _noop_sleep()
    skill = _skill(slow, timeout_s=0.01, max_retries=1)
    ex = ToolExecutor(SkillRegistry([skill]), sleep=sleep)
    res = await ex.execute(ToolCall(id="1", name="probe", arguments={"timezone": "UTC"}), _ctx())
    assert res.error_type == "timeout"
    assert backoffs == [0.5]  # 退避 0.5s 一次（=max_retries 次重试）


# ---- 硬要求 3：上游错误重试 + 熔断 ----

async def test_upstream_error_retries_then_circuit_opens():
    calls = {"n": 0}

    async def boom(ctx, args):
        calls["n"] += 1
        raise UpstreamError("5xx")

    sleep, _ = _noop_sleep()
    skill = _skill(boom, max_retries=1)
    ex = ToolExecutor(SkillRegistry([skill]), sleep=sleep)
    first = await ex.execute(ToolCall(id="1", name="probe", arguments={"timezone": "UTC"}), _ctx())
    assert first.error_type == "upstream_error"
    assert calls["n"] == 2  # 初次 + 重试 1 次
    # 熔断：冷却期内不再执行 handler
    second = await ex.execute(ToolCall(id="2", name="probe", arguments={"timezone": "UTC"}), _ctx())
    assert second.error_type == "upstream_error"
    assert calls["n"] == 2


async def test_circuit_expires_after_cooldown():
    clock = {"t": 1000.0}
    calls = {"n": 0}

    async def boom(ctx, args):
        calls["n"] += 1
        raise UpstreamError("5xx")

    sleep, _ = _noop_sleep()
    skill = _skill(boom, max_retries=0)
    ex = ToolExecutor(SkillRegistry([skill]), sleep=sleep, clock=lambda: clock["t"])
    await ex.execute(ToolCall(id="1", name="probe", arguments={"timezone": "UTC"}), _ctx())
    clock["t"] += CIRCUIT_COOLDOWN_S + 1
    await ex.execute(ToolCall(id="2", name="probe", arguments={"timezone": "UTC"}), _ctx())
    assert calls["n"] == 2  # 冷却结束后重新执行


# ---- 硬要求 4：结果截断 ----

async def test_result_truncated_over_budget():
    async def big(ctx, args):
        return ToolResult(ok=True, data={"results": [{"snippet": "中" * 400} for _ in range(50)]})

    ex = ToolExecutor(SkillRegistry([_skill(big)]))
    res = await ex.execute(ToolCall(id="1", name="probe", arguments={"timezone": "UTC"}), _ctx())
    assert res.ok is True
    assert res.data["truncated"] is True
    assert "summary" in res.data
    from app.lore.chunker import estimate_tokens

    assert estimate_tokens(str(res.data)) <= MAX_RESULT_TOKENS * 2  # 结构保留后的近似上界


async def test_result_within_budget_untouched():
    async def small(ctx, args):
        return ToolResult(ok=True, data={"a": 1})

    ex = ToolExecutor(SkillRegistry([_skill(small)]))
    res = await ex.execute(ToolCall(id="1", name="probe", arguments={"timezone": "UTC"}), _ctx())
    assert res.data == {"a": 1}
    assert "truncated" not in res.data


# ---- 硬要求 5：同请求缓存（一次图运行内有效，跨请求必须失效）----

async def test_same_request_cached_second_time():
    calls = {"n": 0}

    async def ok(ctx, args):
        calls["n"] += 1
        return ToolResult(ok=True, data={"n": calls["n"]})

    ex = ToolExecutor(SkillRegistry([_skill(ok)]))
    call = ToolCall(id="1", name="probe", arguments={"timezone": "UTC"})
    first = await ex.execute(call, _ctx())
    second = await ex.execute(call, _ctx())
    assert first.cached is False
    assert second.cached is True
    assert calls["n"] == 1


async def test_cache_reset_between_requests():
    """同一 executor 的两次「请求」之间缓存必须失效（R1）。"""
    calls = {"n": 0}

    async def ok(ctx, args):
        calls["n"] += 1
        return ToolResult(ok=True, data={"n": calls["n"]})

    ex = ToolExecutor(SkillRegistry([_skill(ok)]))
    call = ToolCall(id="1", name="probe", arguments={"timezone": "UTC"})
    await ex.execute(call, _ctx())
    ex.reset()  # 新一次请求
    again = await ex.execute(call, _ctx())
    assert again.cached is False
    assert again.data == {"n": 2}
    assert calls["n"] == 2


class _FakeHit:
    def __init__(self, agent_id: int) -> None:
        self.source_ref = f"a{agent_id}:01"
        self.header_path = "h"
        self.content = f"agent-{agent_id}-chunk"
        self.score = 1.0


class _FakeLoreResult:
    def __init__(self, hit: _FakeHit) -> None:
        self.reranked = [(hit.source_ref, hit.score)]
        self.injected = [hit]


class _FakeRetriever:
    """按 agent_id 返回不同 chunk，用于验证缓存不跨角色共享。"""

    async def retrieve(self, *, agent_id: int, query: str):  # noqa: ANN001, ARG002
        return _FakeLoreResult(_FakeHit(agent_id))

    async def _load_rows(self, *, agent_id: int, refs):  # noqa: ANN001, ARG002
        return {}


async def test_read_lore_cache_not_shared_between_agents():
    from app.skills.builtin.read_lore import make_read_lore_skill

    ex = ToolExecutor(SkillRegistry([make_read_lore_skill(_FakeRetriever())]))
    call = ToolCall(id="1", name="read_lore", arguments={"query": "茶馆"})
    a1 = await ex.execute(call, ToolContext(agent_id=1, session_id=1))
    a2 = await ex.execute(call, ToolContext(agent_id=2, session_id=2))
    assert a1.data["chunks"][0]["content"] == "agent-1-chunk"
    assert a2.data["chunks"][0]["content"] == "agent-2-chunk"
    assert a2.cached is False


async def test_cache_key_order_insensitive():
    assert canonical_key("t", {"a": 1, "b": 2}) == canonical_key("t", {"b": 2, "a": 1})
    assert canonical_key("t", {"a": 1}) != canonical_key("t2", {"a": 1})


# ---- 硬要求 7：异常兜底 ----

async def test_uncaught_exception_becomes_internal():
    async def crash(ctx, args):
        raise RuntimeError("kaboom")

    ex = ToolExecutor(SkillRegistry([_skill(crash)]))
    res = await ex.execute(ToolCall(id="1", name="probe", arguments={"timezone": "UTC"}), _ctx())
    assert res.ok is False
    assert res.error_type == "internal"


# ---- 未知工具 / 未授权 ----

async def test_unknown_tool_and_unauthorized():
    ex = ToolExecutor(SkillRegistry([_skill(lambda *a: None)]))
    unknown = await ex.execute(ToolCall(id="1", name="nope", arguments={}), _ctx())
    assert unknown.error_type == "unknown_tool"
    assert unknown.data["available"] == ["probe"]
    unauth = await ex.execute(
        ToolCall(id="2", name="probe", arguments={"timezone": "UTC"}),
        _ctx(allowed_skills=("other",)),
    )
    assert unauth.error_type == "unknown_tool"
