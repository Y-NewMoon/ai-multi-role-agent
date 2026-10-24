"""内置工具单测 ①：get_current_time / roll_dice / web_search（契约 §2.5-1~3）。"""

import asyncio
import random

from app.skills.base import ToolCall, ToolContext
from app.skills.builtin.dice import make_roll_dice_skill
from app.skills.builtin.time import get_current_time_skill
from app.skills.builtin.web_search import FakeSearchProvider, make_web_search_skill
from app.skills.executor import ToolExecutor
from app.skills.registry import SkillRegistry

CTX = ToolContext(agent_id=1, session_id=2)


def _noop_sleep():
    async def _sleep(sec: float) -> None:
        pass

    return _sleep


async def _call(skill, args, **kw):
    ex = ToolExecutor(SkillRegistry([skill]), **kw)
    return await ex.execute(ToolCall(id="c", name=skill.name, arguments=args), CTX)


# ---- get_current_time ----


async def test_get_current_time_default_ok():
    res = await _call(get_current_time_skill, {})
    assert res.ok
    assert res.data["timezone"] == "Asia/Shanghai"
    assert res.data["offset"] == "+08:00"
    assert "Asia/Shanghai" in res.data["human"]
    assert res.data["iso"]


async def test_get_current_time_tokyo_offset():
    res = await _call(get_current_time_skill, {"timezone": "Asia/Tokyo"})
    assert res.data["offset"] == "+09:00"


async def test_get_current_time_invalid_timezone():
    res = await _call(get_current_time_skill, {"timezone": "Mars/Olympus"})
    assert res.ok is False
    assert res.error_type == "invalid_timezone"
    assert res.data["timezone"] == "Mars/Olympus"


# ---- roll_dice ----


async def test_roll_dice_seeded():
    rng = random.Random(42)
    skill = make_roll_dice_skill(rng=rng)
    res = await _call(skill, {"dice": "2d6", "times": 3})
    assert res.ok
    assert len(res.data["rolls"]) == 3
    assert all(len(group) == 2 for group in res.data["rolls"])
    assert res.data["total"] == sum(sum(g) for g in res.data["rolls"])
    assert res.data["dice"] == "2d6" and res.data["times"] == 3


async def test_roll_dice_bounds():
    assert (await _call(make_roll_dice_skill(), {"dice": "1d6"})).ok
    assert (await _call(make_roll_dice_skill(), {"dice": "10d100", "times": 10})).ok


async def test_roll_dice_invalid():
    for bad in ({"dice": "0d6"}, {"dice": "1d1"}, {"dice": "11d6"}, {"dice": "1d101"}, {"dice": "abc"}):
        res = await _call(make_roll_dice_skill(), bad)
        assert res.error_type == "invalid_dice"
    res = await _call(make_roll_dice_skill(), {"dice": "1d6", "times": 11})
    assert res.error_type == "invalid_dice"


# ---- web_search ----


async def test_web_search_unavailable_without_provider():
    res = await _call(make_web_search_skill(None), {"query": "天气"})
    assert res.error_type == "unavailable"


async def test_web_search_unavailable_when_provider_not_available():
    provider = FakeSearchProvider([{"title": "t", "url": "u", "snippet": "s"}], available=False)
    res = await _call(make_web_search_skill(provider), {"query": "天气"})
    assert res.error_type == "unavailable"


async def test_web_search_ok_and_provider_name():
    provider = FakeSearchProvider(
        [{"title": "标题", "url": "http://x", "snippet": "摘要"}] * 4,
    )
    res = await _call(make_web_search_skill(provider), {"query": "天气", "top_k": 3})
    assert res.ok
    assert res.data["provider"] == "fake"
    assert len(res.data["results"]) == 3


async def test_web_search_empty():
    res = await _call(make_web_search_skill(FakeSearchProvider([])), {"query": "天气"})
    assert res.ok is False
    assert res.error_type == "empty"


async def test_web_search_invalid_arguments():
    provider = FakeSearchProvider([])
    assert (await _call(make_web_search_skill(provider), {"query": ""})).error_type == "invalid_arguments"
    assert (await _call(make_web_search_skill(provider), {"query": "x", "top_k": 9})).error_type == "invalid_arguments"


async def test_web_search_timeout_via_executor():
    async def slow(query, top_k):
        await asyncio.sleep(0.05)
        return []

    provider = FakeSearchProvider([])
    provider.search = slow  # type: ignore[method-assign]
    skill = make_web_search_skill(provider)
    skill.timeout_s = 0.01  # 缩短超时以便快速断言（生产默认 5s）
    res = await _call(skill, {"query": "天气"}, sleep=_noop_sleep())
    assert res.error_type == "timeout"
