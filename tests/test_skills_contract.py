"""五个内置工具的契约表单测（§2.5 参数/返回 data/失败模式；§2.6 错误码清单）。

与 `test_tools_basic.py` / `test_tools_lore_reminder.py`（正常路径 + 超时 + 空结果）互补：
本文件把 §2.5 表格**逐格**钉成断言（参数模型字段与默认值、成功 `data` 键、失败 `error_type`），
并校验 §8.2 的 10 个错误码与 §2.1 数据结构的默认值。
"""

import random

import pytest

from app.skills.base import ToolCall, ToolContext, ToolResult
from app.skills.builtin.dice import RollDiceArgs, make_roll_dice_skill
from app.skills.builtin.read_lore import ReadLoreArgs, make_read_lore_skill
from app.skills.builtin.reminder import SetReminderArgs, make_set_reminder_skill, parse_when
from app.skills.builtin.time import GetCurrentTimeArgs, get_current_time_skill
from app.skills.builtin.web_search import FakeSearchProvider, WebSearchArgs, make_web_search_skill


def _ctx(**kw) -> ToolContext:
    return ToolContext(agent_id=1, session_id=2, **kw)


# ---- §8.2 错误码清单 ----

CONTRACT_ERROR_TYPES = (
    "invalid_arguments",
    "unknown_tool",
    "timeout",
    "upstream_error",
    "empty",
    "truncated",
    "loop_detected",
    "unavailable",
    "context_missing",
    "internal",
)


def test_error_type_closed_set():
    """所有失败通道产出的 `error_type` 必须落在 §8.2 的 10 个之内（不新增私货）。"""
    from pydantic import BaseModel as _BM

    from app.core.errors import UpstreamError
    from app.skills.base import BaseSkill as _BS
    from app.skills.executor import ToolExecutor
    from app.skills.registry import SkillRegistry

    class _A(_BM):
        value: str  # 必填：空 arguments → invalid_arguments

    def _mk(handler, name):
        return _BS(
            name=name, description="d", when_to_use="w", parameters=_A,
            handler=handler, max_retries=0,
        )

    async def ok(ctx, args):
        return ToolResult(ok=True, data={})

    async def crash(ctx, args):
        raise RuntimeError("kaboom")

    async def upstream(ctx, args):
        raise UpstreamError("5xx")

    async def run():
        ex = ToolExecutor(SkillRegistry([_mk(ok, "t"), _mk(crash, "crash"), _mk(upstream, "up")]))
        seen = set()
        seen.add((await ex.execute(ToolCall("1", "t", {}), _ctx())).error_type)
        seen.add((await ex.execute(ToolCall("2", "ghost", {}), _ctx())).error_type)
        seen.add((await ex.execute(ToolCall("3", "crash", {"value": "x"}), _ctx())).error_type)
        seen.add((await ex.execute(ToolCall("4", "up", {"value": "x"}), _ctx())).error_type)
        return seen

    import asyncio

    seen = asyncio.run(run())
    assert seen <= set(CONTRACT_ERROR_TYPES)
    assert {"invalid_arguments", "unknown_tool", "internal", "upstream_error"} <= seen


# ---- §2.1 数据结构默认值 ----

def test_tool_result_defaults_and_tool_context():
    res = ToolResult(ok=False)
    assert (res.data, res.error_type, res.message, res.latency_ms, res.cached) == (
        None, None, "", 0, False,
    )
    ctx = ToolContext(agent_id=1, session_id=2)
    assert ctx.user_id is None and ctx.allowed_skills == ()
    call = ToolCall(id="c1", name="n", arguments={})
    with pytest.raises(Exception):  # frozen dataclass
        call.name = "other"  # type: ignore[misc]


# ---- §2.5-1 get_current_time ----

def test_get_current_time_params_and_success():
    import asyncio

    assert GetCurrentTimeArgs().timezone == "Asia/Shanghai"  # 默认时区
    res = asyncio.run(get_current_time_skill.handler(_ctx(), GetCurrentTimeArgs(timezone="UTC")))
    assert res.ok is True
    assert set(res.data) == {"iso", "human", "timezone", "offset"}
    assert res.data["timezone"] == "UTC"


def test_get_current_time_invalid_timezone():
    import asyncio

    res = asyncio.run(
        get_current_time_skill.handler(_ctx(), GetCurrentTimeArgs(timezone="Mars/Olympus"))
    )
    assert res.ok is False and res.error_type == "invalid_timezone"
    assert res.data == {"timezone": "Mars/Olympus"}


# ---- §2.5-2 roll_dice ----

def test_roll_dice_params_and_success():
    assert RollDiceArgs().dice == "1d6" and RollDiceArgs().times == 1
    import asyncio

    skill = make_roll_dice_skill(rng=random.Random(0))  # 测试注入种子
    res = asyncio.run(skill.handler(_ctx(), RollDiceArgs(dice="2d6", times=3)))
    assert res.ok is True
    assert set(res.data) == {"rolls", "total", "dice", "times"}
    assert len(res.data["rolls"]) == 3 and all(len(g) == 2 for g in res.data["rolls"])
    assert res.data["total"] == sum(sum(g) for g in res.data["rolls"])


@pytest.mark.parametrize(
    "dice,times",
    [("0d6", 1), ("11d6", 1), ("1d1", 1), ("1d101", 1), ("d6", 1), ("1d6", 0), ("1d6", 11)],
)
def test_roll_dice_invalid_forms(dice, times):
    import asyncio

    skill = make_roll_dice_skill()
    res = asyncio.run(skill.handler(_ctx(), RollDiceArgs(dice=dice, times=times)))
    assert res.ok is False and res.error_type == "invalid_dice"
    assert res.data == {"dice": dice, "times": times}


# ---- §2.5-3 web_search ----

def test_web_search_params_defaults():
    args = WebSearchArgs(query="x")
    assert args.top_k == 3
    with pytest.raises(Exception):
        WebSearchArgs(query="")


def test_web_search_unavailable_without_provider():
    import asyncio

    skill = make_web_search_skill(None)
    res = asyncio.run(skill.handler(_ctx(), WebSearchArgs(query="天气")))
    assert res.ok is False and res.error_type == "unavailable"


def test_web_search_fake_provider_shape():
    import asyncio

    fake = FakeSearchProvider([{"title": "t", "url": "u", "snippet": "s"}])
    skill = make_web_search_skill(fake)
    res = asyncio.run(skill.handler(_ctx(), WebSearchArgs(query="天气")))
    assert res.ok is True
    assert set(res.data) == {"results", "provider"}
    assert res.data["provider"] == "fake"


# ---- §2.5-4 set_reminder ----

def test_set_reminder_params_and_parse_when():
    assert set(SetReminderArgs.model_fields) == {"when", "content"}
    assert parse_when("2099-01-01 09:00") is not None  # 未来时间可解析
    assert parse_when("2000-01-01 09:00") is None  # 早于当前 → None
    assert parse_when("not-a-time") is None


def test_set_reminder_invalid_time_and_content():
    import asyncio

    skill = make_set_reminder_skill(None)
    bad_time = asyncio.run(skill.handler(_ctx(), SetReminderArgs(when="2000-01-01 09:00", content="x")))
    assert bad_time.error_type == "invalid_time" and bad_time.data == {"when": "2000-01-01 09:00"}
    bad_content = asyncio.run(
        skill.handler(_ctx(), SetReminderArgs(when="2099-01-01 09:00", content="  "))
    )
    assert bad_content.error_type == "invalid_content"


def test_set_reminder_unavailable_without_session_factory():
    import asyncio

    skill = make_set_reminder_skill(None)
    res = asyncio.run(
        skill.handler(_ctx(), SetReminderArgs(when="2099-01-01 09:00", content="交房租"))
    )
    assert res.ok is False and res.error_type == "unavailable"


def test_set_reminder_is_side_effect():
    assert make_set_reminder_skill().side_effect is True


# ---- §2.5-5 read_lore ----

class _Hit:
    def __init__(self, ref, header, content, score=1.0):
        self.source_ref, self.header_path, self.content, self.score = ref, header, content, score


class _Result:
    def __init__(self, reranked, injected):
        self.reranked, self.injected = reranked, injected


class _FakeRetriever:
    def __init__(self, *, hits=True):
        self._hits = hits

    async def retrieve(self, *, agent_id, query):  # noqa: ANN001
        if not self._hits:
            return _Result([], [])
        return _Result(
            [("yunqi_tea:01", 0.9)],
            [_Hit("yunqi_tea:01", "创始人", "云栖茶庄由苏云栖创办。", 1.0)],
        )

    async def _load_rows(self, *, agent_id, refs):  # noqa: ANN001
        return {}


def test_read_lore_params_defaults():
    args = ReadLoreArgs(query="招牌茶")
    assert args.top_k == 5
    with pytest.raises(Exception):
        ReadLoreArgs(query="")


def test_read_lore_unavailable_without_retriever():
    import asyncio

    skill = make_read_lore_skill(None)
    res = asyncio.run(skill.handler(_ctx(), ReadLoreArgs(query="招牌茶")))
    assert res.ok is False and res.error_type == "unavailable"


def test_read_lore_success_chunk_shape():
    import asyncio

    skill = make_read_lore_skill(_FakeRetriever())
    res = asyncio.run(skill.handler(_ctx(), ReadLoreArgs(query="招牌茶", top_k=3)))
    assert res.ok is True
    chunk = res.data["chunks"][0]
    assert set(chunk) == {"source_ref", "header_path", "content", "score"}
    assert chunk["source_ref"] == "yunqi_tea:01"


def test_read_lore_empty_when_no_hit():
    import asyncio

    skill = make_read_lore_skill(_FakeRetriever(hits=False))
    res = asyncio.run(skill.handler(_ctx(), ReadLoreArgs(query="无关问题")))
    assert res.ok is False and res.error_type == "empty"
