"""V3 M1 时间感全量化（契约 §2.1–§2.6）。

覆盖：
- 时间块常驻（无工具角色 / 空闲轮）、位置 `base → time → …`、不参与裁剪；
- 间隔四档与 §8.1 边界（9m59s / 10min / 59min / 3h / 隔天 / 跨午夜 / 7 天 / 首次）；
- 消息前缀四种（首条 / 同日 <2h / 同日 >2h / 跨天）与旧消息（无 `at`）兼容；
- 摘要输入日期与摘要 prompt 指令；
- 冻结时钟（固定 / 未固定 / 非法 / 优先级）；
- 服务层三入口（私聊 / 群聊 / 空闲）的 `last_user_at` 注入与 fail-soft 查询。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.nodes.compose_prompt import (
    TIME_HEADER,
    TZ,
    _now_block,
    _to_chat_messages,
    compose_prompt,
)
from app.agents.supervisor import GroupMember, Supervisor
from app.core import clock
from app.memory.summarizer import _format_conversation
from app.prompts.summarize import SUMMARIZE_PROMPT
from app.services.chat_service import ChatContext, ChatService
from app.services.group_service import GroupService

NOW = datetime(2026, 10, 25, 10, 35, tzinfo=TZ)  # 周日（真实日历）
NOW_TEXT = "现在是 2026-10-25 10:35（Asia/Shanghai，周日）。"
TIME_INSTRUCTION = "用户说「今晚8点35」「明天早上7点」这类相对时间时，先按此换算成绝对时间。"


def _block(last_user_at: str | None = None) -> str:
    return _now_block(NOW, last_user_at)


# ---------------------------------------------------------------------------
# 时间块（§2.1）
# ---------------------------------------------------------------------------
async def test_time_block_all_roles_and_position():
    state = {
        "system_prompt": "人设",
        "summary": "摘要",
        "recalled_memories": [],
        "recalled_lore": [],
        "messages": [HumanMessage(content="你好")],
    }
    text = (await compose_prompt(state, now=NOW))["prompt"][0]["content"]
    assert text == f"人设\n\n{TIME_HEADER}\n{NOW_TEXT}\n{TIME_INSTRUCTION}\n\n【本会话摘要】\n摘要"


async def test_time_block_present_without_tools_or_registry():
    state = {"system_prompt": "人设", "messages": [HumanMessage(content="hi")]}
    text = (await compose_prompt(state, now=NOW))["prompt"][0]["content"]
    assert text == f"人设\n\n{TIME_HEADER}\n{NOW_TEXT}\n{TIME_INSTRUCTION}"


async def test_idle_turn_has_time_block_after_base():
    state = {
        "system_prompt": "人设",
        "idle_scope": "private",
        "messages": [AIMessage(content="上一条")],
    }
    text = (await compose_prompt(state, now=NOW))["prompt"][0]["content"]
    assert TIME_HEADER in text
    assert text.index("人设") < text.index(TIME_HEADER)


async def test_time_block_survives_budget_trim():
    state = {
        "system_prompt": "人设",
        "summary": "很长的摘要" * 200,
        "recalled_memories": ["记忆" * 100],
        "messages": [HumanMessage(content="最后一句")],
    }
    out = await compose_prompt(state, token_budget=1, now=NOW)
    text = out["prompt"][0]["content"]
    assert f"{TIME_HEADER}\n{NOW_TEXT}\n{TIME_INSTRUCTION}" in text  # 时间块常驻
    assert "【本会话摘要】" not in text  # 其余仍按原顺序裁剪


# ---------------------------------------------------------------------------
# 间隔分档（§2.2 / §8.1）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("last", "expected"),
    [
        ("2026-10-25T10:25:01+08:00", None),  # 9m59s
        ("2026-10-25T10:25:00+08:00", "你们上次说话是 10 分钟前。"),  # 10min 边界
        ("2026-10-25T09:36:00+08:00", "你们上次说话是 59 分钟前。"),
        ("2026-10-25T07:35:00+08:00", "你们上次说话是今天早些时候。"),  # 3h 同日
        ("2026-10-24T10:35:00+08:00", "你们上次说话是昨天。"),  # 隔天
        ("2026-10-24T23:50:00+08:00", "你们上次说话是昨天。"),  # 10h45m，日历日差 1
        ("2026-10-18T10:35:00+08:00", "你们上次说话是 7 天前，隔了挺久。"),
    ],
)
def test_interval_bucket(last, expected):
    lines = _block(last).split("\n")
    assert lines[:3] == [TIME_HEADER, NOW_TEXT, TIME_INSTRUCTION]
    if expected is None:
        assert len(lines) == 3
    else:
        assert lines[3] == expected
        assert len(lines) == 4


def test_interval_cross_midnight_is_minutes_not_yesterday():
    now = datetime(2026, 10, 25, 0, 35, tzinfo=TZ)
    lines = _now_block(now, "2026-10-25T00:15:00+08:00").split("\n")
    assert lines[3] == "你们上次说话是 20 分钟前。"


def test_interval_uses_calendar_days_not_24h():
    # 昨天 02:00 → 今天 08:00 = 30 小时，但日历日差 1 → “昨天”，不是“今天早些时候”
    now = datetime(2026, 10, 25, 8, 0, tzinfo=TZ)
    lines = _now_block(now, "2026-10-24T02:00:00+08:00").split("\n")
    assert lines[3] == "你们上次说话是昨天。"


@pytest.mark.parametrize("last", [None, "", "   ", "不是时间", "2026-10-25T11:00:00+08:00"])
def test_interval_silent_when_missing_invalid_or_future(last):
    assert len(_block(last).split("\n")) == 3


# ---------------------------------------------------------------------------
# 消息前缀（§2.3）
# ---------------------------------------------------------------------------
def test_message_prefix_first_short_gap_long_gap_cross_day():
    messages = [
        HumanMessage(content="A", additional_kwargs={"at": "2026-10-25T10:30:00+08:00"}),
        AIMessage(content="B"),  # 无 at：不更新比较基准
        HumanMessage(content="C", additional_kwargs={"at": "2026-10-25T10:31:00+08:00"}),
        HumanMessage(content="D", additional_kwargs={"at": "2026-10-25T14:20:00+08:00"}),
        HumanMessage(content="E", additional_kwargs={"at": "2026-10-26T09:00:00+08:00"}),
    ]
    rendered = [m["content"] for m in _to_chat_messages(messages)]
    assert rendered == [
        "[10-25 10:30] A",
        "B",
        "[10:31] C",
        "[10-25 14:20] D",
        "[10-26 09:00] E",
    ]


def test_message_without_at_is_unchanged():
    rendered = _to_chat_messages([HumanMessage(content="旧消息\n第二行")])
    assert rendered == [{"role": "user", "content": "旧消息\n第二行"}]


def test_message_multiline_gets_single_prefix():
    rendered = _to_chat_messages(
        [HumanMessage(content="第一行\n第二行", additional_kwargs={"at": "2026-10-25T10:30:00+08:00"})]
    )
    assert rendered[0]["content"] == "[10-25 10:30] 第一行\n第二行"


# ---------------------------------------------------------------------------
# 摘要带日期（§2.4）
# ---------------------------------------------------------------------------
def test_summarizer_adds_date_prefix_only_when_at_present():
    messages = [
        HumanMessage(content="早", additional_kwargs={"at": "2026-10-25T10:30:00+08:00"}),
        AIMessage(content="嗯"),
        HumanMessage(content="旧消息"),
    ]
    text = _format_conversation(messages, "此前")
    lines = text.split("\n")
    assert lines[0] == "【此前摘要】此前"
    assert "用户：[2026-10-25 10:30] 早" in lines
    assert "角色：嗯" in lines
    assert "用户：旧消息" in lines


def test_summarize_prompt_requires_time_retention():
    assert "对话里消息前的 [日期 时间]（若有）是它发生的时间" in SUMMARIZE_PROMPT
    assert "必须把时间一起保留" in SUMMARIZE_PROMPT


# ---------------------------------------------------------------------------
# 冻结时钟（§2.5）
# ---------------------------------------------------------------------------
def test_parse_fixed_now_variants():
    assert clock.parse_fixed_now("2026-10-25T10:35:00+08:00") == NOW
    assert clock.parse_fixed_now("2026-10-25T02:35:00Z") == NOW
    assert clock.parse_fixed_now("2026-10-25T10:35:00") == NOW  # 无偏移按 Asia/Shanghai
    assert clock.parse_fixed_now("") is None
    assert clock.parse_fixed_now(None) is None


def test_parse_fixed_now_rejects_invalid():
    with pytest.raises(ValueError, match="EVAL_FIXED_NOW"):
        clock.parse_fixed_now("不是时间")


def test_clock_uses_fixed_env(monkeypatch):
    monkeypatch.setenv(clock.ENV_KEY, "2026-10-25T10:35:00+08:00")
    assert clock.now() == NOW
    assert clock.now_iso() == "2026-10-25T10:35:00+08:00"


def test_clock_falls_back_to_real_time(monkeypatch):
    monkeypatch.delenv(clock.ENV_KEY, raising=False)
    monkeypatch.setattr(clock, "_settings_fixed_now", lambda: None)
    assert clock.fixed_now_raw() is None
    before = datetime.now(TZ)
    value = clock.now()
    after = datetime.now(TZ)
    assert before <= value <= after


def test_validate_fixed_now_rejects_invalid_env(monkeypatch):
    monkeypatch.setenv(clock.ENV_KEY, "2026-13-40T99:99:99+08:00")
    with pytest.raises(ValueError):
        clock.validate_fixed_now()


async def test_same_now_same_prompt_and_different_now_differs():
    state = {"system_prompt": "人设", "messages": [HumanMessage(content="你好")]}
    first = (await compose_prompt(state, now=NOW))["prompt"]
    second = (await compose_prompt(state, now=NOW))["prompt"]
    assert first == second
    other = (await compose_prompt(state, now=NOW + timedelta(minutes=1)))["prompt"]
    assert other != first
    assert "2026-10-25 10:36" in other[0]["content"]


# ---------------------------------------------------------------------------
# 服务层接线（§2.6）
# ---------------------------------------------------------------------------
class _Store:
    async def delete(self, agent_id, content: str) -> None:  # pragma: no cover - 未触发
        pass


class _Scalar:
    def __init__(self, value) -> None:
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeDB:
    def __init__(self, value) -> None:
        self._value = value

    async def execute(self, _stmt):
        return _Scalar(self._value)


class _FakeFactory:
    """最小 async_sessionmaker 替身：`execute` 返回固定标量。"""

    def __init__(self, value) -> None:
        self._value = value

    def __call__(self):
        value = self._value

        class _Ctx:
            async def __aenter__(self):
                return _FakeDB(value)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


class _BrokenFactory:
    def __call__(self):
        class _Ctx:
            async def __aenter__(self):
                raise RuntimeError("db down")

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _chat_service(factory) -> ChatService:
    return ChatService(graph=None, session_factory=factory, memory_store=_Store())


def _context() -> ChatContext:
    return ChatContext(
        session_id=1,
        agent_id=2,
        agent_name="四季夏目",
        persona_prompt="茶艺师",
        background="古镇",
    )


def test_chat_state_in_carries_at_and_last_user_at(monkeypatch):
    monkeypatch.setenv(clock.ENV_KEY, "2026-10-25T10:35:00+08:00")
    state = ChatService._state_in(
        _context(), "在吗", message_id="m1", last_user_at="2026-10-25T10:25:00+08:00"
    )
    message = state["messages"][0]
    assert message.id == "m1"
    assert message.additional_kwargs["at"] == "2026-10-25T10:35:00+08:00"
    assert state["last_user_at"] == "2026-10-25T10:25:00+08:00"


def test_chat_idle_state_in_carries_last_user_at_only(monkeypatch):
    monkeypatch.setenv(clock.ENV_KEY, "2026-10-25T10:35:00+08:00")
    state = ChatService._idle_state_in(_context(), last_user_at="2026-10-25T07:35:00+08:00")
    assert state["messages"] == []
    assert state["last_user_at"] == "2026-10-25T07:35:00+08:00"


async def test_chat_last_user_at_query_formats_iso_and_is_fail_soft():
    naive = datetime(2026, 10, 25, 10, 30)  # 无时区 → 按 Asia/Shanghai
    assert (
        await _chat_service(_FakeFactory(naive))._last_user_at(1)
        == "2026-10-25T10:30:00+08:00"
    )
    assert await _chat_service(_FakeFactory(None))._last_user_at(1) is None
    # 查询异常不阻断本轮
    assert await _chat_service(_BrokenFactory())._last_user_at(1) is None


def _supervisor() -> tuple[Supervisor, GroupMember]:
    member = GroupMember(agent_id=1, name="甲", persona_prompt="人设", background="")
    return Supervisor(llm=None, graph_factory=lambda: None, members=[member]), member


def test_supervisor_state_in_and_idle_carry_time(monkeypatch):
    monkeypatch.setenv(clock.ENV_KEY, "2026-10-25T10:35:00+08:00")
    supervisor, member = _supervisor()
    state = supervisor._state_in(
        member, 7, "在吗", [], last_user_at="2026-10-25T10:25:00+08:00"
    )
    assert state["messages"][0].additional_kwargs["at"] == "2026-10-25T10:35:00+08:00"
    assert state["last_user_at"] == "2026-10-25T10:25:00+08:00"

    idle = supervisor._idle_state_in(
        member, 7, [], 2, last_user_at="2026-10-25T10:25:00+08:00"
    )
    assert idle["messages"] == []
    assert idle["last_user_at"] == "2026-10-25T10:25:00+08:00"


async def test_group_last_user_at_query_formats_iso_and_is_fail_soft():
    aware = datetime(2026, 10, 24, 18, 30, tzinfo=timezone.utc)  # → +08:00 次日 02:30
    service = GroupService(
        session_factory=_FakeFactory(aware),
        memory_store=_Store(),
        llm=None,
        graph_factory=lambda: None,
        checkpointer=None,
    )
    assert await service._last_user_at(7) == "2026-10-25T02:30:00+08:00"

    broken = GroupService(
        session_factory=_BrokenFactory(),
        memory_store=_Store(),
        llm=None,
        graph_factory=lambda: None,
        checkpointer=None,
    )
    assert await broken._last_user_at(7) is None
