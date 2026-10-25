"""图/state/generate 工具循环单测（契约 §2.4、§1.3 不变量 1/2/3）。"""

from datetime import datetime

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from app.agents.graph import build_graph, should_continue
from app.agents.nodes.compose_prompt import TIME_HEADER, TOOLS_HEADER, TZ, compose_prompt
from app.agents.nodes.generate import (
    _CLOCK_RE,
    _explicit_tool_intent,
    _retry_for_thinkings,
    generate,
    invalid_argument_streak,
)
from app.agents.nodes.tools import execute_tools
from app.llm.base import LLMReply
from app.memory.store import MemoryStore
from app.prompts.roleplay_system import MESSAGE_DELIMITER as MSG
from app.prompts.roleplay_system import THINK_DELIMITER as T
from app.skills.base import BaseSkill, ToolCall
from app.skills.builtin.time import get_current_time_skill
from app.skills.executor import ToolExecutor
from app.skills.registry import SkillRegistry

_DIMS = 1024


class _Emb(Embeddings):
    def embed_documents(self, texts):
        return [[0.1] * _DIMS for _ in texts]

    def embed_query(self, text):
        return [0.1] * _DIMS


def _store() -> InMemoryStore:
    return InMemoryStore(index={"dims": _DIMS, "embed": _Emb(), "fields": ["text"]})


def _registry() -> SkillRegistry:
    return SkillRegistry([get_current_time_skill])


class ToolLLM:
    """脚本化工具 LLM：按脚本依次返回 LLMReply；记录 tool_choice。"""

    def __init__(self, script, *, stream_reply="流式兜底文本", generate_reply="非流式兜底文本"):
        self.script = list(script)
        self.tool_choices: list[str | dict] = []
        self.tools_seen: list[list[dict]] = []
        self.messages_seen: list[list[dict]] = []
        self.generate_calls = 0
        self._stream_reply = stream_reply
        self._generate_reply = generate_reply

    async def generate(self, messages):
        self.generate_calls += 1
        return self._generate_reply

    async def astream(self, messages):
        yield self._stream_reply

    async def generate_with_tools(self, messages, tools, *, tool_choice="auto", temperature=0.0):
        self.tool_choices.append(tool_choice)
        self.tools_seen.append(tools)
        self.messages_seen.append(messages)
        return self.script.pop(0)


def _state(**over) -> dict:
    base = {
        "messages": [HumanMessage(content="现在东京几点")],
        "agent_id": 1,
        "session_id": 1,
        "agent_name": "四季夏目",
        "persona_prompt": "茶艺师。",
        "background": "古镇",
        "prompt": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "现在东京几点"},
        ],
    }
    base.update(over)
    return base


# ---- 不变量 1：空白名单 prompt 逐字一致 ----

_FIXED_NOW = datetime(2026, 10, 24, 22, 31, tzinfo=TZ)
_TIME_BLOCK = (
    f"{TIME_HEADER}\n"
    "现在是 2026-10-24 22:31（Asia/Shanghai，周六）。\n"
    "用户说「今晚8点35」「明天早上7点」这类相对时间时，先按此换算成绝对时间。"
)


async def test_compose_prompt_empty_whitelist_byte_identical():
    state = _state()
    no_tools = await compose_prompt(state, now=_FIXED_NOW)
    with_empty_registry = await compose_prompt(
        state, skill_registry=_registry(), allowed_skills=[], now=_FIXED_NOW
    )
    assert no_tools == with_empty_registry
    # V3 M1：system=时间块（空人设），其后为近轮消息，无【可用工具】段
    assert no_tools == {
        "prompt": [
            {"role": "system", "content": _TIME_BLOCK},
            {"role": "user", "content": "现在东京几点"},
        ]
    }


async def test_compose_prompt_adds_tools_section():
    state = _state(allowed_skills=["get_current_time"])
    out = await compose_prompt(state, skill_registry=_registry())
    assert TOOLS_HEADER in out["prompt"][0]["content"]
    assert "get_current_time：需要知道当前日期/时间时使用" in out["prompt"][0]["content"]
    # 段内只放 name + when_to_use + 一条作答规则，不得塞 JSON Schema
    assert "parameters" not in out["prompt"][0]["content"]
    assert "不得声称已完成" in out["prompt"][0]["content"]


async def test_current_time_block_is_injectable():
    """时间锚点可注入：固定 now 后段文本逐字可断言（不再依赖 datetime.now）。"""
    state = _state(allowed_skills=["get_current_time"])
    out = await compose_prompt(state, skill_registry=_registry(), now=_FIXED_NOW)

    expected = (
        f"{TIME_HEADER}\n"
        "现在是 2026-10-24 22:31（Asia/Shanghai，周六）。\n"
        "用户说「今晚8点35」「明天早上7点」这类相对时间时，先按此换算成绝对时间。"
    )
    assert expected in out["prompt"][0]["content"]


async def test_no_tools_still_has_current_time_block():
    """V3 M1：无工具角色也必须有时间块（不再依赖 tools_block 非空）。"""
    state = _state(allowed_skills=[])
    out = await compose_prompt(
        state,
        skill_registry=_registry(),
        now=_FIXED_NOW,
    )
    assert out["prompt"][0]["content"] == _TIME_BLOCK


# ---- 不变量 2：skill_registry=None 图拓扑不变 ----

def test_graph_topology_without_registry_has_no_tools_node():
    graph = build_graph(
        llm=ToolLLM([]), memory_store=MemoryStore(_store()), checkpointer=InMemorySaver()
    )
    assert "tools" not in set(graph.get_graph().nodes)


def test_graph_topology_with_registry_has_tools_node():
    graph = build_graph(
        llm=ToolLLM([]),
        memory_store=MemoryStore(_store()),
        checkpointer=InMemorySaver(),
        skill_registry=_registry(),
    )
    assert "tools" in set(graph.get_graph().nodes)


# ---- should_continue ----

def test_should_continue_routing():
    assert should_continue({"tool_calls": [{"id": "1"}], "tool_rounds": 0}, max_tool_rounds=3) == "tools"
    assert should_continue({"tool_calls": [], "tool_rounds": 0}, max_tool_rounds=3) == "update_memory"
    assert should_continue({"tool_calls": [{"id": "1"}], "tool_rounds": 3}, max_tool_rounds=3) == "update_memory"


# ---- generate：tool_calls 分支不写空 AIMessage ----

async def test_generate_returns_tool_calls_without_ai_message():
    llm = ToolLLM([LLMReply(tool_calls=[ToolCall("c1", "get_current_time", {"timezone": "Asia/Tokyo"})])])
    state = _state(allowed_skills=["get_current_time"])
    out = await generate(state, llm, skill_registry=_registry())
    assert out["tool_calls"] == [{"id": "c1", "name": "get_current_time", "arguments": {"timezone": "Asia/Tokyo"}}]
    assert out["messages"] == [] and out["replies"] == [] and out["reply"] == ""
    assert out["tool_rounds"] == 0
    assert llm.tool_choices == ["auto"]
    assert llm.tools_seen[0][0]["function"]["name"] == "get_current_time"


async def test_generate_text_branch_parses_and_emits_once():
    llm = ToolLLM([LLMReply(content="现在是晚上八点。")])
    state = _state(allowed_skills=["get_current_time"])
    out = await generate(state, llm, skill_registry=_registry())
    assert out["replies"] == ["现在是晚上八点。"]
    assert len(out["messages"]) == 1


# ---- 循环保护：达上限强制 tool_choice=none 并丢弃 tool_calls ----

async def test_generate_forced_none_at_round_limit():
    llm = ToolLLM(
        [LLMReply(tool_calls=[ToolCall("c1", "get_current_time", {})])],
        generate_reply="已经到点了。",
    )
    state = _state(allowed_skills=["get_current_time"], tool_rounds=3, tool_results=[])
    out = await generate(state, llm, skill_registry=_registry(), max_tool_rounds=3)
    assert llm.tool_choices == ["none"]
    assert out["reply"] == "已经到点了。"
    assert out["tool_results"][0]["error_type"] == "loop_detected"


def test_invalid_argument_streak():
    assert invalid_argument_streak([]) == 0
    assert invalid_argument_streak([{"name": "a", "error_type": "invalid_arguments"}]) == 1
    assert (
        invalid_argument_streak(
            [
                {"name": "a", "error_type": "invalid_arguments"},
                {"name": "a", "error_type": "invalid_arguments"},
            ]
        )
        == 2
    )
    assert (
        invalid_argument_streak(
            [
                {"name": "a", "error_type": "invalid_arguments"},
                {"name": "b", "error_type": "invalid_arguments"},
            ]
        )
        == 1
    )


# ---- provider 不支持 tools → 降级 + tools_unavailable ----

async def test_generate_falls_back_when_tools_unsupported():
    class NoToolsLLM(ToolLLM):
        async def generate_with_tools(self, *a, **kw):
            raise NotImplementedError

    llm = NoToolsLLM([])
    state = _state(allowed_skills=["get_current_time"])
    out = await generate(state, llm, skill_registry=_registry())
    assert out["tools_unavailable"] is True
    assert out["replies"] == ["流式兜底文本"]


# ---- tools 节点 ----

async def test_execute_tools_records_and_clears():
    out = await execute_tools(
        _state(allowed_skills=["get_current_time"], tool_calls=[{"id": "c1", "name": "get_current_time", "arguments": {}}]),
        ToolExecutor(_registry()),
    )
    assert out["tool_calls"] == [] and out["tool_rounds"] == 1
    record = out["tool_results"][0]
    assert record["name"] == "get_current_time" and record["ok"] is True
    assert record["tool_round"]


async def test_execute_tools_continues_after_failure():
    state = _state(
        allowed_skills=["get_current_time", "dice"],
        tool_calls=[
            {"id": "c1", "name": "unknown", "arguments": {}},
            {"id": "c2", "name": "get_current_time", "arguments": {}},
        ],
    )
    out = await execute_tools(state, ToolExecutor(_registry()))
    assert [r["error_type"] for r in out["tool_results"]] == ["unknown_tool", None]


# ---- 端到端：generate → tools → generate ----

async def test_graph_runs_tool_loop_end_to_end():
    llm = ToolLLM(
        [
            LLMReply(tool_calls=[ToolCall("c1", "get_current_time", {"timezone": "Asia/Tokyo"})]),
            LLMReply(content="东京现在是晚上。"),
        ]
    )
    store = _store()
    graph = build_graph(
        llm=llm,
        memory_store=MemoryStore(store),
        checkpointer=InMemorySaver(),
        store=store,
        skill_registry=_registry(),
    )
    result = await graph.ainvoke(
        _state(allowed_skills=["get_current_time"]),
        config={"configurable": {"thread_id": "tools:1"}},
    )
    assert result["reply"] == "东京现在是晚上。"
    assert result["tool_results"][0]["name"] == "get_current_time"
    assert result["tool_rounds"] == 1
    contents = [m.content for m in result["messages"]]
    # 只有用户消息与最终 AI 回复；工具往返消息不进 state.messages
    assert contents == ["现在东京几点", "东京现在是晚上。"]
    # 第二轮 generate 的本地消息列表里含 assistant(tool_calls) 与 tool 结果
    roles = [m["role"] for m in llm.messages_seen[-1]]
    assert "tool" in roles


# ---- 声称完成却没调用工具 → 补调一次（提醒抽屉为空的产品事故兜底）----


def _reminder_registry() -> SkillRegistry:
    class _Args(BaseModel):
        when: str
        content: str

    async def handler(ctx, args):  # pragma: no cover - 只测补调，不执行
        raise AssertionError("不应被调用")

    return SkillRegistry(
        [
            BaseSkill(
                name="set_reminder",
                description="创建提醒",
                when_to_use="用户要求在某时间提醒",
                parameters=_Args,
                handler=handler,
            )
        ]
    )


async def test_声称设好却没调用工具时补调成工具轮():
    llm = ToolLLM(
        [
            LLMReply(content="七点五十六的提醒，我这边设好了，你确认一下。", tool_calls=[]),
            LLMReply(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="set_reminder",
                        arguments={"when": "2026-10-25T07:56:00+08:00", "content": "测试"},
                    )
                ],
            ),
        ]
    )
    state = _state(
        messages=[HumanMessage(content="7点56提醒我")],
        prompt=[{"role": "system", "content": ""}, {"role": "user", "content": "7点56提醒我"}],
        allowed_skills=["set_reminder"],
    )

    out = await generate(state, llm, skill_registry=_reminder_registry())

    assert [c["name"] for c in out["tool_calls"]] == ["set_reminder"]
    assert out["reply"] == ""
    assert len(llm.messages_seen) == 2  # 首轮 + 补调
    assert "没有调用 set_reminder" in llm.messages_seen[-1][-1]["content"]
    # 补调必须锁死工具（指定函数），否则模型仍可能只回客套话
    assert llm.tool_choices[-1] == {
        "type": "function",
        "function": {"name": "set_reminder"},
    }


async def test_没声称完成但用户给了钟点也补调():
    """用户「今晚8点35提醒我喝水」+ 模型只回文本（不说“设好了”）→ 仍要补调。

    这正是线上复现的失败形态：模型回“今晚20点35，提醒你喝水。你确认一下。”，
    不含“已设置/设好”等措辞，旧守卫因此放过，提醒抽屉一直是空的。
    """
    llm = ToolLLM(
        [
            LLMReply(content="今晚20点35，提醒你喝水。你确认一下。", tool_calls=[]),
            LLMReply(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="set_reminder",
                        arguments={"when": "2026-10-24 20:35", "content": "喝水"},
                    )
                ],
            ),
        ]
    )
    state = _state(
        messages=[HumanMessage(content="今晚8点35提醒我喝水")],
        prompt=[
            {"role": "system", "content": ""},
            {"role": "user", "content": "今晚8点35提醒我喝水"},
        ],
        allowed_skills=["set_reminder"],
    )

    out = await generate(state, llm, skill_registry=_reminder_registry())

    assert [c["name"] for c in out["tool_calls"]] == ["set_reminder"]
    assert len(llm.messages_seen) == 2


async def test_中文钟点也补调():
    """「今晚十一点提醒我喝水」用中文数字，旧正则只认阿拉伯数字会漏判。"""
    llm = ToolLLM(
        [
            LLMReply(content="今晚23点，提醒你喝水。你确认一下。", tool_calls=[]),
            LLMReply(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="set_reminder",
                        arguments={"when": "2026-10-24 23:00", "content": "喝水"},
                    )
                ],
            ),
        ]
    )
    state = _state(
        messages=[HumanMessage(content="今晚十一点提醒我喝水")],
        prompt=[
            {"role": "system", "content": ""},
            {"role": "user", "content": "今晚十一点提醒我喝水"},
        ],
        allowed_skills=["set_reminder"],
    )

    out = await generate(state, llm, skill_registry=_reminder_registry())

    assert [c["name"] for c in out["tool_calls"]] == ["set_reminder"]
    assert len(llm.messages_seen) == 2


async def test_缺少内心独白时补写一次():
    """工具模式下模型只回正文（无 THINK 段）→ 补写一次内心独白（正文保持不变）。"""
    llm = ToolLLM(
        [LLMReply(content="现在是晚上八点。", tool_calls=[])],
        generate_reply=f"现在是晚上八点。{T}心里有点急",
    )
    state = _state(allowed_skills=["get_current_time"])

    out = await generate(state, llm, skill_registry=_registry())

    assert out["replies"] == ["现在是晚上八点。"]
    assert out["thinkings"] == ["心里有点急"]
    assert len(llm.messages_seen) == 1  # 补写走 generate，不重复走工具调用


async def test_已有内心独白时不补写():
    """首轮已带独白 → 不再多发一次补写请求。"""
    llm = ToolLLM(
        [LLMReply(content=f"现在是晚上八点。{T}早就看过了", tool_calls=[])],
        generate_reply=f"不该用到{T}不该出现的独白",
    )
    state = _state(allowed_skills=["get_current_time"])

    out = await generate(state, llm, skill_registry=_registry())

    assert out["thinkings"] == ["早就看过了"]


async def test_补写条数不一致时保留原文只取独白():
    """补写返回多条时不该改写正文，只按顺序取独白。"""
    llm = ToolLLM(
        [LLMReply(content="现在是晚上八点。", tool_calls=[])],
        generate_reply=f"甲{T}想甲{MSG}乙{T}想乙",
    )
    state = _state(allowed_skills=["get_current_time"])

    out = await generate(state, llm, skill_registry=_registry())

    assert out["replies"] == ["现在是晚上八点。"]
    assert out["thinkings"] == ["想甲"]


async def test_没有钟点时不补调():
    """「提醒我喝水」没给具体时间 → 不应强行调工具（模型可以先追问几点）。"""
    llm = ToolLLM([LLMReply(content="好啊，你想几点喝？", tool_calls=[])])
    state = _state(
        messages=[HumanMessage(content="提醒我喝水")],
        prompt=[{"role": "system", "content": ""}, {"role": "user", "content": "提醒我喝水"}],
        allowed_skills=["set_reminder"],
    )

    out = await generate(state, llm, skill_registry=_reminder_registry())

    assert out["replies"] == ["好啊，你想几点喝？"]
    assert len(llm.messages_seen) == 1


async def test_本轮已有工具结果时不补调():
    llm = ToolLLM([LLMReply(content="设好了，你确认一下。", tool_calls=[])])
    state = _state(
        messages=[HumanMessage(content="7点56提醒我")],
        prompt=[{"role": "system", "content": ""}, {"role": "user", "content": "7点56提醒我"}],
        allowed_skills=["set_reminder"],
        tool_results=[{"name": "set_reminder", "ok": True, "tool_round": 1}],
        tool_rounds=1,
    )

    out = await generate(state, llm, skill_registry=_reminder_registry())

    assert out["replies"] == ["设好了，你确认一下。"]
    assert len(llm.messages_seen) == 1


# ---- V2.6：钟点正则收紧（正例 / 反例各 3 条）----


@pytest.mark.parametrize(
    "text",
    ["提醒我今晚8点35", "明天早上七点叫我", "21:30 提醒我吃药"],
)
def test_钟点正则_正例(text):
    assert _CLOCK_RE.search(text) is not None


@pytest.mark.parametrize(
    "text",
    ["提醒我一点小事", "还有一点没做完", "提醒我下周三"],
)
def test_钟点正则_反例(text):
    assert _CLOCK_RE.search(text) is None


def test_一点小事不判成建提醒():
    """误报回归：含「提醒」但无明确钟点 → 不强制 set_reminder。"""
    messages = [{"role": "user", "content": "提醒我一点小事"}]
    assert _explicit_tool_intent(messages, ["set_reminder"]) is None


# ---- V2.6：内心独白补写开关 ----


async def test_关闭补写开关时不调用LLM():
    """REPLY_THINK_RETRY_ENABLED=false：整轮缺独白也不额外调用，返回原样。"""
    llm = ToolLLM(
        [LLMReply(content="现在是晚上八点。", tool_calls=[])],
        generate_reply=f"甲{T}乙",
    )
    state = _state(allowed_skills=["get_current_time"])

    replies, thinkings = await _retry_for_thinkings(
        state,
        llm,
        ["现在是晚上八点。"],
        [""],
        max_messages=3,
        enabled=False,
    )

    assert llm.generate_calls == 0
    assert replies == ["现在是晚上八点。"]
    assert thinkings == [""]
