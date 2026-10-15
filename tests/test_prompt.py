"""prompt 组装单测：角色映射与 system prompt 注入。"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.nodes.compose_prompt import _to_chat_messages, compose_prompt
from app.prompts.roleplay_system import build_system_prompt


def test_build_system_prompt_contains_persona_fields():
    sp = build_system_prompt(name="四季夏目", persona="茶艺师", background="江南古镇")
    assert "四季夏目" in sp
    assert "茶艺师" in sp
    assert "江南古镇" in sp


def test_build_system_prompt_falls_back_on_empty():
    sp = build_system_prompt(name="", persona="")
    assert "（未命名角色）" in sp
    assert "（未提供人设）" in sp


def test_role_mapping_langchain_to_openai():
    msgs = [
        SystemMessage(content="s"),
        HumanMessage(content="h"),
        AIMessage(content="a"),
    ]
    assert [m["role"] for m in _to_chat_messages(msgs)] == ["system", "user", "assistant"]


async def test_compose_prompt_injects_memories_and_keeps_messages():
    state = {
        "system_prompt": "SYSTEM",
        "recalled_memories": ["记得A", "记得B"],
        "messages": [HumanMessage(content="你好")],
    }
    prompt = (await compose_prompt(state))["prompt"]

    assert prompt[0]["role"] == "system"
    assert "SYSTEM" in prompt[0]["content"]
    assert "记得A" in prompt[0]["content"]
    assert "记得B" in prompt[0]["content"]
    assert prompt[-1] == {"role": "user", "content": "你好"}


async def test_compose_prompt_without_memories_has_no_memory_block():
    state = {
        "system_prompt": "SYSTEM",
        "recalled_memories": [],
        "messages": [HumanMessage(content="你好")],
    }
    prompt = (await compose_prompt(state))["prompt"]
    assert "你记得的相关信息" not in prompt[0]["content"]


async def test_compose_prompt_includes_summary_when_under_budget():
    state = {
        "system_prompt": "SYSTEM",
        "summary": "他叫阿飞",
        "recalled_memories": [],
        "messages": [HumanMessage(content="你好")],
    }
    prompt = (await compose_prompt(state))["prompt"]
    assert "本会话摘要" in prompt[0]["content"]
    assert "他叫阿飞" in prompt[0]["content"]


async def test_compose_prompt_trims_low_priority_when_over_budget():
    # 方案 3.9：预算不足时先丢摘要、再丢召回记忆；persona 必留、最新一轮必留
    state = {
        "system_prompt": "SYSTEM",
        "summary": "很长很长的摘要" * 50,
        "recalled_memories": ["记忆A"],
        "messages": [HumanMessage(content="最近一句")],
    }
    prompt = (await compose_prompt(state, token_budget=1))["prompt"]
    assert "SYSTEM" in prompt[0]["content"]
    assert "本会话摘要" not in prompt[0]["content"]
    assert "你记得的相关信息" not in prompt[0]["content"]
    assert prompt[-1] == {"role": "user", "content": "最近一句"}

