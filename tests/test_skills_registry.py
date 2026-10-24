"""Skill 注册表与内置工具组装单测（契约 §2.1 / §2.2 注册表接口）。

与 `test_skills_executor.py`（执行器七条硬要求）互补：本文件聚焦**注册表 API 形状**、
**五个内置工具的注册顺序与 schema 形状**、以及工具名/参数模型的契约约束。
"""

import re

import pytest
from pydantic import BaseModel

from app.skills.base import SKILL_NAME_PATTERN, BaseSkill
from app.skills.builtin import build_default_registry
from app.skills.builtin.web_search import FakeSearchProvider
from app.skills.registry import SkillRegistry

EXPECTED_TOOLS = ("get_current_time", "roll_dice", "web_search", "set_reminder", "read_lore")


class _Args(BaseModel):
    value: str


def _skill(name: str) -> BaseSkill:
    async def handler(ctx, args):  # pragma: no cover - 仅形状测试，不调用
        raise AssertionError("不应被调用")

    return BaseSkill(
        name=name, description="d", when_to_use="w", parameters=_Args, handler=handler
    )


# ---- 注册表 API ----

def test_get_returns_skill_or_none():
    a = _skill("alpha")
    reg = SkillRegistry([a])
    assert reg.get("alpha") is a
    assert reg.get("missing") is None


def test_names_is_insertion_ordered():
    reg = SkillRegistry([_skill("bbb"), _skill("aaa")])
    assert reg.names() == ["bbb", "aaa"]  # 不排序：保持注册顺序


def test_resolve_warns_on_unknown(caplog):
    reg = SkillRegistry([_skill("alpha")])
    with caplog.at_level("WARNING"):
        resolved = reg.resolve(["ghost", "alpha"])
    assert [s.name for s in resolved] == ["alpha"]  # 未知忽略、不抛异常
    assert any("ghost" in r.message for r in caplog.records)


def test_to_openai_tools_only_whitelisted():
    reg = SkillRegistry([_skill("alpha"), _skill("beta")])
    tools = reg.to_openai_tools(["beta"])
    assert [t["function"]["name"] for t in tools] == ["beta"]


def test_empty_whitelist_yields_no_tools():
    reg = SkillRegistry([_skill("alpha")])
    assert reg.to_openai_tools([]) == []


# ---- 内置注册表 ----

def test_build_default_registry_has_five_tools_in_order():
    reg = build_default_registry()
    assert reg.names() == list(EXPECTED_TOOLS)


def test_builtin_tool_schemas_are_openai_shaped():
    # V2.4 B-01：web_search 需有可用后端才会下发，这里注入可用的假后端
    reg = build_default_registry(search_provider=FakeSearchProvider())
    tools = reg.to_openai_tools(list(EXPECTED_TOOLS))
    assert len(tools) == 5
    for tool in tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert re.match(SKILL_NAME_PATTERN, fn["name"]), fn["name"]
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert set(params["properties"])  # 至少一个参数


def test_builtin_name_pattern_all_match():
    assert all(re.match(SKILL_NAME_PATTERN, name) for name in EXPECTED_TOOLS)


def test_duplicate_name_raises_at_construction():
    s = _skill("dup")
    with pytest.raises(ValueError):
        SkillRegistry([s, s])
