"""B-01 工具可用性过滤（V2.4 §2.6）：Skill.available / registry / compose_prompt / web_search。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.agents.nodes.compose_prompt import _tools_block
from app.skills.base import BaseSkill
from app.skills.builtin import build_default_registry
from app.skills.builtin.web_search import FakeSearchProvider, make_web_search_skill
from app.skills.registry import SkillRegistry


class _Args(BaseModel):
    value: str = "x"


def _skill(name: str, *, available: bool | None = None) -> BaseSkill:
    async def handler(ctx, args):  # pragma: no cover - 只测过滤，不执行
        raise AssertionError("不应被调用")

    probe = None if available is None else (lambda: available)
    return BaseSkill(
        name=name,
        description="d",
        when_to_use="w",
        parameters=_Args,
        handler=handler,
        available=probe,
    )


# ---- BaseSkill ----

def test_默认可用():
    assert _skill("alpha").is_available() is True


def test_显式不可用():
    assert _skill("alpha", available=False).is_available() is False
    assert _skill("alpha", available=True).is_available() is True


# ---- 注册表 ----

def test_resolve默认保持v22语义():
    reg = SkillRegistry([_skill("alpha", available=False)])
    assert [s.name for s in reg.resolve(["alpha"])] == ["alpha"]


def test_resolve_only_available过滤():
    reg = SkillRegistry([_skill("alpha", available=False), _skill("beta")])
    assert [s.name for s in reg.resolve(["alpha", "beta"], only_available=True)] == ["beta"]


def test_to_openai_tools默认只下发可用():
    reg = SkillRegistry([_skill("alpha", available=False), _skill("beta")])
    assert [t["function"]["name"] for t in reg.to_openai_tools(["alpha", "beta"])] == ["beta"]
    # 显式关闭过滤时保持 V2.2 输出
    assert [t["function"]["name"] for t in reg.to_openai_tools(
        ["alpha", "beta"], only_available=False
    )] == ["alpha", "beta"]


class _BareSkill:
    """没有 is_available 的自定义 Skill：registry 应视为可用（不强制改 Protocol）。"""

    name = "bare"
    description = "d"
    when_to_use = "w"

    def to_openai_tool(self) -> dict:
        return {"type": "function", "function": {"name": self.name}}


def test_无is_available的自定义skill视为可用():
    reg = SkillRegistry([_BareSkill()])
    assert [s.name for s in reg.resolve(["bare"], only_available=True)] == ["bare"]
    assert [t["function"]["name"] for t in reg.to_openai_tools(["bare"])] == ["bare"]


# ---- web_search ----

def test_web_search无后端不可用():
    assert make_web_search_skill(None).is_available() is False


def test_web_search假后端按available标记():
    assert make_web_search_skill(FakeSearchProvider()).is_available() is True
    assert make_web_search_skill(FakeSearchProvider(available=False)).is_available() is False


def test_无key时to_openai_tools不含web_search():
    reg = build_default_registry()  # search_provider=None
    names = [t["function"]["name"] for t in reg.to_openai_tools(reg.names())]
    assert "web_search" not in names
    assert "get_current_time" in names


def test_有可用后端时web_search仍下发():
    reg = build_default_registry(search_provider=FakeSearchProvider())
    names = [t["function"]["name"] for t in reg.to_openai_tools(["web_search"])]
    assert names == ["web_search"]


# ---- 【可用工具】段 ----

def test_可用工具段过滤不可用工具():
    reg = build_default_registry()  # 无搜索后端
    block = _tools_block(reg, ["get_current_time", "web_search"])
    assert "get_current_time" in block
    assert "web_search" not in block


def test_全部不可用时工具段为空():
    reg = build_default_registry()
    assert _tools_block(reg, ["web_search"]) == ""


# ---- MCP ----

async def test_mcp_list_tools不暴露不可用工具(monkeypatch):
    pytest.importorskip("mcp")
    from app.skills import mcp_server

    monkeypatch.setattr(mcp_server, "_REGISTRY", build_default_registry())
    names = [tool.name for tool in await mcp_server.list_tools()]
    assert "web_search" not in names
    assert "get_current_time" in names
