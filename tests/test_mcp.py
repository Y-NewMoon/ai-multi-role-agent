"""MCP server（V2.2 契约 §2.9）：5 个工具、OpenAI 兼容 schema、context_missing 语义。

未安装 `mcp` 时整体跳过（§4.3：PR CI 不启动 MCP）。
"""

import json

import pytest

pytest.importorskip("mcp")

from app.skills import mcp_server  # noqa: E402
from app.skills.builtin import build_default_registry  # noqa: E402
from app.skills.builtin.web_search import FakeSearchProvider  # noqa: E402

EXPECTED = ["get_current_time", "read_lore", "roll_dice", "set_reminder", "web_search"]


@pytest.fixture(autouse=True)
def _available_registry(monkeypatch):
    """V2.4 B-01：web_search 需要有可用后端才会被 list_tools 暴露。"""
    monkeypatch.setattr(
        mcp_server, "_REGISTRY", build_default_registry(search_provider=FakeSearchProvider())
    )


async def test_list_tools_returns_five_openai_compatible_schemas():
    tools = await mcp_server.list_tools()
    assert sorted(t.name for t in tools) == EXPECTED
    for tool in tools:
        assert tool.description
        assert tool.inputSchema["type"] == "object"
        assert tool.inputSchema["additionalProperties"] is False


async def test_stateful_tool_without_context_returns_context_missing(monkeypatch):
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    monkeypatch.delenv("MCP_SESSION_ID", raising=False)
    payload = json.loads((await mcp_server.call_tool("read_lore", {"query": "茶馆"}))[0].text)
    assert payload["ok"] is False
    assert payload["error_type"] == "context_missing"


async def test_stateful_tool_with_context_reaches_executor(monkeypatch):
    # 有上下文时不再短路为 context_missing：read_lore 无检索器 → unavailable
    monkeypatch.setenv("MCP_AGENT_ID", "1")
    monkeypatch.setenv("MCP_SESSION_ID", "1")
    payload = json.loads((await mcp_server.call_tool("read_lore", {"query": "茶馆"}))[0].text)
    assert payload["ok"] is False
    assert payload["error_type"] == "unavailable"


async def test_stateless_tool_ok_and_unknown_tool(monkeypatch):
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    monkeypatch.delenv("MCP_SESSION_ID", raising=False)
    ok = json.loads((await mcp_server.call_tool("get_current_time", {"timezone": "Asia/Tokyo"}))[0].text)
    assert ok["ok"] is True and ok["data"]["timezone"] == "Asia/Tokyo"

    bad = json.loads((await mcp_server.call_tool("nope", {}))[0].text)
    assert bad["ok"] is False and bad["error_type"] == "unknown_tool"
