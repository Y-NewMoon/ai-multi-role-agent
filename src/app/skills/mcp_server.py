"""MCP stdio server（契约 §2.9）：把 5 个内置工具暴露给 MCP 客户端。

启动（仓库根目录，`src` 需在 `PYTHONPATH`）：
    PYTHONPATH=src python -m app.skills.mcp_server

上下文（D6）：stdio 无请求上下文，用环境变量显式注入：
    MCP_AGENT_ID / MCP_SESSION_ID —— 缺省时 `read_lore` / `set_reminder` 返回
    `context_missing`，其余工具照常可用。

`list_tools` 返回 5 个工具的 OpenAI 兼容 schema；`call_tool` 返回 `ToolResult` 的 JSON。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

# 允许直接以文件路径执行（`python src/app/skills/mcp_server.py`）时也能 import `app`
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mcp.types as types  # noqa: E402
from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402

from app.skills.base import ToolCall, ToolContext, ToolResult  # noqa: E402
from app.skills.builtin import build_default_registry  # noqa: E402
from app.skills.executor import ToolExecutor  # noqa: E402
from app.skills.registry import SkillRegistry  # noqa: E402

# 需要 agent/session 上下文的有状态工具（缺上下文 → context_missing，D6）
CONTEXT_REQUIRED = frozenset({"read_lore", "set_reminder"})

server = Server("ai-agent-skills")
_REGISTRY: SkillRegistry | None = None
_EXECUTOR: ToolExecutor | None = None


def _build_registry() -> SkillRegistry:
    """组装 5 个工具；依赖（DB / 搜索后端）不可用时降级（工具各自返回 unavailable）。

    R4（V2.2 已知边界）：MCP 侧不接 `Retriever`，故 `read_lore` 即使上下文齐全也返回
    `unavailable`；V2.2 的 MCP 只保证无状态工具（time/dice/web_search/reminder）可用，
    `read_lore` 仅在本服务（FastAPI 进程）内可用。见交接文档 §5 / §7。
    """
    session_factory = None
    provider = None
    try:  # 无 .env / 无 DATABASE_URL 时仍要能启动（否则 smoke 会因环境缺失而失败）
        from app.config import get_settings
        from app.db.base import create_engine, create_session_factory
        from app.skills.builtin.web_search import HttpSearchProvider

        settings = get_settings()
        session_factory = create_session_factory(create_engine(settings.sqlalchemy_url))
        provider = HttpSearchProvider(
            url=settings.web_search_url, api_key=settings.web_search_api_key
        )
    except Exception:  # noqa: BLE001 - 环境缺失只降级，不阻断 stdio 启动
        pass
    return build_default_registry(
        search_provider=provider, session_factory=session_factory, lore_retriever=None
    )


def _registry() -> SkillRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def _executor() -> ToolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ToolExecutor(_registry())
    return _EXECUTOR


def _context() -> tuple[int | None, int | None]:
    """从环境读入 agent/session 上下文（缺省或非法 → None）。"""

    def _int(name: str) -> int | None:
        raw = os.environ.get(name, "").strip()
        try:
            return int(raw)
        except ValueError:
            return None

    return _int("MCP_AGENT_ID"), _int("MCP_SESSION_ID")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """5 个工具，按 OpenAI 兼容 schema（`function.parameters` → `inputSchema`）。"""
    registry = _registry()
    return [
        types.Tool(
            name=skill.name,
            description=skill.description,
            inputSchema=skill.to_openai_tool()["function"]["parameters"],
        )
        for skill in registry.resolve(registry.names(), only_available=True)
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict | None = None) -> list[types.TextContent]:
    """执行一次工具调用，返回 `ToolResult` 的 JSON 文本（失败也不抛异常）。"""
    agent_id, session_id = _context()
    if name in CONTEXT_REQUIRED and (agent_id is None or session_id is None):
        result = ToolResult(
            ok=False,
            error_type="context_missing",
            message="缺少 MCP_AGENT_ID / MCP_SESSION_ID 上下文",
        )
    else:
        registry = _registry()
        ctx = ToolContext(
            agent_id=agent_id or 0,
            session_id=session_id or 0,
            allowed_skills=tuple(registry.names()),
        )
        executor = _executor()
        executor.reset()  # 每次 call_tool 视为一次请求（同请求缓存，§2.2-5）
        result = await executor.execute(
            ToolCall(id=uuid.uuid4().hex, name=name, arguments=arguments or {}), ctx
        )
    payload = asdict(result)
    return [types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, default=str))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
