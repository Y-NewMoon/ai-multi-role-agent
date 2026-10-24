"""MCP 客户端 smoke（契约 §2.9）。

默认以 stdio 启动**本服务**的 MCP server，断言：
- `list_tools` 有 5 个工具；
- 调用 `get_current_time` 成功（`ok=True`）；
- 退出码 0。

未安装 `mcp` → 打印 skip 并退出 0（PR CI 不启动 MCP，§4.3）。
`--external CMD [ARGS...]` 可选：改为连接外部 server，仅演示（不校验数量）。

用法：
    uv run python scripts/mcp_smoke.py
    uv run python scripts/mcp_smoke.py --external npx -y @modelcontextprotocol/server-filesystem .
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP stdio smoke")
    parser.add_argument(
        "--external",
        nargs=argparse.REMAINDER,
        default=None,
        help="外部 MCP server 命令（可选，仅演示）",
    )
    return parser.parse_args(argv)


async def _run(external: list[str] | None) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    if external:
        params = StdioServerParameters(command=external[0], args=external[1:], env=env)
    else:
        # 子进程需要 `src` 在 PYTHONPATH 上才能 `-m app.skills.mcp_server`
        env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "app.skills.mcp_server"], env=env
        )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            names = sorted(t.name for t in tools)
            print(f"list_tools({len(names)}): {names}")
            if not external and len(names) != 5:
                print(f"[FAIL] list_tools 应为 5 个，实际 {len(names)}")
                return 1
            if external:
                print("[ok] 外部 server 已连通（仅演示，不校验工具数量）")
                return 0

            result = await session.call_tool(
                "get_current_time", {"timezone": "Asia/Shanghai"}
            )
            text = "".join(
                c.text for c in result.content if getattr(c, "text", None) is not None
            )
            print(f"call_tool get_current_time: {text}")
            payload = json.loads(text)
            if not payload.get("ok"):
                print(f"[FAIL] get_current_time 调用失败：{payload}")
                return 1

    print("[ok] mcp_smoke 通过（list_tools=5、get_current_time ok）")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("[skip] 未安装 mcp>=1.0，跳过 mcp_smoke（退出码 0，契约 §4.3）")
        return 0
    return asyncio.run(_run(args.external))


if __name__ == "__main__":
    sys.exit(main())
