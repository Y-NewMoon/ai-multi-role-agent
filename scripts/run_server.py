"""开发用启动脚本。

Windows 下 psycopg 异步不支持默认的 ProactorEventLoop，这里显式用 SelectorEventLoop
作为 uvicorn 的 loop factory（uvicorn 默认会选 Proactor，会绕过事件循环策略）。

用法：
    uv run python scripts/run_server.py
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uvicorn.config import Config  # noqa: E402
from uvicorn.server import Server  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="开发/压测用启动脚本")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    args = parser.parse_args()

    config = Config("app.main:app", host=args.host, port=args.port, log_level="info")
    server = Server(config)
    asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
