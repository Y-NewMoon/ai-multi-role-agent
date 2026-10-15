"""开发用启动脚本。

Windows 下 psycopg 异步不支持默认的 ProactorEventLoop，这里显式用 SelectorEventLoop
作为 uvicorn 的 loop factory（uvicorn 默认会选 Proactor，会绕过事件循环策略）。

用法：
    uv run python scripts/run_server.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uvicorn.config import Config  # noqa: E402
from uvicorn.server import Server  # noqa: E402


def main() -> None:
    config = Config("app.main:app", host="127.0.0.1", port=8000, log_level="info")
    server = Server(config)
    asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
