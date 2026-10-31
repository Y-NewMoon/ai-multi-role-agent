"""画像（profile）查看 / 修复工具。

用途：
- 查看某个 agent 的 4 个画像槽位；
- 把历史脏值按当前规则重新收尾（补引号、断标点、截断补省略号）；
- 手工改 / 清空某个槽位（`put_profile` 收到空值会删除该槽位）。

例：
    uv run python scripts/repair_profile.py --agent ext --show
    uv run python scripts/repair_profile.py --agent ext --normalize
    uv run python scripts/repair_profile.py --agent ext --set "重要日期=角色生日：10-25（出生首日）"
    uv run python scripts/repair_profile.py --agent ext --clear 禁忌
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg_pool import AsyncConnectionPool  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.memory.extractor import _truncate_profile  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    pool = AsyncConnectionPool(
        settings.database_url,
        open=False,
        min_size=1,
        max_size=2,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    await pool.wait()
    try:
        store = AsyncPostgresStore(pool)
        await store.setup()
        memory_store = MemoryStore(store)
        agent = args.agent

        before = await memory_store.get_profile(agent)
        print("== 当前画像 ==")
        print(json.dumps(before, ensure_ascii=False, indent=2))

        if args.show:
            return

        if args.normalize:
            for slot, value in before.items():
                fixed = _truncate_profile(value)
                if fixed != value:
                    await memory_store.put_profile(agent, slot, fixed)
                    print(f"[normalize] {slot}: {value!r} -> {fixed!r}")

        for item in args.set or []:
            slot, sep, value = item.partition("=")
            if not sep:
                print(f"[skip] --set 需要 SLOT=VALUE：{item!r}")
                continue
            await memory_store.put_profile(agent, slot.strip(), value)
            print(f"[set] {slot.strip()} = {value}")

        for slot in args.clear or []:
            await memory_store.put_profile(agent, slot, "")
            print(f"[clear] {slot}")

        after = await memory_store.get_profile(agent)
        print("== 修复后 ==")
        print(json.dumps(after, ensure_ascii=False, indent=2))
    finally:
        await pool.close()


def main() -> None:
    # Windows 下 psycopg 异步必须跑在 SelectorEventLoop（Proactor 会卡在连接池初始化）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description="查看/修复画像槽位")
    parser.add_argument("--agent", default="ext", help="记忆命名空间里的 agent，默认 ext")
    parser.add_argument("--show", action="store_true", help="只查看，不改")
    parser.add_argument("--normalize", action="store_true", help="按当前规则重新收尾所有槽位")
    parser.add_argument("--set", action="append", metavar="SLOT=VALUE", help="设置某个槽位")
    parser.add_argument("--clear", action="append", metavar="SLOT", help="清空某个槽位")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
