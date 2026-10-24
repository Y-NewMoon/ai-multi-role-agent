"""V2.3 离线 L2 合成（契约 §2.7-8）。

- `--force`：不看计数立即重合成（回填与测试用）；
- 不带 `--force`：只在 `l2:counter >= memory_l2_every_merges` 时合成，否则只打印计数；
- 原料：现有 L2 + 最新会话 L1 摘要 + 高 importance facts（由 `force_update_l2` 内部组装）；
- 合成失败/空输出：不归零、不写 L2，退出码 1；成功退出 0。

用法：
    uv run python scripts/compact_memory.py --agent 1 --force
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.config import get_settings  # noqa: E402
from app.memory.namespace import memory_namespace  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402


async def _latest_summary(store, ns: tuple[str, ...]) -> str:
    items = await store.asearch(ns, limit=1000)
    summaries = [
        (str(item.value.get("updated_at") or item.updated_at or ""), str(item.value.get("text") or ""))
        for item in items
        if item.value.get("kind") == "summary"
    ]
    if not summaries:
        return ""
    summaries.sort(key=lambda pair: pair[0], reverse=True)
    return summaries[0][1]


async def _run(args: argparse.Namespace) -> int:
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    settings = get_settings()
    if not settings.llm_api_key:
        print("[error] LLM_API_KEY 为空，无法合成 L2", file=sys.stderr)
        return 2

    pool = AsyncConnectionPool(
        settings.database_url,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    try:
        await pool.open()
        await pool.wait(timeout=5)
    except Exception as exc:  # noqa: BLE001
        await pool.close()
        print(f"[error] 数据库不可用：{exc}", file=sys.stderr)
        return 2

    try:
        from langgraph.store.postgres.aio import AsyncPostgresStore

        embed = None
        try:
            from app.llm.factory import create_embedding_provider

            embed = create_embedding_provider(settings)
        except Exception as exc:  # noqa: BLE001 - L2 写入不依赖向量，但索引可用时保持矢量一致
            print(f"[warn] 嵌入器不可用：{exc}", flush=True)
        index = (
            {"dims": settings.embedding_dims, "embed": embed, "fields": ["text"]}
            if embed is not None
            else None
        )
        store = AsyncPostgresStore(pool, index=index)
        await store.setup()
        ms = MemoryStore(store)

        merges = await ms.get_l2_counter(args.agent)
        every_merges = settings.memory_l2_every_merges
        print(
            f"[info] agent={args.agent} L1 合并计数={merges}/{every_merges} "
            f"force={args.force}",
            flush=True,
        )
        if not args.force and merges < every_merges:
            print("[info] 计数未达标，跳过合成（用 --force 立即合成）")
            return 0

        from app.llm.factory import create_llm_provider
        from app.memory.l2 import force_update_l2

        current_l1 = await _latest_summary(store, memory_namespace(args.agent))
        llm = create_llm_provider(settings)
        text = await force_update_l2(
            ms,
            args.agent,
            llm,
            current_l1=current_l1,
            max_chars=settings.memory_l2_max_chars,
        )
        if not text:
            print("[error] L2 合成失败（计数不归零，未写入）", file=sys.stderr)
            return 1
        print(f"[info] L2 已更新（{len(text)} 字）：{text[:80]}…")
        return 0
    finally:
        await pool.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="V2.3 离线合成 L2 关系档案")
    ap.add_argument("--agent", type=int, required=True, help="角色 id")
    ap.add_argument("--force", action="store_true", help="不看计数立即合成")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
