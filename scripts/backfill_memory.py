"""V2.3 存量记忆回填（契约 §2.10）：补 `kind` / `importance` / `created_at`，幂等、可 dry-run。

- 目标：旧语义记忆（`text` 存在且 kind 缺失或为 `memory`）；`summary` / `fact` / `fact_cur` /
  `profile` / `l2` / `l2_counter` 一律跳过（不改 text / kind / key，不动新条目）；
- 缺 `kind` → 补 `"memory"`；缺 `importance` → 补 `0.5`；缺 `created_at` → 用 `item.updated_at`；
- 幂等：再跑一次不改任何值（输出「已是最新 N 条 / 待补 M 条」）；
- 单条失败只记 log，最后统计；失败数 > 0 → 退出码 1；
- `--dry-run` 必须零写入（只 aget/打印，不 aput）。

用法：
    uv run python scripts/backfill_memory.py --agent 1 --dry-run
    uv run python scripts/backfill_memory.py --all
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
from app.memory.namespace import MEMORY_SEGMENT, memory_namespace  # noqa: E402

_SKIP_KINDS = {"summary", "fact", "fact_cur", "profile", "l2", "l2_counter"}


def _needs_fix(value: dict, item) -> dict | None:
    """返回需要写入的字段补丁；无需修改返回 None。"""
    text = value.get("text")
    if not text:
        return None
    kind = value.get("kind")
    if kind is not None and kind != "memory":
        return None

    patch: dict = {}
    if kind is None:
        patch["kind"] = "memory"
    if value.get("importance") is None:
        patch["importance"] = 0.5
    if not value.get("created_at"):
        created = item.updated_at or item.created_at
        patch["created_at"] = created.isoformat() if created else None
        if patch["created_at"] is None:
            patch.pop("created_at")
    return patch or None


async def _scan_namespace(store, ns: tuple[str, ...], *, dry_run: bool) -> dict:
    stats = {"scanned": 0, "to_fix": 0, "fixed": 0, "skipped": 0, "failed": 0}
    offset = 0
    page = 1000
    while True:
        items = await store.asearch(ns, limit=page, offset=offset)
        if not items:
            break
        for item in items:
            stats["scanned"] += 1
            value = dict(item.value or {})
            if value.get("kind") in _SKIP_KINDS or not value.get("text"):
                stats["skipped"] += 1
                continue
            patch = _needs_fix(value, item)
            if patch is None:
                continue
            stats["to_fix"] += 1
            if dry_run:
                if stats["to_fix"] <= 5:
                    print(f"  [dry-run] {item.namespace}/{item.key} ← {patch}", flush=True)
                continue
            try:
                await store.aput(ns, item.key, {**value, **patch})
                stats["fixed"] += 1
            except Exception as exc:  # noqa: BLE001 - 单条失败只记 log
                stats["failed"] += 1
                print(f"  [warn] 回填失败 {item.namespace}/{item.key}：{exc}", flush=True)
        if len(items) < page:
            break
        offset += page
    return stats


async def _all_namespaces(store) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    offset = 0
    page = 1000
    while True:
        rows = await store.alist_namespaces(limit=page, offset=offset)
        if not rows:
            break
        out.extend(ns for ns in rows if len(ns) >= 2 and ns[-1] == MEMORY_SEGMENT)
        if len(rows) < page:
            break
        offset += page
    return out


async def _run(args: argparse.Namespace) -> int:
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    settings = get_settings()
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
        except Exception as exc:  # noqa: BLE001 - 回填只改元数据，无嵌入器也可继续
            print(f"[warn] 嵌入器不可用（不回写向量）：{exc}", flush=True)
        index = (
            {"dims": settings.embedding_dims, "embed": embed, "fields": ["text"]}
            if embed is not None
            else None
        )
        store = AsyncPostgresStore(pool, index=index)
        await store.setup()

        if args.all:
            namespaces = await _all_namespaces(store)
        else:
            namespaces = [memory_namespace(args.agent)]
        print(f"[info] 扫描 namespace 数：{len(namespaces)}（dry_run={args.dry_run}）", flush=True)

        total = {"scanned": 0, "to_fix": 0, "fixed": 0, "skipped": 0, "failed": 0}
        for ns in namespaces:
            stats = await _scan_namespace(store, ns, dry_run=args.dry_run)
            for key in total:
                total[key] += stats[key]
            if stats["to_fix"]:
                print(f"[info] {ns}: 待补 {stats['to_fix']}", flush=True)
    finally:
        await pool.close()

    print("---- 回填汇总 ----")
    print(f"扫描数        : {total['scanned']}")
    print(f"待补数        : {total['to_fix']}")
    print(f"已补数        : {total['fixed']}")
    print(f"跳过数        : {total['skipped']}")
    print(f"失败数        : {total['failed']}")
    if args.dry_run:
        up_to_date = total["scanned"] - total["to_fix"]
        print(f"已是最新 {up_to_date} 条 / 待补 {total['to_fix']} 条（dry-run 未写入）")
    return 1 if total["failed"] > 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="V2.3 存量记忆回填（幂等、可 dry-run）")
    ap.add_argument("--agent", type=int, default=None, help="只回填该角色")
    ap.add_argument("--all", action="store_true", help="回填所有角色的 namespace")
    ap.add_argument("--dry-run", action="store_true", help="只读不写")
    args = ap.parse_args()
    if not args.all and args.agent is None:
        ap.error("必须指定 --agent N 或 --all")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
