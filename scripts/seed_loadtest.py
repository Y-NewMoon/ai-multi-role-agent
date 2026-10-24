"""压测会话预置 / 清理（V2.4 R9）。

`load_test.py` 需要 20 个**不同** session（同 session 会被 B-02 串行）。
本脚本在 `DATABASE_URL` 指向的库里按前缀建会话，并输出可直接传给
`--session-ids` 的 id 列表；压测完用同一前缀清理（只删本脚本建的会话）。

用法：
    $env:DATABASE_URL="postgresql+psycopg://.../roleplay_test"
    uv run python scripts/seed_loadtest.py --count 20 --prefix loadtest-r9 \
      --out eval/report/loadtest-sessions.json
    uv run python scripts/load_test.py --session-ids <上面的 ids> ...
    uv run python scripts/seed_loadtest.py --cleanup --prefix loadtest-r9

注意：删除会话会连带删除 `messages` 快照；LangGraph checkpointer 的 thread
（`{session_id}:{agent_id}`）不在业务表内，本脚本不清理（测试库可忽略）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import delete, select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.base import create_engine, create_session_factory  # noqa: E402
from app.db.models import Agent, Message  # noqa: E402
from app.db.models import Session as ChatSession  # noqa: E402


async def _create(args) -> tuple[list[int], int]:
    settings = get_settings()
    engine = create_engine(settings.sqlalchemy_url)
    factory = create_session_factory(engine)
    try:
        async with factory() as db:
            agent_id = args.agent_id
            if agent_id is None:
                agent_id = (
                    await db.execute(select(Agent.id).order_by(Agent.id).limit(1))
                ).scalar_one_or_none()
            if agent_id is None:
                raise SystemExit("[seed_loadtest] 库里没有 agent，请先跑 seed_personas.py")
            actual_agent_id = int(agent_id)
            titles = [f"{args.prefix}-{i:02d}" for i in range(1, args.count + 1)]
            rows = [ChatSession(agent_id=actual_agent_id, title=t) for t in titles]
            for row in rows:
                db.add(row)
            await db.commit()
            ids = (
                await db.execute(
                    select(ChatSession.id)
                    .where(ChatSession.title.in_(titles))
                    .order_by(ChatSession.id)
                )
            ).scalars().all()
    finally:
        await engine.dispose()
    return [int(i) for i in ids], actual_agent_id


async def _cleanup(args) -> int:
    settings = get_settings()
    engine = create_engine(settings.sqlalchemy_url)
    factory = create_session_factory(engine)
    try:
        async with factory() as db:
            ids = (
                await db.execute(
                    select(ChatSession.id).where(ChatSession.title.like(f"{args.prefix}-%"))
                )
            ).scalars().all()
            if ids:
                await db.execute(delete(Message).where(Message.session_id.in_(ids)))
                await db.execute(delete(ChatSession).where(ChatSession.id.in_(ids)))
                await db.commit()
    finally:
        await engine.dispose()
    return len(ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="压测会话预置/清理（V2.4 R9）")
    parser.add_argument("--count", type=int, default=20, help="创建会话数（默认 20）")
    parser.add_argument(
        "--prefix",
        default=f"loadtest-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}",
        help="会话标题前缀；清理时传同一个值",
    )
    parser.add_argument("--agent-id", type=int, default=None, help="默认取库里第一个 agent")
    parser.add_argument("--cleanup", action="store_true", help="按前缀删除会话与消息")
    parser.add_argument(
        "--out", default="eval/report/loadtest-sessions.json",
        help="创建结果 JSON（含 session_ids）",
    )
    args = parser.parse_args(argv)

    # Windows 下 psycopg 异步需要 SelectorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if args.cleanup:
        removed = asyncio.run(_cleanup(args))
        print(f"[seed_loadtest] 已删除 {removed} 个会话（prefix={args.prefix!r}）")
        return 0

    ids, agent_id = asyncio.run(_create(args))
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"prefix": args.prefix, "agent_id": agent_id, "session_ids": ids},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[seed_loadtest] 已创建 {len(ids)} 个会话：{','.join(str(i) for i in ids)}")
    print(f"[seed_loadtest] 记录：{out_path}")
    return 0 if ids else 2


if __name__ == "__main__":
    raise SystemExit(main())
