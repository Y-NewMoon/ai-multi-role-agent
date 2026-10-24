"""V2.1 数据层测试（T1）：lore_docs / lore_chunks 表结构、默认值与唯一约束。

跑真库 `roleplay_test`（需 vector 扩展与 `alembic upgrade head`）；库不可用时自动跳过。
"""

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.db.models import Agent, LoreChunk, LoreDoc


async def _make_agent(session_factory) -> int:
    async with session_factory() as db:
        agent = Agent(name="lore-db-test", persona_prompt="测试人设")
        db.add(agent)
        await db.flush()
        agent_id = agent.id
        await db.commit()
    return agent_id


async def _cleanup(session_factory, agent_id: int) -> None:
    async with session_factory() as db:
        await db.execute(delete(LoreChunk).where(LoreChunk.agent_id == agent_id))
        await db.execute(delete(LoreDoc).where(LoreDoc.agent_id == agent_id))
        await db.execute(delete(Agent).where(Agent.id == agent_id))
        await db.commit()


async def _make_doc(db, agent_id: int, **kwargs) -> LoreDoc:
    doc = LoreDoc(
        agent_id=agent_id,
        slug=kwargs.get("slug", "tea_house"),
        title=kwargs.get("title", "一盏春茶馆"),
        content_hash=kwargs.get("content_hash", "hash-1"),
        always_on=kwargs.get("always_on", False),
    )
    db.add(doc)
    await db.flush()
    return doc


async def test_write_doc_and_chunk_roundtrip(real_session_factory) -> None:
    """写入 doc + chunk 后能读回，默认值正确（status=active、keywords={}）。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        async with real_session_factory() as db:
            doc = await _make_doc(db, agent_id)
            doc_id = doc.id
            db.add(
                LoreChunk(
                    doc_id=doc_id,
                    agent_id=agent_id,
                    seq=1,
                    source_ref="tea_house:01",
                    header_path="一盏春茶馆 / 历史",
                    content="祖父在 1979 年盘下这间铺面。",
                    token_count=16,
                    content_hash="chunk-hash-1",
                )
            )
            await db.commit()

        async with real_session_factory() as db:
            doc_row = (
                await db.execute(select(LoreDoc).where(LoreDoc.id == doc_id))
            ).scalar_one()
            assert doc_row.status == "active"
            assert doc_row.version == 1
            assert doc_row.always_on is False

            chunk = (
                await db.execute(
                    select(LoreChunk).where(LoreChunk.agent_id == agent_id)
                )
            ).scalar_one()
            assert chunk.source_ref == "tea_house:01"
            assert chunk.keywords == []
            assert chunk.status == "active"
            assert chunk.embedding is None
            assert chunk.tsv is None
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_source_ref_unique_per_agent(real_session_factory) -> None:
    """`UNIQUE (agent_id, source_ref)` 生效：同角色同 source_ref 第二次插入报错。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        async with real_session_factory() as db:
            doc = await _make_doc(db, agent_id)
            db.add_all(
                [
                    LoreChunk(
                        doc_id=doc.id,
                        agent_id=agent_id,
                        seq=1,
                        source_ref="tea_house:01",
                        header_path="t / a",
                        content="甲",
                        token_count=1,
                        content_hash="h1",
                    ),
                    LoreChunk(
                        doc_id=doc.id,
                        agent_id=agent_id,
                        seq=1,
                        source_ref="tea_house:01",
                        header_path="t / b",
                        content="乙",
                        token_count=1,
                        content_hash="h2",
                    ),
                ]
            )
            with pytest.raises(IntegrityError):
                await db.commit()
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_slug_unique_per_agent(real_session_factory) -> None:
    """`UNIQUE (agent_id, slug)` 生效：同角色同 slug 第二次插入报错。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        async with real_session_factory() as db:
            await _make_doc(db, agent_id)
            await db.commit()
        async with real_session_factory() as db:
            with pytest.raises(IntegrityError):
                await _make_doc(db, agent_id, content_hash="hash-2")
                await db.commit()
    finally:
        await _cleanup(real_session_factory, agent_id)
