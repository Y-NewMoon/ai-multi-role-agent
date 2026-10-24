"""V2.1 索引测试（T3）：幂等、增量只动变化 chunk、tsv 中文分词、软删、增量重建。

跑真库 `roleplay_test`（需 `alembic upgrade head`）；库不可用时自动跳过。
"""

import hashlib

from langchain_core.embeddings import Embeddings
from sqlalchemy import delete, select, text, update

from app.db.models import Agent, LoreChunk, LoreDoc
from app.lore.indexer import Indexer, content_hash, source_ref

_SLUG = "tea_house"
_TITLE = "一盏春茶馆"
_HIST = "祖父在民国年间盘下这间铺面，取名一盏春茶馆，从此以茶为生。" * 3
_SIGN = "招牌是一盏春茉莉花茶，客人来了先奉一盏。" * 3


def _content(hist: str = _HIST, sign: str = _SIGN) -> str:
    return f"# 一盏春\n\n## 历史\n{hist}\n\n## 招牌\n{sign}\n"


class _FakeEmbedder(Embeddings):
    """确定性 1024 维向量（结构测试用，不做语义召回）。"""

    def __init__(self, dims: int = 1024) -> None:
        self._dims = dims

    def _vec(self, one: str) -> list[float]:
        digest = hashlib.sha256(one.encode("utf-8")).digest()
        raw = (digest * ((self._dims // len(digest)) + 1))[: self._dims]
        return [b / 255.0 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, one: str) -> list[float]:
        return self._vec(one)


async def _make_agent(session_factory) -> int:
    async with session_factory() as db:
        agent = Agent(name="lore-index-test", persona_prompt="测试人设")
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


async def _rows(session_factory, agent_id: int) -> list[LoreChunk]:
    async with session_factory() as db:
        return list(
            (
                await db.execute(
                    select(LoreChunk)
                    .where(LoreChunk.agent_id == agent_id)
                    .order_by(LoreChunk.source_ref)
                )
            )
            .scalars()
            .all()
        )


async def _tsv_refs(session_factory, agent_id: int, query: str) -> set[str]:
    """用 to_tsquery('simple', query) 命中哪些 source_ref。"""
    async with session_factory() as db:
        result = await db.execute(
            text(
                "SELECT source_ref FROM lore_chunks "
                "WHERE agent_id = :a AND tsv @@ to_tsquery('simple', :q)"
            ),
            {"a": agent_id, "q": query},
        )
        return {r[0] for r in result.all()}


async def test_upsert_idempotent(real_session_factory) -> None:
    """连导两次：版本不变、chunk 数不变、`changed=false`，且 chunk 主键不动。"""
    agent_id = await _make_agent(real_session_factory)
    indexer = Indexer(session_factory=real_session_factory, embedder=_FakeEmbedder())
    try:
        first = await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        before = await _rows(real_session_factory, agent_id)

        second = await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        after = await _rows(real_session_factory, agent_id)

        assert first.version == 1 and first.changed is True
        assert first.chunks == 2 and first.tokens > 0
        assert second.changed is False and second.version == 1 and second.chunks == 2
        assert [r.id for r in after] == [r.id for r in before]
        assert [r.content_hash for r in after] == [r.content_hash for r in before]
        assert [r.content for r in after] == [r.content for r in before]
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_import_after_soft_delete_restores(real_session_factory) -> None:
    """软删后同 slug 再导入：doc 与 chunk 恢复 active（同内容也不能静默无效）。"""
    agent_id = await _make_agent(real_session_factory)
    indexer = Indexer(session_factory=real_session_factory, embedder=_FakeEmbedder())
    try:
        await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        # 模拟服务层 delete_doc：doc 与 chunks 置 deleted，并清派生列
        async with real_session_factory() as db:
            doc = (
                await db.execute(select(LoreDoc).where(LoreDoc.agent_id == agent_id))
            ).scalar_one()
            doc.status = "deleted"
            await db.execute(
                update(LoreChunk)
                .where(LoreChunk.doc_id == doc.id)
                .values(status="deleted", embedding=None, tsv=None)
            )
            await db.commit()

        # 同内容重新导入：必须恢复而不是「内容未变化」静默返回
        restored = await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        async with real_session_factory() as db:
            doc = (
                await db.execute(select(LoreDoc).where(LoreDoc.agent_id == agent_id))
            ).scalar_one()
        rows = await _rows(real_session_factory, agent_id)

        assert restored.changed is True and restored.chunks == 2
        assert doc.status == "active"
        assert doc.version == 1  # 内容未变，不算新版本
        assert rows and all(r.status == "active" for r in rows)
        assert all(r.tsv is not None and r.embedding is not None for r in rows)

        # 恢复后再次导入：回到幂等
        again = await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        assert again.changed is False and again.version == 1
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_source_ref_is_slug_and_seq(real_session_factory) -> None:
    """`source_ref` 规则与 DB 一致：`slug:NN`（契约 4.3）。"""
    agent_id = await _make_agent(real_session_factory)
    indexer = Indexer(session_factory=real_session_factory, embedder=_FakeEmbedder())
    try:
        await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        refs = [r.source_ref for r in await _rows(real_session_factory, agent_id)]
        assert refs == [source_ref(_SLUG, 1), source_ref(_SLUG, 2)]
        assert refs == ["tea_house:01", "tea_house:02"]
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_incremental_only_changed_chunk(real_session_factory) -> None:
    """改一个 section：只有该 chunk 重算，另一 chunk 逐字不动，version+1。"""
    agent_id = await _make_agent(real_session_factory)
    indexer = Indexer(session_factory=real_session_factory, embedder=_FakeEmbedder())
    try:
        await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        before = {r.source_ref: r for r in await _rows(real_session_factory, agent_id)}

        changed_sign = _SIGN + "新添一句店训：先奉茶，后叙旧。"
        result = await indexer.upsert_doc(
            agent_id=agent_id,
            slug=_SLUG,
            title=_TITLE,
            content=_content(sign=changed_sign),
        )
        after = {r.source_ref: r for r in await _rows(real_session_factory, agent_id)}

        assert result.version == 2 and result.changed is True and result.chunks == 2
        # 未变化的 chunk：主键 / 内容 / hash 全部不动
        assert after["tea_house:01"].id == before["tea_house:01"].id
        assert after["tea_house:01"].content == before["tea_house:01"].content
        assert after["tea_house:01"].content_hash == before["tea_house:01"].content_hash
        # 变化的 chunk：内容与 hash 都变，主键复用（原地更新）
        assert after["tea_house:02"].content != before["tea_house:02"].content
        assert after["tea_house:02"].content_hash != before["tea_house:02"].content_hash
        assert after["tea_house:02"].status == "active"
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_tsv_is_segmented_not_whole_sentence(real_session_factory) -> None:
    """tsv 由 jieba 分词生成：中文词能单独命中，且词元数 > 1（非整句）。"""
    agent_id = await _make_agent(real_session_factory)
    indexer = Indexer(session_factory=real_session_factory, embedder=_FakeEmbedder())
    try:
        await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        # 两个词各自命中，且合起来（或连接）也命中
        assert "tea_house:01" in await _tsv_refs(real_session_factory, agent_id, "祖父")
        assert "tea_house:02" in await _tsv_refs(
            real_session_factory, agent_id, "招牌"
        )
        assert await _tsv_refs(real_session_factory, agent_id, "祖父 | 招牌") == {
            "tea_house:01",
            "tea_house:02",
        }
        # 词元数 >1：若整句存成一个词元，这里会是 1
        async with real_session_factory() as db:
            lexemes = (
                await db.execute(
                    text(
                        "SELECT array_length(tsvector_to_array(tsv), 1) FROM lore_chunks "
                        "WHERE agent_id = :a AND source_ref = :r"
                    ),
                    {"a": agent_id, "r": "tea_house:01"},
                )
            ).scalar_one()
        assert lexemes is not None and lexemes > 1
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_extra_chunk_soft_deleted(real_session_factory) -> None:
    """删掉一个 section 后，多余 `source_ref` 软删并清 embedding / tsv。"""
    agent_id = await _make_agent(real_session_factory)
    indexer = Indexer(session_factory=real_session_factory, embedder=_FakeEmbedder())
    try:
        await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        result = await indexer.upsert_doc(
            agent_id=agent_id,
            slug=_SLUG,
            title=_TITLE,
            content=f"# 一盏春\n\n## 历史\n{_HIST}\n",
        )
        rows = {r.source_ref: r for r in await _rows(real_session_factory, agent_id)}
        assert result.chunks == 1 and result.version == 2
        assert rows["tea_house:01"].status == "active"
        assert rows["tea_house:02"].status == "deleted"
        assert rows["tea_house:02"].embedding is None
        assert rows["tea_house:02"].tsv is None
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_embedding_and_tsv_written(real_session_factory) -> None:
    """落库后 embedding 为 1024 维、tsv 非空。"""
    agent_id = await _make_agent(real_session_factory)
    indexer = Indexer(session_factory=real_session_factory, embedder=_FakeEmbedder())
    try:
        await indexer.upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        for row in await _rows(real_session_factory, agent_id):
            assert row.embedding is not None and len(row.embedding) == 1024
            assert row.tsv is not None
            assert row.keywords, "关键词不应为空"
            assert row.content_hash == content_hash(row.content)
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_reindex_backfills_missing_derived(real_session_factory) -> None:
    """无嵌入器导入后 tsv 已在；带嵌入器 reindex 只补 embedding，幂等归零。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        # 无嵌入器：embedding 缺失，但 tsv（jieba）仍然写入
        await Indexer(session_factory=real_session_factory).upsert_doc(
            agent_id=agent_id, slug=_SLUG, title=_TITLE, content=_content()
        )
        rows = await _rows(real_session_factory, agent_id)
        assert all(r.embedding is None for r in rows)
        assert all(r.tsv is not None for r in rows)

        indexer = Indexer(session_factory=real_session_factory, embedder=_FakeEmbedder())
        dry = await indexer.reindex(agent_id=agent_id, dry_run=True)
        assert dry.dry_run is True and dry.documents == 1 and dry.chunks_rebuilt == 2

        done = await indexer.reindex(agent_id=agent_id)
        assert done.dry_run is False and done.chunks_rebuilt == 2
        assert all(r.embedding is not None for r in await _rows(real_session_factory, agent_id))

        again = await indexer.reindex(agent_id=agent_id)
        assert again.chunks_rebuilt == 0
    finally:
        await _cleanup(real_session_factory, agent_id)
