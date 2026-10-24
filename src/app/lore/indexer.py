"""索引与增量重建（契约 4.3 / 4.9）：embedding + tsv + content_hash 去重 + 软删。

- `source_ref = f"{slug}:{seq:02d}"`（seq ≥ 100 时自然三位，不截断）；
- tsv 必须由 jieba 分词后再 `to_tsvector('simple', ...)`（契约 4.4，§10.1）；
- 增量：同 slug 再导入只重算内容变化的 chunk，多余 `source_ref` 软删；
- 幂等：同一文本连导两次，chunk 数与内容不变，`changed=false`。
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import LoreChunk, LoreDoc
from app.lore.chunker import chunk_document
from app.lore.keywords import cut_for_tsv, extract_keywords
from app.lore.parser import parse_markdown

_TSV_SQL = text(
    "UPDATE lore_chunks SET tsv = to_tsvector('simple', :t) WHERE id = :i"
)


def content_hash(value: str) -> str:
    """整篇 / 单个 chunk 的 sha256（契约 4.1）。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_ref(slug: str, seq: int) -> str:
    """稳定外部 id：`slug:NN`（契约 4.3）。"""
    return f"{slug}:{seq:02d}"


@dataclass
class IndexResult:
    """导入/更新的结果（对齐契约 4.9 的 POST /lore/docs 响应）。"""

    doc_id: int
    version: int
    chunks: int
    tokens: int
    changed: bool


@dataclass
class ReindexResult:
    """增量重建结果（契约 4.9 的 POST /lore/reindex 响应）。"""

    agent_id: int
    dry_run: bool
    documents: int
    chunks_rebuilt: int


class Indexer:
    """文档索引器：解析 → 切分 → 关键词 / tsv / 向量 → 落库。"""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embeddings | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder

    async def upsert_doc(
        self,
        *,
        agent_id: int,
        slug: str,
        title: str,
        content: str,
        always_on: bool = False,
    ) -> IndexResult:
        """导入 / 更新一篇设定：同 slug 即更新，内容或标题变化时 version+1。"""
        new_hash = content_hash(content)
        async with self._session_factory() as db:
            doc = (
                await db.execute(
                    select(LoreDoc).where(
                        LoreDoc.agent_id == agent_id, LoreDoc.slug == slug
                    )
                )
            ).scalar_one_or_none()

            if doc is None:
                doc = LoreDoc(
                    agent_id=agent_id,
                    slug=slug,
                    title=title,
                    version=1,
                    content_hash=new_hash,
                    always_on=always_on,
                )
                db.add(doc)
                await db.flush()
                chunks, tokens = await self._rebuild(
                    db, doc=doc, slug=slug, title=title, content=content
                )
                await db.commit()
                return IndexResult(doc.id, 1, chunks, tokens, changed=True)

            content_changed = doc.content_hash != new_hash
            title_changed = doc.title != title
            always_changed = doc.always_on != always_on
            # 软删（status=deleted）后同 slug 再导入：必须恢复为 active 并重建 chunks。
            # 否则同内容导入会静默无效（仍是隐藏文档），改内容导入则留下
            # doc=deleted / chunk=active 的不一致状态（检索能命中、列表看不到）。
            restored = doc.status != "active"

            if content_changed or title_changed or restored:
                doc.title = title
                doc.always_on = always_on
                doc.content_hash = new_hash
                doc.status = "active"
                if content_changed or title_changed:
                    doc.version += 1
                chunks, tokens = await self._rebuild(
                    db, doc=doc, slug=slug, title=title, content=content
                )
                await db.commit()
                return IndexResult(doc.id, doc.version, chunks, tokens, changed=True)

            # 内容未变：补齐缺失的派生列（embedding / tsv），保证幂等且可自愈
            rows = await self._active_chunks(db, agent_id=agent_id, doc_id=doc.id)
            pending = self._pending(rows)
            if pending:
                await self._write_derived(db, pending)
            if always_changed:
                doc.always_on = always_on
            await db.commit()
            return IndexResult(
                doc.id,
                doc.version,
                len(rows),
                sum(r.token_count for r in rows),
                changed=always_changed or bool(pending),
            )

    async def reindex(self, *, agent_id: int, dry_run: bool = False) -> ReindexResult:
        """增量重建：只补算 `status='active'` 且派生列缺失的 chunk。"""
        async with self._session_factory() as db:
            documents = len(
                (
                    await db.execute(
                        select(LoreDoc.id).where(
                            LoreDoc.agent_id == agent_id, LoreDoc.status == "active"
                        )
                    )
                )
                .scalars()
                .all()
            )
            rows = await self._active_chunks(db, agent_id=agent_id)
            pending = self._pending(rows)
            if dry_run or not pending:
                return ReindexResult(agent_id, dry_run, documents, len(pending))
            await self._write_derived(db, pending)
            await db.commit()
            return ReindexResult(agent_id, False, documents, len(pending))

    # ---- 内部 ----

    async def _active_chunks(
        self, db: AsyncSession, *, agent_id: int, doc_id: int | None = None
    ) -> list[LoreChunk]:
        stmt = select(LoreChunk).where(
            LoreChunk.agent_id == agent_id, LoreChunk.status == "active"
        )
        if doc_id is not None:
            stmt = stmt.where(LoreChunk.doc_id == doc_id)
        return list((await db.execute(stmt.order_by(LoreChunk.source_ref))).scalars().all())

    def _pending(self, rows: list[LoreChunk]) -> list[LoreChunk]:
        """派生列缺失、可补算的行（无 embedder 时不把向量缺失算作待补）。"""
        out: list[LoreChunk] = []
        for row in rows:
            if row.tsv is None or (row.embedding is None and self._embedder is not None):
                out.append(row)
        return out

    async def _rebuild(
        self, db: AsyncSession, *, doc: LoreDoc, slug: str, title: str, content: str
    ) -> tuple[int, int]:
        """重建该文档的 chunks：内容变→重算，仅标题变→只更新 header_path/keywords。"""
        new_chunks = chunk_document(parse_markdown(content, doc_title=title))
        existing = {
            row.source_ref: row
            for row in (
                await db.execute(select(LoreChunk).where(LoreChunk.doc_id == doc.id))
            )
            .scalars()
            .all()
        }

        touched: list[LoreChunk] = []
        seen: set[str] = set()
        for chunk in new_chunks:
            ref = source_ref(slug, chunk.seq)
            seen.add(ref)
            chunk_hash = content_hash(chunk.content)
            row = existing.get(ref)

            if row is None:
                row = LoreChunk(
                    doc_id=doc.id,
                    agent_id=doc.agent_id,
                    seq=chunk.seq,
                    source_ref=ref,
                    header_path=chunk.header_path,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    keywords=extract_keywords(chunk.content, chunk.header_path),
                    content_hash=chunk_hash,
                    status="active",
                )
                db.add(row)
                touched.append(row)
                continue

            if row.status == "active" and row.content_hash == chunk_hash:
                continue  # 未变化：整行不动

            row.seq = chunk.seq
            row.header_path = chunk.header_path
            row.content = chunk.content
            row.token_count = chunk.token_count
            row.keywords = extract_keywords(chunk.content, chunk.header_path)
            row.content_hash = chunk_hash
            row.status = "active"
            touched.append(row)

        # 多余 source_ref 软删，并清 embedding / tsv
        for ref, row in existing.items():
            if ref not in seen and row.status != "deleted":
                row.status = "deleted"
                row.embedding = None
                row.tsv = None

        await db.flush()
        await self._write_derived(db, touched)
        return len(new_chunks), sum(c.token_count for c in new_chunks)

    async def _write_derived(self, db: AsyncSession, rows: list[LoreChunk]) -> None:
        """写 tsv（jieba 分词后 to_tsvector）与 embedding。"""
        for row in rows:
            await db.execute(_TSV_SQL, {"t": cut_for_tsv(row.content), "i": row.id})
        if self._embedder is not None and rows:
            vectors = await asyncio.to_thread(
                self._embedder.embed_documents, [row.content for row in rows]
            )
            for row, vector in zip(rows, vectors):
                row.embedding = vector
