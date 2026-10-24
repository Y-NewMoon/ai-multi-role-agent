"""世界书业务编排（契约 4.9）：导入 / 列出 / 详情 / 软删 / 增量重建 / preview。

服务只做编排与存在性校验；解析切分与索引在 `Indexer`，三路检索在 `Retriever`。
所有端点先校验 `agent_id` 存在（不存在 → `NotFoundError` → 404）。
"""

from __future__ import annotations

from dataclasses import asdict

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import NotFoundError
from app.db.models import Agent, LoreChunk, LoreDoc
from app.lore.indexer import Indexer
from app.lore.retriever import Retriever
from app.lore.schemas import (
    BudgetInfo,
    FusedItem,
    InjectedItem,
    LoreDocCreate,
    LoreDocImportOut,
    LoreDocRead,
    PreviewOut,
    ReindexIn,
    ReindexOut,
)


class LoreService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        indexer: Indexer,
        retriever: Retriever,
    ) -> None:
        self._session_factory = session_factory
        self._indexer = indexer
        self._retriever = retriever

    async def import_doc(self, data: LoreDocCreate) -> LoreDocImportOut:
        async with self._session_factory() as db:
            await self._require_agent(db, data.agent_id)
        result = await self._indexer.upsert_doc(
            agent_id=data.agent_id,
            slug=data.slug,
            title=data.title,
            content=data.content,
            always_on=data.always_on,
        )
        return LoreDocImportOut(**asdict(result))

    async def list_docs(self, agent_id: int) -> list[LoreDocRead]:
        async with self._session_factory() as db:
            await self._require_agent(db, agent_id)
            rows = (
                await db.execute(self._doc_read_stmt(agent_id))
            ).all()
        return [_to_read(*row) for row in rows]

    async def get_doc(self, doc_id: int) -> LoreDocRead:
        async with self._session_factory() as db:
            row = (
                await db.execute(
                    self._doc_read_stmt(None).where(LoreDoc.id == doc_id)
                )
            ).first()
        if row is None:
            raise NotFoundError(f"lore doc {doc_id} 不存在")
        return _to_read(*row)

    async def delete_doc(self, doc_id: int) -> None:
        """软删：doc 与 chunks 置 deleted，并清 embedding / tsv。"""
        async with self._session_factory() as db:
            doc = await db.get(LoreDoc, doc_id)
            if doc is None or doc.status != "active":
                raise NotFoundError(f"lore doc {doc_id} 不存在")
            doc.status = "deleted"
            await db.execute(
                update(LoreChunk)
                .where(LoreChunk.doc_id == doc_id)
                .values(status="deleted", embedding=None, tsv=None)
            )
            await db.commit()

    async def reindex(self, data: ReindexIn) -> ReindexOut:
        async with self._session_factory() as db:
            await self._require_agent(db, data.agent_id)
        result = await self._indexer.reindex(agent_id=data.agent_id, dry_run=data.dry_run)
        return ReindexOut(**asdict(result))

    async def preview(self, *, agent_id: int, q: str) -> PreviewOut:
        async with self._session_factory() as db:
            await self._require_agent(db, agent_id)
        result = await self._retriever.retrieve(agent_id=agent_id, query=q)
        return PreviewOut(
            query=result.query,
            rewritten=result.rewritten,
            rewrite_used=result.rewrite_used,
            routes=result.routes,
            fused=[FusedItem(source_ref=ref, score=score) for ref, score in result.fused],
            reranked=[
                FusedItem(source_ref=ref, score=score) for ref, score in result.reranked
            ],
            rerank=result.rerank,
            injected=[
                InjectedItem(source_ref=hit.source_ref, tokens=hit.token_count)
                for hit in result.injected
            ],
            budget=BudgetInfo(lore_tokens=result.lore_tokens, lore_cap=result.lore_cap),
        )

    # ---- 内部 ----

    async def _require_agent(self, db: AsyncSession, agent_id: int) -> None:
        if await db.get(Agent, agent_id) is None:
            raise NotFoundError(f"agent {agent_id} 不存在")

    @staticmethod
    def _doc_read_stmt(agent_id: int | None):
        """文档 + 活跃 chunk 数与 token 总数（左连接聚合）。"""
        stmt = (
            select(
                LoreDoc.id,
                LoreDoc.agent_id,
                LoreDoc.slug,
                LoreDoc.title,
                LoreDoc.version,
                LoreDoc.always_on,
                LoreDoc.status,
                func.count(LoreChunk.id),
                func.coalesce(func.sum(LoreChunk.token_count), 0),
                LoreDoc.created_at,
                LoreDoc.updated_at,
            )
            .outerjoin(
                LoreChunk,
                and_(LoreChunk.doc_id == LoreDoc.id, LoreChunk.status == "active"),
            )
            .where(LoreDoc.status == "active")
            .group_by(LoreDoc.id)
            .order_by(LoreDoc.id)
        )
        if agent_id is not None:
            stmt = stmt.where(LoreDoc.agent_id == agent_id)
        return stmt


def _to_read(*row) -> LoreDocRead:  # noqa: ANN002 - 聚合查询的整行
    return LoreDocRead(
        id=row[0],
        agent_id=row[1],
        slug=row[2],
        title=row[3],
        version=row[4],
        always_on=row[5],
        status=row[6],
        chunks=row[7],
        tokens=row[8],
        created_at=row[9],
        updated_at=row[10],
    )
