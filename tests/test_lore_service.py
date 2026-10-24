"""V2.1 服务与 API 测试（T5）：导入/列表/详情/软删/重建/preview 与状态码。

路由层用替身服务（httpx ASGITransport）验证状态码；服务层用真库验证语义。
真库不可用时自动跳过。
"""

import hashlib

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.embeddings import Embeddings
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.api.v1 import api_router
from app.core.error_handlers import register_exception_handlers
from app.core.errors import NotFoundError
from app.db.models import Agent, LoreChunk, LoreDoc
from app.lore.indexer import Indexer
from app.lore.retriever import NullReranker, Retriever
from app.lore.schemas import LoreDocCreate, ReindexIn
from app.lore.service import LoreService

_TITLE = "一盏春茶馆"
_HIST = "祖父在民国年间盘下这间铺面，取名一盏春茶馆，从此以茶为生。" * 3
_SIGN = "招牌是一盏春茶馆的茉莉花茶，客人来了先奉一盏。" * 3
_CONTENT = f"# 一盏春\n\n## 历史\n{_HIST}\n\n## 招牌\n{_SIGN}\n"


class _FakeEmbedder(Embeddings):
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


# ============================ 路由层（替身服务） ============================


class _FakeLoreService:
    def __init__(self, *, boom: Exception | None = None) -> None:
        self.boom = boom
        self.deleted: list[int] = []

    async def import_doc(self, data):
        if self.boom:
            raise self.boom
        return {"doc_id": 1, "version": 1, "chunks": 2, "tokens": 120, "changed": True}

    async def list_docs(self, agent_id: int):
        if self.boom:
            raise self.boom
        return []

    async def get_doc(self, doc_id: int):
        if self.boom:
            raise self.boom
        return {
            "id": doc_id, "agent_id": 1, "slug": "tea_house", "title": _TITLE,
            "version": 1, "always_on": False, "status": "active", "chunks": 2,
            "tokens": 120, "created_at": "2026-10-18T00:00:00Z",
            "updated_at": "2026-10-18T00:00:00Z",
        }

    async def delete_doc(self, doc_id: int):
        if self.boom:
            raise self.boom
        self.deleted.append(doc_id)

    async def reindex(self, data):
        if self.boom:
            raise self.boom
        return {
            "agent_id": data.agent_id, "dry_run": data.dry_run,
            "documents": 1, "chunks_rebuilt": 0,
        }

    async def preview(self, *, agent_id: int, q: str):
        if self.boom:
            raise self.boom
        return {
            "query": q, "rewritten": q, "rewrite_used": False,
            "routes": {"always_on": [], "keyword": ["tea_house:01"], "vector": []},
            "fused": [{"source_ref": "tea_house:01", "score": 0.03}],
            "reranked": [{"source_ref": "tea_house:01", "score": 0.03}],
            "rerank": False,
            "injected": [{"source_ref": "tea_house:01", "tokens": 81}],
            "budget": {"lore_tokens": 81, "lore_cap": 1600},
        }


def _app(service) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    register_exception_handlers(app)
    app.state.lore_service = service
    return app


async def _request(method: str, url: str, *, service, payload: dict | None = None):
    transport = httpx.ASGITransport(app=_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, json=payload)


async def test_lore_routes_status_codes() -> None:
    service = _FakeLoreService()
    created = await _request(
        "POST",
        "/api/v1/lore/docs",
        service=service,
        payload={"agent_id": 1, "slug": "tea_house", "title": _TITLE, "content": _CONTENT},
    )
    assert created.status_code == 201 and created.json()["chunks"] == 2
    assert (await _request("GET", "/api/v1/lore/docs?agent_id=1", service=service)).status_code == 200
    assert (await _request("GET", "/api/v1/lore/docs/1", service=service)).status_code == 200
    assert (await _request("DELETE", "/api/v1/lore/docs/1", service=service)).status_code == 204
    assert (await _request("POST", "/api/v1/lore/reindex", service=service,
                           payload={"agent_id": 1, "dry_run": True})).status_code == 200
    preview = await _request("GET", "/api/v1/lore/preview?agent_id=1&q=%E8%8C%B6%E9%A6%86",
                             service=service)
    assert preview.status_code == 200 and preview.json()["rerank"] is False
    assert service.deleted == [1]


async def test_lore_route_slug_pattern_422() -> None:
    """slug 不合法 → 422（契约 4.3 的 API 校验）。"""
    resp = await _request(
        "POST",
        "/api/v1/lore/docs",
        service=_FakeLoreService(),
        payload={"agent_id": 1, "slug": "Bad Slug", "title": _TITLE, "content": _CONTENT},
    )
    assert resp.status_code == 422


async def test_lore_route_not_found_404() -> None:
    resp = await _request(
        "GET", "/api/v1/lore/docs?agent_id=999", service=_FakeLoreService(boom=NotFoundError("x"))
    )
    assert resp.status_code == 404


async def test_lore_route_import_conflict_409() -> None:
    """并发导入同一 slug：唯一约束兜底 → 409（契约 4.3 / §10.10）。"""
    boom = IntegrityError("INSERT INTO lore_docs", {}, Exception("duplicate key"))
    resp = await _request(
        "POST",
        "/api/v1/lore/docs",
        service=_FakeLoreService(boom=boom),
        payload={"agent_id": 1, "slug": "tea_house", "title": _TITLE, "content": _CONTENT},
    )
    assert resp.status_code == 409


# ============================ 服务层（真库） ============================


def _service(session_factory, *, embedder=None) -> LoreService:
    return LoreService(
        session_factory=session_factory,
        indexer=Indexer(session_factory=session_factory, embedder=embedder),
        retriever=Retriever(
            session_factory=session_factory, embedder=embedder, reranker=NullReranker()
        ),
    )


async def _make_agent(session_factory) -> int:
    async with session_factory() as db:
        agent = Agent(name="lore-service-test", persona_prompt="测试人设")
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


def _create(agent_id: int, **kwargs) -> LoreDocCreate:
    data = {
        "agent_id": agent_id, "slug": "tea_house", "title": _TITLE, "content": _CONTENT,
    }
    data.update(kwargs)
    return LoreDocCreate(**data)


async def test_import_list_detail(real_session_factory) -> None:
    agent_id = await _make_agent(real_session_factory)
    service = _service(real_session_factory, embedder=_FakeEmbedder())
    try:
        imported = await service.import_doc(_create(agent_id))
        assert imported.version == 1 and imported.chunks == 2 and imported.changed is True

        docs = await service.list_docs(agent_id)
        assert len(docs) == 1 and docs[0].chunks == 2 and docs[0].tokens > 0

        detail = await service.get_doc(imported.doc_id)
        assert detail.slug == "tea_house" and detail.chunks == 2
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_import_twice_idempotent(real_session_factory) -> None:
    """同 slug 同内容再导入：version 不变、changed=false。"""
    agent_id = await _make_agent(real_session_factory)
    service = _service(real_session_factory, embedder=_FakeEmbedder())
    try:
        first = await service.import_doc(_create(agent_id))
        second = await service.import_doc(_create(agent_id))
        assert second.changed is False and second.version == 1
        assert second.doc_id == first.doc_id and second.chunks == first.chunks
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_delete_soft(real_session_factory) -> None:
    """DELETE 软删：doc 与 chunks 置 deleted、清 embedding，列表不再可见。"""
    agent_id = await _make_agent(real_session_factory)
    service = _service(real_session_factory, embedder=_FakeEmbedder())
    try:
        imported = await service.import_doc(_create(agent_id))
        await service.delete_doc(imported.doc_id)

        assert await service.list_docs(agent_id) == []
        with pytest.raises(NotFoundError):
            await service.get_doc(imported.doc_id)

        async with real_session_factory() as db:
            chunks = list(
                (await db.execute(select(LoreChunk).where(LoreChunk.agent_id == agent_id)))
                .scalars()
                .all()
            )
            doc = await db.get(LoreDoc, imported.doc_id)
        assert doc.status == "deleted"
        assert all(c.status == "deleted" and c.embedding is None for c in chunks)
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_agent_not_found(real_session_factory) -> None:
    """所有端点先校验 agent 存在（不存在 → NotFoundError）。"""
    service = _service(real_session_factory, embedder=_FakeEmbedder())
    missing = 987654321
    with pytest.raises(NotFoundError):
        await service.import_doc(_create(missing))
    with pytest.raises(NotFoundError):
        await service.list_docs(missing)
    with pytest.raises(NotFoundError):
        await service.reindex(ReindexIn(agent_id=missing, dry_run=True))
    with pytest.raises(NotFoundError):
        await service.preview(agent_id=missing, q="茶馆")


async def test_preview_structure(real_session_factory) -> None:
    agent_id = await _make_agent(real_session_factory)
    service = _service(real_session_factory, embedder=_FakeEmbedder())
    try:
        await service.import_doc(_create(agent_id))
        preview = await service.preview(agent_id=agent_id, q="祖父开的茶馆")
        assert set(preview.routes) == {"always_on", "keyword", "vector"}
        assert "tea_house:01" in preview.routes["keyword"]
        assert preview.rerank is False
        assert preview.injected and preview.injected[0].tokens > 0
        assert preview.budget.lore_cap == 1600
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_reindex_dry_run_and_backfill(real_session_factory) -> None:
    """无嵌入器导入 → dry_run 统计待补数；带嵌入器重建后归零。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        await _service(real_session_factory).import_doc(_create(agent_id))

        dry = await _service(real_session_factory, embedder=_FakeEmbedder()).reindex(
            ReindexIn(agent_id=agent_id, dry_run=True)
        )
        assert dry.dry_run is True and dry.documents == 1 and dry.chunks_rebuilt == 2

        done = await _service(real_session_factory, embedder=_FakeEmbedder()).reindex(
            ReindexIn(agent_id=agent_id)
        )
        assert done.chunks_rebuilt == 2

        again = await _service(real_session_factory, embedder=_FakeEmbedder()).reindex(
            ReindexIn(agent_id=agent_id)
        )
        assert again.chunks_rebuilt == 0
    finally:
        await _cleanup(real_session_factory, agent_id)
