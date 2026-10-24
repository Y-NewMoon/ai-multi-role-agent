"""世界书接口：6 端点（契约 4.9）。"""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError

from app.api.deps import get_lore_service
from app.lore.schemas import (
    LoreDocCreate,
    LoreDocImportOut,
    LoreDocRead,
    PreviewOut,
    ReindexIn,
    ReindexOut,
)
from app.lore.service import LoreService

router = APIRouter(prefix="/lore", tags=["lore"])


@router.post("/docs", response_model=LoreDocImportOut, status_code=status.HTTP_201_CREATED)
async def import_doc(
    payload: LoreDocCreate, service: LoreService = Depends(get_lore_service)
) -> LoreDocImportOut:
    """导入 / 更新一篇设定（同 slug 即更新，version+1）。"""
    try:
        return await service.import_doc(payload)
    except IntegrityError as exc:  # 并发导入同一 slug 时由唯一约束兜底（§10.10）
        raise HTTPException(status_code=409, detail="slug 已存在") from exc


@router.get("/docs", response_model=list[LoreDocRead])
async def list_docs(
    agent_id: int, service: LoreService = Depends(get_lore_service)
) -> list[LoreDocRead]:
    return await service.list_docs(agent_id)


@router.get("/docs/{doc_id}", response_model=LoreDocRead)
async def get_doc(
    doc_id: int, service: LoreService = Depends(get_lore_service)
) -> LoreDocRead:
    return await service.get_doc(doc_id)


@router.delete("/docs/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_doc(
    doc_id: int, service: LoreService = Depends(get_lore_service)
) -> Response:
    await service.delete_doc(doc_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/reindex", response_model=ReindexOut)
async def reindex(
    payload: ReindexIn, service: LoreService = Depends(get_lore_service)
) -> ReindexOut:
    """增量重建：补齐缺失的 embedding / tsv（`dry_run` 只统计）。"""
    return await service.reindex(payload)


@router.get("/preview", response_model=PreviewOut)
async def preview(
    agent_id: int, q: str, service: LoreService = Depends(get_lore_service)
) -> PreviewOut:
    """调试：三路命中、RRF、rerank、最终注入（rerank 降级时为 false）。"""
    return await service.preview(agent_id=agent_id, q=q)
