"""V3 M2：本机记忆接口（4 个端点，只绑 127.0.0.1、不鉴权）。

- `POST /memory/turn`    追加一轮（用户 + 角色回复），触发记忆写入/抽取/摘要/L2；
- `POST /memory/context` 按 query 读取 profile/facts/L2/summary/memories；
- `POST /memory/forget`  按内容删除语义记忆；
- `GET  /memory/state`   查看条目/事实/缓冲规模。

`/memory/boundary` 不在 M2（随 M4 交付）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_memory_api_service
from app.schemas.memory import (
    ContextRequest,
    ContextResponse,
    ForgetRequest,
    ForgetResponse,
    StateResponse,
    TurnRequest,
    TurnResponse,
)
from app.services.memory_api_service import MemoryApiService

router = APIRouter(prefix="/memory", tags=["memory"])

_SAFE_PATTERN = r"^[A-Za-z0-9:_-]{1,64}$"


@router.post("/turn", response_model=TurnResponse)
async def memory_turn(
    payload: TurnRequest,
    service: MemoryApiService = Depends(get_memory_api_service),
) -> TurnResponse:
    data = await service.turn(
        agent=payload.agent,
        session=payload.session,
        user_text=payload.user_text,
        replies=payload.replies,
        at=payload.at,
        message_id=payload.message_id,
        idempotency_key=payload.idempotency_key,
    )
    return TurnResponse(**data)


@router.post("/context", response_model=ContextResponse)
async def memory_context(
    payload: ContextRequest,
    service: MemoryApiService = Depends(get_memory_api_service),
) -> ContextResponse:
    data = await service.context(
        agent=payload.agent,
        session=payload.session,
        query=payload.query,
        limit=payload.limit,
    )
    return ContextResponse(**data)


@router.post("/forget", response_model=ForgetResponse)
async def memory_forget(
    payload: ForgetRequest,
    service: MemoryApiService = Depends(get_memory_api_service),
) -> ForgetResponse:
    data = await service.forget(agent=payload.agent, contents=payload.contents)
    return ForgetResponse(**data)


@router.get("/state", response_model=StateResponse)
async def memory_state(
    agent: str = Query(..., pattern=_SAFE_PATTERN),
    session: str | None = Query(default=None, pattern=_SAFE_PATTERN),
    service: MemoryApiService = Depends(get_memory_api_service),
) -> StateResponse:
    data = await service.state(agent=agent, session=session)
    return StateResponse(**data)
