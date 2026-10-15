"""对话接口：非流式 `POST /chat` + SSE 流式 `POST /chat/stream`（M3）。"""

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.errors import AppError
from app.schemas.chat import ChatIdleRequest, ChatRequest, ChatResponse
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关闭反代缓冲，保证逐 token 下发
}


def _sse(event: str, data: dict) -> str:
    """按 SSE 规范拼一帧：`event: <name>\\ndata: <json>\\n\\n`。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service: ChatService = request.app.state.chat_service
    return await service.chat(session_id=payload.session_id, message=payload.message)


@router.post("/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """流式对话。事件：`token`（增量）/ `done`（本轮结束）/ `error`（异常）。

    session/agent 校验在构造 StreamingResponse 之前完成，缺失时仍返回正常 404；
    一旦开始推流，后续异常只能以 `event: error` 表达（HTTP 状态已固定为 200）。
    """
    service: ChatService = request.app.state.chat_service
    context = await service.load_context(session_id=payload.session_id)

    async def event_source():
        try:
            async for event, data in service.stream(context=context, message=payload.message):
                yield _sse(event, data)
        except AppError as exc:
            yield _sse("error", {"detail": str(exc)})
        except Exception:  # noqa: BLE001 - 流已开始，兜底为 error 事件，避免连接裸断
            logger.exception("chat stream failed for session %s", context.session_id)
            yield _sse("error", {"detail": "internal server error"})

    return StreamingResponse(
        event_source(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.post("/idle/stream")
async def chat_idle_stream(payload: ChatIdleRequest, request: Request) -> StreamingResponse:
    """自主闲聊（私聊）：用户点击按钮触发一次。事件同 `/chat/stream`。

    session/agent 校验在建流前完成（缺失 → 404）。
    """
    service: ChatService = request.app.state.chat_service
    context = await service.load_context(session_id=payload.session_id)

    async def event_source():
        try:
            async for event, data in service.idle_stream(context=context):
                yield _sse(event, data)
        except AppError as exc:
            yield _sse("error", {"detail": str(exc)})
        except Exception:  # noqa: BLE001 - 流已开始，兜底为 error 事件
            logger.exception("chat idle stream failed for session %s", context.session_id)
            yield _sse("error", {"detail": "internal server error"})

    return StreamingResponse(
        event_source(), media_type="text/event-stream", headers=_SSE_HEADERS
    )
