"""群聊接口（M4）：建群 / 列表 / 详情 / SSE 群聊。

SSE 事件（ADR-0004 §3.3）：`speaker` / `token` / `turn_done` / `done` / `error`。
"""

import json
import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response, StreamingResponse

from app.api.deps import get_group_service
from app.core.errors import AppError
from app.db.models import GroupSession
from app.schemas.group import GroupChatRequest, GroupCreate, GroupMessageRead, GroupRead
from app.schemas.session import RecallResult
from app.services.group_service import GroupService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/groups", tags=["groups"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _to_read(service: GroupService, group: GroupSession) -> GroupRead:
    return GroupRead(
        id=group.id,
        name=group.name,
        agent_ids=await service.member_ids(group.id),
        created_at=group.created_at,
    )


@router.get("", response_model=list[GroupRead])
async def list_groups(service: GroupService = Depends(get_group_service)) -> list[GroupRead]:
    return [await _to_read(service, g) for g in await service.list_groups()]


@router.post("", response_model=GroupRead, status_code=status.HTTP_201_CREATED)
async def create_group(
    payload: GroupCreate, service: GroupService = Depends(get_group_service)
) -> GroupRead:
    return await _to_read(service, await service.create(payload))


@router.get("/{group_id}", response_model=GroupRead)
async def get_group(
    group_id: int, service: GroupService = Depends(get_group_service)
) -> GroupRead:
    return await _to_read(service, await service.get(group_id))


@router.get("/{group_id}/messages", response_model=list[GroupMessageRead])
async def list_group_messages(
    group_id: int, service: GroupService = Depends(get_group_service)
) -> list[GroupMessageRead]:
    """该群的历史消息（展示用快照），按时间序。"""
    return await service.list_messages(group_id)


@router.post("/{group_id}/messages/{message_id}/recall", response_model=RecallResult)
async def recall_group_message(
    group_id: int,
    message_id: int,
    service: GroupService = Depends(get_group_service),
) -> RecallResult:
    """撤回一条群聊用户消息：整轮（含本轮全部角色回复）连带删除，并清除相关记忆。

    已归档为会话摘要的消息不允许撤回（返回 400）。
    """
    removed = await service.recall(group_session_id=group_id, message_id=message_id)
    return RecallResult(removed_message_ids=removed)


@router.post("/{group_id}/chat/stream")
async def group_chat_stream(
    group_id: int, payload: GroupChatRequest, service: GroupService = Depends(get_group_service)
) -> StreamingResponse:
    """流式群聊。群/成员校验在构造 StreamingResponse 之前完成（缺失 → 404）。"""
    context = await service.load_context(group_id)

    async def event_source():
        try:
            async for event, data in service.stream(context=context, message=payload.message):
                yield _sse(event, data)
        except AppError as exc:
            yield _sse("error", {"detail": str(exc)})
        except Exception:  # noqa: BLE001 - 流已开始，兜底为 error 事件，避免连接裸断
            logger.exception("group chat stream failed for group %s", group_id)
            yield _sse("error", {"detail": "internal server error"})

    return StreamingResponse(
        event_source(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.post("/{group_id}/idle/stream")
async def group_idle_stream(
    group_id: int, service: GroupService = Depends(get_group_service)
) -> StreamingResponse:
    """群聊自主闲聊：用户点击按钮触发一次，角色之间互相搭话。

    事件同 `/chat/stream`：`speaker` / `token` / `message_break` / `turn_done` / `done`。
    """
    context = await service.load_context(group_id)

    async def event_source():
        try:
            async for event, data in service.idle_stream(context=context):
                yield _sse(event, data)
        except AppError as exc:
            yield _sse("error", {"detail": str(exc)})
        except Exception:  # noqa: BLE001 - 流已开始，兜底为 error 事件
            logger.exception("group idle stream failed for group %s", group_id)
            yield _sse("error", {"detail": "internal server error"})

    return StreamingResponse(
        event_source(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.post("/{group_id}/idle/clear", response_model=RecallResult)
async def clear_group_idle(
    group_id: int, service: GroupService = Depends(get_group_service)
) -> RecallResult:
    """清空该群的全部自主闲聊消息（含写入各角色记忆的部分），返回被移除的消息 id。"""
    removed = await service.clear_idle(group_session_id=group_id)
    return RecallResult(removed_message_ids=removed)


@router.delete("/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_group(
    group_id: int, service: GroupService = Depends(get_group_service)
) -> Response:
    """删除群：连同该群全部消息与各成员的群聊工作记忆一并清理。"""
    await service.delete(group_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{group_id}/members/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_group_member(
    group_id: int, agent_id: int, service: GroupService = Depends(get_group_service)
) -> Response:
    """将某角色移出群；若移出后群内已无成员，则连同该群一并删除。"""
    await service.remove_member(group_id, agent_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
