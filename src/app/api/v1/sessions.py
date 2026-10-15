"""会话接口：CRUD（M3）。"""

from fastapi import APIRouter, Depends, Query, Response, status

from app.api.deps import get_chat_service, get_session_service
from app.schemas.session import MessageRead, RecallResult, SessionCreate, SessionRead, SessionUpdate
from app.services.chat_service import ChatService
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionRead])
async def list_sessions(
    agent_id: int | None = Query(default=None),
    service: SessionService = Depends(get_session_service),
) -> list[SessionRead]:
    return await service.list(agent_id=agent_id)


@router.get("/{session_id}/messages", response_model=list[MessageRead])
async def list_session_messages(
    session_id: int, service: SessionService = Depends(get_session_service)
) -> list[MessageRead]:
    """该会话的历史消息（展示用快照），按时间序。"""
    return await service.list_messages(session_id)


@router.post("/{session_id}/messages/{message_id}/recall", response_model=RecallResult)
async def recall_session_message(
    session_id: int,
    message_id: int,
    service: ChatService = Depends(get_chat_service),
) -> RecallResult:
    """撤回一条用户消息：整轮（含角色回复）连带删除，并清除该消息产生的记忆。

    已归档为会话摘要的消息不允许撤回（返回 400）。
    """
    removed = await service.recall(session_id=session_id, message_id=message_id)
    return RecallResult(removed_message_ids=removed)


@router.post("/{session_id}/idle/clear", response_model=RecallResult)
async def clear_session_idle(
    session_id: int, service: ChatService = Depends(get_chat_service)
) -> RecallResult:
    """清空该会话的全部自主闲聊消息（含其写入的角色记忆），返回被移除的消息 id。"""
    removed = await service.clear_idle(session_id=session_id)
    return RecallResult(removed_message_ids=removed)


@router.post("", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate, service: SessionService = Depends(get_session_service)
) -> SessionRead:
    return await service.create(payload)


@router.get("/{session_id}", response_model=SessionRead)
async def get_session(
    session_id: int, service: SessionService = Depends(get_session_service)
) -> SessionRead:
    return await service.get(session_id)


@router.patch("/{session_id}", response_model=SessionRead)
async def update_session(
    session_id: int, payload: SessionUpdate, service: SessionService = Depends(get_session_service)
) -> SessionRead:
    return await service.update(session_id, payload)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: int, service: SessionService = Depends(get_session_service)
) -> Response:
    await service.delete(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
