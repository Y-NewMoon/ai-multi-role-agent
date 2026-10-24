"""提醒接口（V2.2 契约 §2.8）：list / confirm / cancel。

HITL 通过 API 完成（D2）：工具只落 `pending`，用户在这里确认后才 `active`。
V2.2 不做定时投递；`fired` 状态保留给 V2.4。
"""

from fastapi import APIRouter, Depends, Response, status

from app.api.deps import get_reminder_service
from app.schemas.reminder import ReminderRead
from app.services.reminder_service import ReminderService, ReminderView

router = APIRouter(prefix="/reminders", tags=["reminders"])


def _to_read(view: ReminderView) -> ReminderRead:
    return ReminderRead(
        id=view.reminder_id,
        agent_id=view.agent_id,
        session_id=view.session_id,
        content=view.content,
        remind_at=view.remind_at,
        status=view.status,
        idempotent=view.idempotent,
    )


@router.get("", response_model=list[ReminderRead])
async def list_reminders(
    agent_id: int,
    session_id: int,
    service: ReminderService = Depends(get_reminder_service),
) -> list[ReminderRead]:
    """列出该会话的提醒（默认不含 cancelled）；agent/session 不存在 → 404。"""
    views = await service.list_reminders(agent_id=agent_id, session_id=session_id)
    return [_to_read(v) for v in views]


@router.post("/{reminder_id}/confirm", response_model=ReminderRead)
async def confirm_reminder(
    reminder_id: int, service: ReminderService = Depends(get_reminder_service)
) -> ReminderRead:
    """pending → active（幂等）；不存在 → 404。"""
    return _to_read(await service.confirm_reminder(reminder_id))


@router.delete("/{reminder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_reminder(
    reminder_id: int, service: ReminderService = Depends(get_reminder_service)
) -> Response:
    """pending/active → cancelled（幂等）；不存在 → 404。"""
    await service.cancel_reminder(reminder_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
