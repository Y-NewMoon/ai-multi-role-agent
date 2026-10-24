"""`set_reminder`（契约 §2.5-4）：HITL + 幂等；工具只创建 pending，不激活。"""

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from app.skills.base import BaseSkill, ToolContext, ToolResult
from app.services.reminder_service import create_pending

TZ = ZoneInfo("Asia/Shanghai")
MAX_CONTENT_LEN = 200


class SetReminderArgs(BaseModel):
    when: str
    content: str


def parse_when(when: str, *, now: datetime | None = None) -> datetime | None:
    """ISO8601（带时区）或 `YYYY-MM-DD HH:MM`（按 Asia/Shanghai）；必须晚于当前时间。

    解析失败 / 早于当前时间 一律返回 None（→ `invalid_time`）。
    """
    reference = now or datetime.now(TZ)
    text = (when or "").strip()
    if not text:
        return None
    moment: datetime | None = None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        try:
            moment = datetime.strptime(text, "%Y-%m-%d %H:%M")
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=TZ)
    if moment <= reference:
        return None
    return moment


def make_set_reminder_skill(session_factory=None) -> BaseSkill:  # noqa: ANN001
    async def handler(ctx: ToolContext, args: SetReminderArgs) -> ToolResult:
        moment = parse_when(args.when)
        if moment is None:
            return ToolResult(
                ok=False,
                error_type="invalid_time",
                message=f"无法解析或已过期的时间：{args.when}",
                data={"when": args.when},
            )
        content = (args.content or "").strip()
        if not 1 <= len(content) <= MAX_CONTENT_LEN:
            return ToolResult(
                ok=False,
                error_type="invalid_content",
                message=f"提醒内容长度须在 1~{MAX_CONTENT_LEN} 字",
                data={"content": args.content},
            )
        if session_factory is None:
            return ToolResult(ok=False, error_type="unavailable", message="提醒存储不可用")
        view = await create_pending(
            session_factory,
            agent_id=ctx.agent_id,
            session_id=ctx.session_id,
            content=content,
            remind_at=moment,
        )
        return ToolResult(
            ok=True,
            data={
                "reminder_id": view.reminder_id,
                "status": view.status,
                "remind_at": view.remind_at.isoformat(),
                "idempotent": view.idempotent,
            },
        )

    return BaseSkill(
        name="set_reminder",
        description=(
            "为用户创建一个待确认的提醒事项；创建后必须明确请用户确认，"
            "不得说‘已记下/已设置’。"
        ),
        when_to_use=(
            "用户要求在某时间提醒他做某事时使用；创建后必须明确请用户确认，"
            "不得说‘已记下/已设置’。"
        ),
        parameters=SetReminderArgs,
        handler=handler,
        side_effect=True,
    )
