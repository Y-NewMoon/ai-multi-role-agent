"""`get_current_time`（契约 §2.5-1）：无网络，按 IANA 时区返回当前时间。"""

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from app.skills.base import BaseSkill, ToolContext, ToolResult

DEFAULT_TIMEZONE = "Asia/Shanghai"


class GetCurrentTimeArgs(BaseModel):
    timezone: str = DEFAULT_TIMEZONE


def _offset_str(moment: datetime) -> str:
    offset = moment.utcoffset()
    minutes = int(offset.total_seconds() // 60) if offset else 0
    sign = "+" if minutes >= 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"{sign}{hours:02d}:{mins:02d}"


async def get_current_time(ctx: ToolContext, args: GetCurrentTimeArgs) -> ToolResult:
    try:
        tz = ZoneInfo(args.timezone)
    except Exception:  # noqa: BLE001 - 非法时区名（ZoneInfoNotFoundError 等）
        return ToolResult(
            ok=False,
            error_type="invalid_timezone",
            message=f"未知时区：{args.timezone}",
            data={"timezone": args.timezone},
        )
    now = datetime.now(tz)
    return ToolResult(
        ok=True,
        data={
            "iso": now.isoformat(),
            "human": f"{now.strftime('%Y-%m-%d %H:%M')} {args.timezone}",
            "timezone": args.timezone,
            "offset": _offset_str(now),
        },
    )


get_current_time_skill = BaseSkill(
    name="get_current_time",
    description="获取指定时区的当前日期与时间",
    when_to_use="需要知道当前日期/时间时使用",
    parameters=GetCurrentTimeArgs,
    handler=get_current_time,
)
