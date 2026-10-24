"""提醒 DTO（V2.2 契约 §2.8）。"""

from datetime import datetime

from pydantic import BaseModel


class ReminderRead(BaseModel):
    """提醒资源视图；`status ∈ {pending, active, cancelled, fired}`。"""

    id: int
    agent_id: int
    session_id: int
    content: str
    remind_at: datetime
    status: str
    idempotent: bool = False
