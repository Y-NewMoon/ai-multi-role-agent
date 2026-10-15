"""会话 DTO（M3：CRUD）。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SessionCreate(BaseModel):
    agent_id: int
    title: str | None = Field(default=None, max_length=255)


class SessionUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_id: int
    title: str | None = None
    created_at: datetime


class MessageRead(BaseModel):
    """展示用消息快照（历史记录回显）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    is_idle: bool = False
    # 角色该条发言的「内心独白」（思考链）；用户消息/无思考时为空。
    thinking: str | None = None
    created_at: datetime


class RecallResult(BaseModel):
    """撤回结果：被移除的消息 id（用户消息 + 连带回复）。"""

    removed_message_ids: list[int]
