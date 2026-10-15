"""群聊 DTO（M4）。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    agent_ids: list[int] = Field(min_length=1)


class GroupRead(BaseModel):
    id: int
    name: str
    agent_ids: list[int]
    created_at: datetime


class GroupChatRequest(BaseModel):
    message: str = Field(min_length=1)


class GroupMessageRead(BaseModel):
    """群聊展示用消息快照（历史回显）；agent_id 为空表示用户发言。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    agent_id: int | None = None
    content: str
    is_idle: bool = False
    # 该发言角色对应的「内心独白」（思考链）；用户发言/无思考时为空。
    thinking: str | None = None
    created_at: datetime
