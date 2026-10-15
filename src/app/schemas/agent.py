"""角色 DTO（M3：CRUD）。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AgentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    persona_prompt: str = Field(min_length=1)
    background: str | None = None
    avatar: str | None = Field(default=None, max_length=255)


class AgentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    persona_prompt: str | None = Field(default=None, min_length=1)
    background: str | None = None
    avatar: str | None = Field(default=None, max_length=255)


class AgentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    persona_prompt: str
    background: str | None = None
    avatar: str | None = None
    created_at: datetime
