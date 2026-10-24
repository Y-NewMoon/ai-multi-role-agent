"""角色 DTO（M3：CRUD）。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AgentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    persona_prompt: str = Field(min_length=1)
    background: str | None = None
    avatar: str | None = Field(default=None, max_length=255)
    # V2.2：工具白名单（契约 §2.8）；空数组 = 不绑任何工具
    allowed_skills: list[str] = Field(default_factory=list)


class AgentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    persona_prompt: str | None = Field(default=None, min_length=1)
    background: str | None = None
    avatar: str | None = Field(default=None, max_length=255)
    allowed_skills: list[str] | None = None


class AgentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    persona_prompt: str
    background: str | None = None
    avatar: str | None = None
    allowed_skills: list[str] = Field(default_factory=list)
    created_at: datetime
