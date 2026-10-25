"""V3 M2：记忆接口的请求/响应 DTO（Pydantic）。

字段格式见交接 §2.1；长度类业务校验在 service 里按配置执行（`memory_api_max_*`）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

_SAFE_PATTERN = r"^[A-Za-z0-9:_-]{1,64}$"
_IDEMPOTENCY_PATTERN = r"^[A-Za-z0-9:_-]{1,128}$"


class TurnRequest(BaseModel):
    agent: str = Field(pattern=_SAFE_PATTERN)
    session: str = Field(pattern=_SAFE_PATTERN)
    user_text: str
    replies: list[str] = Field(default_factory=list)
    at: str | None = None
    message_id: str | None = Field(default=None, pattern=_SAFE_PATTERN)
    idempotency_key: str | None = Field(default=None, pattern=_IDEMPOTENCY_PATTERN)


class ContextRequest(BaseModel):
    agent: str = Field(pattern=_SAFE_PATTERN)
    session: str = Field(pattern=_SAFE_PATTERN)
    query: str = ""
    limit: int | None = Field(default=None, ge=1, le=20)


class ForgetRequest(BaseModel):
    agent: str = Field(pattern=_SAFE_PATTERN)
    contents: list[str] = Field(min_length=1, max_length=100)


class TurnResponse(BaseModel):
    ok: bool = True
    contract: str = "1"
    agent: str
    session: str
    rounds: int
    facts_added: int
    profile_changed: bool
    summarized: bool
    l2_merged: bool
    buffer_size: int
    idempotent: bool


class FactView(BaseModel):
    text: str
    subject: str
    predicate: str
    object: str
    importance: float
    updated_at: str | None = None


class ContextResponse(BaseModel):
    ok: bool = True
    contract: str = "1"
    agent: str
    session: str
    profile: dict[str, str] = Field(default_factory=dict)
    boundaries: list[Any] = Field(default_factory=list)
    facts: list[FactView] = Field(default_factory=list)
    l2: str = ""
    memories: list[str] = Field(default_factory=list)
    summary: str = ""
    debug: dict[str, Any] = Field(default_factory=lambda: {"pool": 0, "top": []})


class ForgetResponse(BaseModel):
    ok: bool = True
    contract: str = "1"
    agent: str
    deleted: int
    missing: list[str] = Field(default_factory=list)


class StateResponse(BaseModel):
    ok: bool = True
    contract: str = "1"
    agent: str
    session: str | None = None
    entries: int
    facts: int
    profile_slots: int
    boundaries: int = 0
    l2_merges: int
    buffer_size: int
    last_turn_at: str | None = None
