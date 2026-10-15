"""对话接口 DTO。"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: int
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: int
    agent_id: int
    reply: str


class ChatIdleRequest(BaseModel):
    """自主闲聊请求：仅需 session_id（无需用户消息）。"""

    session_id: int
