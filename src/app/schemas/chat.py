"""对话接口 DTO。"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: int
    message: str = Field(min_length=1)
    # V2.4 D8：仅非流式 /chat 使用；SSE 忽略该字段（本版不做 SSE 幂等）
    client_msg_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: int
    agent_id: int
    reply: str
    # V2.4 D8：命中幂等缓存时为 true（加性字段，默认 false）
    idempotent: bool = False


class ChatIdleRequest(BaseModel):
    """自主闲聊请求：仅需 session_id（无需用户消息）。"""

    session_id: int
