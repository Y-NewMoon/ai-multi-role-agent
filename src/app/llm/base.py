"""LLM Provider 抽象（方案 3.5：统一走 OpenAI 兼容协议）。"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str  # system / user / assistant
    content: str


class LLMProvider(ABC):
    """对话模型抽象：`generate` 一次性生成，`astream` 流式生成（M3）。"""

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> str:
        """给定 OpenAI 风格消息列表，返回助手回复文本。"""
        raise NotImplementedError

    @abstractmethod
    def astream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """流式产出助手回复的文本增量（delta）；M3 SSE 用。"""
        raise NotImplementedError
