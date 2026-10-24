"""LLM Provider 抽象（方案 3.5：统一走 OpenAI 兼容协议）。

V2.2 新增 `generate_with_tools`（默认 NotImplementedError，不强制所有 provider 实现）：
支持工具的 provider 实现 `SupportsTools`；不支持的 provider 由 `generate` 节点捕获
`NotImplementedError` 并降级为无工具文本回答（契约 §2.3）。
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

from app.skills.base import ToolCall


class ChatMessage(TypedDict):
    role: str  # system / user / assistant
    content: str


@dataclass
class LLMReply:
    """带工具调用的一次非流式回复。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class SupportsTools(Protocol):
    async def generate_with_tools(
        self,
        messages: list[dict],  # OpenAI 风格；assistant 可带 tool_calls，tool 带 tool_call_id
        tools: list[dict],
        *,
        tool_choice: str | dict = "auto",  # "auto" | "none" | 指定函数 {"type": "function", ...}
        temperature: float = 0.0,
    ) -> LLMReply: ...


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

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
    ) -> LLMReply:
        """默认不支持工具；由支持 tools 的 provider 覆写（契约 §2.3）。"""
        raise NotImplementedError
