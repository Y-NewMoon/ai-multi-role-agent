"""Skill 协议与数据结构（契约 §2.1）。

`Skill` 用 `Protocol` 描述接口（供类型标注）；`BaseSkill` 是内置工具实际继承的
具体基类，提供 `to_openai_tool()` 的共享实现与默认值（timeout/retries/side_effect/examples）。
两者字段一一对应：任何满足 `Skill` 协议的对象都可注册进 `SkillRegistry`。
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

# 工具名约束：小写字母开头，后接小写字母/数字/下划线，长度 2~41（契约 §2.1）
SKILL_NAME_PATTERN = r"^[a-z][a-z0-9_]{1,40}$"


@dataclass(frozen=True)
class ToolCall:
    """模型请求的一次工具调用。"""

    id: str  # 模型给的 call id；缺失时生成 uuid4
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """工具统一返回值（结构化；失败也不抛异常）。"""

    ok: bool
    data: Any = None
    error_type: str | None = None  # 见契约 §2.6
    message: str = ""
    latency_ms: int = 0
    cached: bool = False


@dataclass(frozen=True)
class ToolContext:
    """一次请求的工具执行上下文（由服务层注入）。"""

    agent_id: int
    session_id: int
    user_id: str | None = None
    allowed_skills: tuple[str, ...] = ()


Handler = Callable[[ToolContext, BaseModel], Awaitable[ToolResult]]


class Skill(Protocol):
    """工具协议：name 全局唯一；parameters 必须是 BaseModel 子类。"""

    name: str
    description: str
    when_to_use: str
    parameters: type[BaseModel]
    handler: Handler
    timeout_s: float
    max_retries: int
    side_effect: bool
    examples: list[dict[str, Any]]

    def to_openai_tool(self) -> dict[str, Any]:
        """{"type":"function","function":{"name","description","parameters": schema}}"""
        ...


@dataclass
class BaseSkill:
    """内置工具的共享基类：给默认值 + 共享 `to_openai_tool()` 实现。"""

    name: str
    description: str
    when_to_use: str
    parameters: type[BaseModel]
    handler: Handler = None  # type: ignore[assignment]
    timeout_s: float = 5.0
    max_retries: int = 1
    side_effect: bool = False
    examples: list[dict[str, Any]] = field(default_factory=list)
    # V2.4 B-01：可选可用性探针；None 表示始终可用（保持 V2.2 语义）
    available: Callable[[], bool] | None = None

    def is_available(self) -> bool:
        """不可用工具不下发给模型（registry / 【可用工具】两处过滤）。"""
        return True if self.available is None else bool(self.available())

    def to_openai_tool(self) -> dict[str, Any]:
        """按契约 §2.1 生成 OpenAI tools schema；`additionalProperties=false`。"""
        schema = self.parameters.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }
