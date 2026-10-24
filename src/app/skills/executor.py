"""工具执行器（契约 §2.2）：校验 → 超时/重试 → 缓存/熔断 → 截断 → 统一 ToolResult。

**绝不抛业务异常**：任何未捕获异常都收敛为 `internal` 的 ToolResult，避免请求 500。
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import replace
from typing import Any

from app.core.errors import UpstreamError
from app.lore.chunker import estimate_tokens
from app.observability.trace import current_trace
from app.skills.base import ToolCall, ToolContext, ToolResult
from app.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# 单条工具结果上限；单次请求工具结果总量上限（契约 §2.2-4 / §2.7）
MAX_RESULT_TOKENS = 4000
MAX_TOTAL_TOKENS = 8000
# 上游错误熔断时长（进程内，契约 §2.2-3）
CIRCUIT_COOLDOWN_S = 60.0
# 重试指数退避基数（秒，契约 §2.2-2）
RETRY_BACKOFF_S = 0.5


def canonical_key(name: str, arguments: dict[str, Any]) -> str:
    """同请求缓存 key = sha256(name + canonical_json(arguments))（契约 §2.2-5）。"""
    payload = json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(f"{name}{payload}".encode()).hexdigest()


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - 序列化兜底
        return str(value)


def _status_code(exc: BaseException) -> int | None:
    """尽力从异常上取 HTTP 状态码（考虑 status_code / response.status_code）。"""
    for holder in (exc, getattr(exc, "response", None)):
        code = getattr(holder, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_upstream_error(exc: BaseException) -> bool:
    """5xx / 429 / 项目统一 UpstreamError → 需要重试并可能熔断（契约 §2.2-3）。"""
    if isinstance(exc, UpstreamError):
        return True
    code = _status_code(exc)
    return code == 429 or (code is not None and code >= 500)


def truncate_result(result: ToolResult) -> ToolResult:
    """`data` 序列化后超过 4000 token → 截断并附 summary；`ok=True`、`truncated=True`。"""
    serialized = _to_json(result.data)
    if estimate_tokens(serialized) <= MAX_RESULT_TOKENS:
        return result

    data = result.data if isinstance(result.data, dict) else {"value": result.data}
    out = dict(data)
    # 优先按「最大的列表字段」逐条裁剪，保留结构
    list_key = max(
        (k for k, v in out.items() if isinstance(v, list) and v),
        key=lambda k: estimate_tokens(_to_json(out[k])),
        default=None,
    )
    if list_key is not None:
        items = list(out[list_key])
        while items and estimate_tokens(_to_json({**out, list_key: items})) > MAX_RESULT_TOKENS:
            items.pop()
        out[list_key] = items
    # 仍超预算则截断长字符串字段
    if estimate_tokens(_to_json(out)) > MAX_RESULT_TOKENS:
        for key, value in list(out.items()):
            if isinstance(value, str) and len(value) > 200:
                out[key] = value[:200]
    out["truncated"] = True
    out["summary"] = f"结果过长，已截断（原始约 {estimate_tokens(serialized)} token）"
    return replace(result, data=out, ok=True)


class ToolExecutor:
    """执行工具调用，返回统一的 `ToolResult`；有界重试 + 缓存 + 熔断。

    缓存作用域是**一次请求 / 一次图运行**（契约 §2.2-5）：调用方在每次图运行
    （ChatService 的 chat/stream/idle_stream、Supervisor 每角色每轮、MCP 每次
    call_tool、评测脚本每个 case）前调用 `reset()`。缓存条目按 `agent_id` 分桶，
    保证工具结果不跨角色共享（§2.4）。
    """

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        max_tool_rounds: int = 3,
        sleep=asyncio.sleep,
        clock=time.monotonic,
    ) -> None:
        self._registry = registry
        self._max_tool_rounds = max_tool_rounds
        self._sleep = sleep
        self._clock = clock
        # key = (agent_id, canonical_key)；agent 分桶防止跨角色命中（§2.4）
        self._cache: dict[tuple[int, str], ToolResult] = {}
        self._circuit_until: dict[str, float] = {}

    def reset(self) -> None:
        """开启一次新的「请求 / 图运行」：清空同请求缓存（契约 §2.2-5）。

        熔断状态不清（§2.2-3 的冷却按进程内上游健康度计算）。
        """
        self._cache.clear()

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        start = time.perf_counter()
        name = call.name

        skill = self._registry.get(name)
        if skill is None or (ctx.allowed_skills and name not in ctx.allowed_skills):
            return self._finish(
                ToolResult(
                    ok=False,
                    error_type="unknown_tool",
                    message=f"未注册或未授权的工具：{name}",
                    data={"available": self._registry.names()},
                ),
                start,
                name,
            )

        # 同请求缓存：第 2 次同 key 直接返回，不再执行（§2.2-5）；按 agent 分桶（§2.4）
        cache_key = (ctx.agent_id, canonical_key(name, call.arguments))
        if cache_key in self._cache:
            return self._finish(replace(self._cache[cache_key], cached=True), start, name)

        # 熔断中：直接返回 upstream_error（§2.2-3）
        until = self._circuit_until.get(name)
        if until is not None and self._clock() < until:
            return self._finish(
                ToolResult(
                    ok=False,
                    error_type="upstream_error",
                    message=f"工具 {name} 熔断中，请稍后再试",
                ),
                start,
                name,
            )

        # 参数校验（§2.2-1）
        try:
            args = skill.parameters.model_validate(call.arguments)
        except Exception as exc:  # pydantic ValidationError
            fields = self._field_errors(exc)
            return self._finish(
                ToolResult(
                    ok=False,
                    error_type="invalid_arguments",
                    message=f"参数校验失败：{fields}",
                    data={"fields": fields},
                ),
                start,
                name,
            )

        # 超时 + 重试（§2.2-2 / §2.2-3）
        result = await self._run_with_retries(skill, ctx, args)
        if result.ok:
            result = truncate_result(result)
            self._cache[cache_key] = result
        return self._finish(result, start, name)

    async def _run_with_retries(self, skill, ctx: ToolContext, args) -> ToolResult:
        attempts = 0
        while True:
            try:
                result = await asyncio.wait_for(skill.handler(ctx, args), timeout=skill.timeout_s)
                if not isinstance(result, ToolResult):
                    return ToolResult(ok=False, error_type="internal", message="工具返回值类型非法")
                return result
            except asyncio.TimeoutError:
                if attempts >= skill.max_retries:
                    return ToolResult(ok=False, error_type="timeout", message=f"工具 {skill.name} 超时")
                attempts += 1
                await self._sleep(RETRY_BACKOFF_S * (2 ** (attempts - 1)))
            except Exception as exc:  # noqa: BLE001
                if _is_upstream_error(exc):
                    if attempts >= skill.max_retries:
                        self._open_circuit(skill.name)
                        return ToolResult(
                            ok=False,
                            error_type="upstream_error",
                            message=f"工具 {skill.name} 上游错误",
                        )
                    attempts += 1
                    await self._sleep(RETRY_BACKOFF_S * (2 ** (attempts - 1)))
                    continue
                logger.exception("工具 %s 未捕获异常", skill.name)
                return ToolResult(ok=False, error_type="internal", message=f"工具 {skill.name} 内部错误")

    def _open_circuit(self, name: str) -> None:
        self._circuit_until[name] = self._clock() + CIRCUIT_COOLDOWN_S

    @staticmethod
    def _field_errors(exc: BaseException) -> dict[str, str]:
        fields: dict[str, str] = {}
        errors = getattr(exc, "errors", None)
        if callable(errors):
            for err in errors():
                loc = ".".join(str(p) for p in err.get("loc", ())) or "__root__"
                fields[loc] = str(err.get("msg", "invalid"))
        if not fields:
            fields["__root__"] = str(exc)
        return fields

    def _finish(self, result: ToolResult, start: float, name: str = "") -> ToolResult:
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        # V2.4：每次工具调用一条 trace（不改既有返回语义）
        recorder = current_trace()
        if recorder is not None:
            recorder.tool(name, result.ok, result.latency_ms, result.error_type)
        return result
