"""InstrumentedLLM：全局信号量 + 首 token/非流式超时兜底 + token/成本/trace 聚合（V2.4 §2.4/§2.9）。

只包装 provider，不改变其重试/断连续写语义（D10）：
- 全局 `asyncio.Semaphore(llm_max_concurrency)` 覆盖 generate / astream / generate_with_tools；
  流式持有到流结束；等待毫秒记入 trace `llm.wait_ms`。
- `generate` / `generate_with_tools`：`asyncio.wait_for(provider, llm_timeout_s)` 超时返回兜底文案，
  置 `llm_degraded`（降级轮不写记忆），不抛 500。
- `astream`：仅**首 token** 等待受 `llm_timeout_s` 约束；已出 token 后沿用 provider 300s 流超时。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from app.llm.base import ChatMessage, LLMProvider, LLMReply
from app.observability.cost import cost_cny
from app.observability.tokens import estimate_messages_tokens, estimate_tokens
from app.observability.trace import current_trace, mark_degraded, mark_error

logger = logging.getLogger(__name__)


class InstrumentedLLM(LLMProvider):
    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str = "",
        max_concurrency: int = 8,
        timeout_s: float = 30.0,
        fallback: str = "（我这边有点走神，稍等一下再说。）",
        input_price_per_1m: float = 2.0,
        output_price_per_1m: float = 8.0,
        usage_source: str = "estimator",
    ) -> None:
        self._provider = provider
        self._model = model or getattr(provider, "_model", "") or ""
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._timeout_s = float(timeout_s)
        self._fallback = fallback
        self._input_price = float(input_price_per_1m)
        self._output_price = float(output_price_per_1m)
        # R3：fake 压测写 `usage_source="fake"`，不得伪装成计费的 estimator
        self._usage_source = str(usage_source)

    # ===== 计数 =====

    async def _acquire(self) -> None:
        """等全局信号量；等待时间只记录，不计入 provider 超时。"""
        started = time.perf_counter()
        await self._semaphore.acquire()
        wait_ms = int((time.perf_counter() - started) * 1000)
        recorder = current_trace()
        if recorder is not None:
            recorder.llm_wait(wait_ms)

    def _record(
        self, *, prompt_tokens: int, completion_text: str, recorder=None
    ) -> None:
        recorder = recorder or current_trace()
        if recorder is None:
            return
        completion_tokens = estimate_tokens(completion_text)
        recorder.llm(
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_cny=cost_cny(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_price_per_1m=self._input_price,
                output_price_per_1m=self._output_price,
            ),
            usage_source=self._usage_source,
        )

    def _timeout_error(self, method: str) -> None:
        mark_degraded()
        mark_error("timeout", f"LLM {method} 超过 {self._timeout_s:g}s 未返回")
        logger.warning("LLM %s 超时（%ss），已返回兜底文案", method, self._timeout_s)

    def record_call(
        self, messages: list[dict], completion_text: str, *, recorder=None
    ) -> None:
        """把 trace 生命周期之外发生的调用（如群聊 supervisor 决策）补记到指定 trace。

        `recorder` 为空时记入当前 contextvar 的 trace；群聊调度会显式传入归属的
        recorder（R5：终止 END 决策归最后一位发言人）。
        """
        self._record(
            prompt_tokens=estimate_messages_tokens(messages),
            completion_text=completion_text,
            recorder=recorder,
        )

    # ===== 三个入口 =====

    async def generate(self, messages: list[ChatMessage]) -> str:
        prompt_tokens = estimate_messages_tokens(messages)  # type: ignore[arg-type]
        await self._acquire()
        try:
            try:
                text = await asyncio.wait_for(
                    self._provider.generate(messages), timeout=self._timeout_s
                )
            except TimeoutError:
                self._timeout_error("generate")
                text = self._fallback
                self._record(prompt_tokens=prompt_tokens, completion_text=text)
                return text
            self._record(prompt_tokens=prompt_tokens, completion_text=text)
            return text
        finally:
            self._semaphore.release()

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
    ) -> LLMReply:
        prompt_tokens = estimate_messages_tokens(messages)
        await self._acquire()
        try:
            try:
                reply = await asyncio.wait_for(
                    self._provider.generate_with_tools(
                        messages,
                        tools,
                        tool_choice=tool_choice,
                        temperature=temperature,
                    ),
                    timeout=self._timeout_s,
                )
            except TimeoutError:
                self._timeout_error("generate_with_tools")
                fallback = LLMReply(content=self._fallback)
                self._record(
                    prompt_tokens=prompt_tokens, completion_text=fallback.content
                )
                return fallback
            self._record(prompt_tokens=prompt_tokens, completion_text=reply.content or "")
            return reply
        finally:
            self._semaphore.release()

    async def astream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """首 token 超时 → 兜底文案作为唯一 token；出 token 后不再施加 30s 限制。"""
        prompt_tokens = estimate_messages_tokens(messages)  # type: ignore[arg-type]
        await self._acquire()
        stream = self._provider.astream(messages)
        produced: list[str] = []
        try:
            try:
                first = await asyncio.wait_for(
                    stream.__anext__(), timeout=self._timeout_s
                )
            except StopAsyncIteration:
                first = None
            except TimeoutError:
                self._timeout_error("astream")
                text = self._fallback
                self._record(prompt_tokens=prompt_tokens, completion_text=text)
                yield text
                return
            if first is not None:
                produced.append(first)
                yield first
                async for delta in stream:
                    produced.append(delta)
                    yield delta
            self._record(
                prompt_tokens=prompt_tokens, completion_text="".join(produced)
            )
        finally:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001 - 关闭流失败不得掩盖主异常
                pass
            self._semaphore.release()
