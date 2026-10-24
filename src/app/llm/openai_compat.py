"""基于官方 openai SDK 的 OpenAI 兼容 Provider（DeepSeek 等）。"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAIError

from app.core.errors import UpstreamError
from app.llm.base import ChatMessage, LLMProvider, LLMReply
from app.llm.dsml import has_dsml_marker, parse_dsml_content
from app.skills.base import ToolCall

logger = logging.getLogger(__name__)

# 非流式请求用客户端默认 60s；流式回答可能较长，单次请求单独放宽到 300s（M3 决策 D2）。
_STREAM_TIMEOUT = 300.0

# 连接层抖动（如 Windows WinError 64 连接被重置）会偶发打断请求：
# - SDK 内建重试覆盖「建连阶段失败」；
# - 应用层重试覆盖「流式中途断连」——SDK 不会自动重试已产出 token 的流，
#   这里改为「携带已产出内容让模型续写」，避免用户看到半句就整段失败。
_SDK_MAX_RETRIES = 2
_MAX_ATTEMPTS = 3  # 流式请求的应用层最大尝试次数
_RETRY_BACKOFF = 0.5  # 指数退避基数（秒）
_CONTINUE_HINT = "（上一条回复因网络中断被打断）请紧接着上文继续输出剩余内容，不要重复已输出的文字。"
_OVERLAP_WINDOW = 200  # 续写去重时用于比较的最大字符数


def _trim_overlap(produced: str, continuation: str) -> str:
    """去掉续写开头与已产出文本结尾的重叠部分，避免重试造成内容重复。"""
    limit = min(len(produced), len(continuation), _OVERLAP_WINDOW)
    for i in range(limit, 0, -1):
        if produced[-i:] == continuation[:i]:
            return continuation[i:]
    return continuation


class OpenAICompatProvider(LLMProvider):
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        # 超时 60s、自动重试：LLM 挂起时不长期占用会话锁；重试仅发生在生成层，不影响记忆/快照写入
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=60.0, max_retries=_SDK_MAX_RETRIES
        )
        self._model = model

    async def generate(self, messages: list[ChatMessage]) -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
            )
        except OpenAIError as exc:
            # 兜底：把 SDK 异常收敛成本项目统一的 UpstreamError，避免裸 500
            logger.exception("LLM 非流式调用失败（model=%s）", self._model)
            raise UpstreamError("模型服务调用失败，请稍后再试") from exc
        return resp.choices[0].message.content or ""

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
    ) -> LLMReply:
        """支持 tools 的非流式调用（契约 §2.3）；`arguments` JSON 解析失败保留 `{"_raw": ...}`。"""
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            logger.exception("LLM 工具调用失败（model=%s）", self._model)
            raise UpstreamError("模型服务调用失败，请稍后再试") from exc

        message = resp.choices[0].message
        calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            raw_args = getattr(raw_call.function, "arguments", "") or ""
            try:
                parsed = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed = {"_raw": raw_args}
            if not isinstance(parsed, dict):
                parsed = {"_raw": raw_args}
            calls.append(
                ToolCall(
                    id=raw_call.id or uuid.uuid4().hex,
                    name=raw_call.function.name,
                    arguments=parsed,
                )
            )
        content = message.content or ""
        if not calls and has_dsml_marker(content):
            # 个别轮次模型把工具调用渲染成 DSML 文本：解析回 tool_calls 继续走工具循环；
            # 解析失败则丢弃文本（返回空 content），由上层空回复兜底重试，原文不落库。
            parsed_calls, cleaned, _ = parse_dsml_content(content)
            if parsed_calls:
                logger.warning(
                    "检测到 DSML 文本工具调用，已解析 %d 个 tool_call（model=%s）",
                    len(parsed_calls),
                    self._model,
                )
                calls, content = parsed_calls, cleaned
            else:
                logger.warning(
                    "检测到无法解析的 DSML 文本，已丢弃以避免污染历史（model=%s）", self._model
                )
                content = ""
        return LLMReply(content=content, tool_calls=calls)

    async def astream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """逐段产出回复文本；连接抖动时自动重试，中途断连则续写而非重头开始。

        失败重试耗尽后统一抛 UpstreamError（并记录底层异常，便于定位）。
        """
        produced = ""  # 已对外产出的全部文本
        convo: list[ChatMessage] = list(messages)
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            # 续写去重缓冲：仅「重试后」需要，先攒够窗口再对外产出，避免与已发内容重复
            buffered = ""
            need_dedup = bool(produced)
            try:
                stream = await self._client.chat.completions.create(
                    model=self._model,
                    messages=convo,  # type: ignore[arg-type]
                    stream=True,
                    timeout=_STREAM_TIMEOUT,
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if not delta:
                        continue
                    if need_dedup:
                        buffered += delta
                        if len(buffered) >= _OVERLAP_WINDOW:
                            out = _trim_overlap(produced, buffered)
                            if out:
                                produced += out
                                yield out
                            need_dedup = False
                    else:
                        produced += delta
                        yield delta
                # 续写片段较短、始终未攒够窗口：流结束时补一次去重
                if need_dedup and buffered:
                    out = _trim_overlap(produced, buffered)
                    if out:
                        produced += out
                        yield out
                return
            except OpenAIError as exc:
                if attempt >= _MAX_ATTEMPTS:
                    logger.exception(
                        "LLM 流式调用失败，已尝试 %d 次（model=%s）", attempt, self._model
                    )
                    raise UpstreamError("模型服务调用失败，请稍后再试") from exc
                logger.warning(
                    "LLM 流式调用中断，%.2fs 后进行第 %d/%d 次重试：%s",
                    _RETRY_BACKOFF * attempt,
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                )
                await asyncio.sleep(_RETRY_BACKOFF * attempt)
                if produced:
                    # 已产出部分内容：带上已产出的 assistant 回复让模型续写，避免重头开始
                    convo = [
                        *messages,
                        {"role": "assistant", "content": produced},
                        {"role": "user", "content": _CONTINUE_HINT},
                    ]
