"""LLM tools 扩展单测（契约 §2.3）：解析 tool_calls、传参、默认 NotImplementedError。"""

import pytest

from app.llm.base import LLMProvider, LLMReply
from app.llm.openai_compat import OpenAICompatProvider


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _Fn(name, arguments)


class _Message:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _Completions:
    def __init__(self, message):
        self._message = message
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self._message)


class _FakeClient:
    def __init__(self, message):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(message)


def _provider(message):
    provider = OpenAICompatProvider(base_url="http://x", api_key="k", model="m")
    provider._client = _FakeClient(message)  # type: ignore[assignment]
    return provider


async def test_generate_with_tools_parses_calls_and_passes_params():
    message = _Message("好的", [_Call("call_1", "get_current_time", '{"timezone": "Asia/Tokyo"}')])
    provider = _provider(message)
    tools = [{"type": "function", "function": {"name": "get_current_time"}}]

    reply = await provider.generate_with_tools(
        [{"role": "user", "content": "几点"}], tools, tool_choice="none", temperature=0.0
    )

    assert isinstance(reply, LLMReply)
    assert reply.content == "好的"
    assert reply.tool_calls[0].name == "get_current_time"
    assert reply.tool_calls[0].arguments == {"timezone": "Asia/Tokyo"}
    sent = provider._client.chat.completions.calls[0]
    assert sent["tool_choice"] == "none"
    assert sent["temperature"] == 0.0
    assert sent["tools"] == tools


async def test_generate_with_tools_keeps_raw_on_bad_json():
    message = _Message("", [_Call("c", "roll_dice", "{not json")])
    reply = await _provider(message).generate_with_tools([{"role": "user", "content": "x"}], [])
    assert reply.tool_calls[0].arguments == {"_raw": "{not json"}


async def test_generate_with_tools_missing_id_generates_uuid():
    message = _Message("", [_Call(None, "roll_dice", "{}")])
    reply = await _provider(message).generate_with_tools([{"role": "user", "content": "x"}], [])
    assert reply.tool_calls[0].id  # 缺失 call id 时生成 uuid4


async def test_default_provider_raises_not_implemented():
    class _Bare(LLMProvider):
        async def generate(self, messages):
            return ""

        def astream(self, messages):
            raise NotImplementedError

    with pytest.raises(NotImplementedError):
        await _Bare().generate_with_tools([], [])
