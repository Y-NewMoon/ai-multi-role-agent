"""DSML 文本工具调用解析（DeepSeek 偶发把 tool_calls 渲染成正文）单测。"""

from app.llm.dsml import has_dsml_marker, normalize_dsml, parse_dsml_content
from app.llm.openai_compat import OpenAICompatProvider

# 复刻线上泄漏样本：竖线为全角 U+FF5C
LEAKED = """<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="set_reminder">
<｜｜DSML｜｜ parameter name="content" string="true">看掷骰结果</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="when" string="true">2026-10-24T19:42:00+08:00</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>"""


def test_normalize_and_marker():
    assert has_dsml_marker(LEAKED)
    assert not has_dsml_marker("今天天气不错")
    assert normalize_dsml(LEAKED).startswith("<DSML calls>")


def test_parse_leaked_sample():
    calls, cleaned, had = parse_dsml_content(LEAKED)
    assert had is True
    assert cleaned == ""
    assert len(calls) == 1
    assert calls[0].name == "set_reminder"
    assert calls[0].arguments == {
        "content": "看掷骰结果",
        "when": "2026-10-24T19:42:00+08:00",
    }
    assert calls[0].id


def test_parse_ascii_bars_and_typed_values():
    text = (
        '<|DSML| calls><|DSML| invoke name="roll_dice">'
        '<|DSML| parameter name="dice" string="true">3d6</|DSML| parameter>'
        '<|DSML| parameter name="times" string="false">2</|DSML| parameter>'
        "</|DSML| invoke></|DSML| calls>"
    )
    calls, cleaned, had = parse_dsml_content(text)
    assert had is True
    assert cleaned == ""
    assert calls[0].name == "roll_dice"
    assert calls[0].arguments == {"dice": "3d6", "times": 2}


def test_parse_keeps_surrounding_prose():
    text = f"我看看……\n{LEAKED}\n好了。"
    calls, cleaned, had = parse_dsml_content(text)
    assert had is True
    assert len(calls) == 1
    assert "我看看" in cleaned and "好了" in cleaned
    assert "DSML" not in cleaned


def test_unparsable_marker_returns_empty_content():
    text = "<｜｜DSML｜｜ calls>\n半截标记没有闭合"
    calls, cleaned, had = parse_dsml_content(text)
    assert had is True
    assert calls == []
    assert cleaned == ""


def test_passthrough_without_marker():
    calls, cleaned, had = parse_dsml_content("七点四十二，记住了。")
    assert (calls, cleaned, had) == ([], "七点四十二，记住了。", False)


class _Message:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Completions:
    def __init__(self, message):
        self._message = message

    async def create(self, **kwargs):
        return type("R", (), {"choices": [type("C", (), {"message": self._message})()]})()


def _provider(message):
    provider = OpenAICompatProvider(base_url="http://x", api_key="k", model="m")
    provider._client = type(
        "FakeClient", (), {"chat": type("Chat", (), {"completions": _Completions(message)})()}
    )()
    return provider


async def test_provider_parses_dsml_into_tool_calls():
    reply = await _provider(_Message(LEAKED)).generate_with_tools(
        [{"role": "user", "content": "x"}], [{"type": "function", "function": {"name": "set_reminder"}}]
    )
    assert reply.content == ""
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "set_reminder"


async def test_provider_drops_unparsable_dsml():
    reply = await _provider(_Message("<｜｜DSML｜｜ calls>\n坏标记")).generate_with_tools(
        [{"role": "user", "content": "x"}], []
    )
    assert reply.content == ""
    assert reply.tool_calls == []
