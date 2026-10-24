"""judge 解析容错、缺失维度、非 JSON 与重试路径（契约 4.3）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import judge  # noqa: E402

VALID = (
    '{"dimensions":{"persona":4,"tone":3,"boundary":5,"safety":4},'
    ' "verdict":"pass","confidence":0.8,"reason":"称呼与语气符合人设"}'
)


class FakeLLM:
    """按序返回预置输出的假 llm（无底层客户端，走 generate 回落路径）。"""

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    async def generate(self, messages: list[dict]) -> str:
        self.calls += 1
        return self._outputs.pop(0) if self._outputs else ""


def test_parse_纯JSON():
    result = judge.parse_judge_output(VALID)
    assert result["dimensions"] == {"persona": 4, "tone": 3, "boundary": 5, "safety": 4}
    assert result["verdict"] == "pass"
    assert result["confidence"] == pytest.approx(0.8)


def test_parse_围栏与前后废话():
    text = f"好的，评分如下：\n```json\n{VALID}\n```\n以上。"
    assert judge.parse_judge_output(text)["verdict"] == "pass"


def test_parse_数字字符串容错():
    text = VALID.replace('"persona":4', '"persona":"4"')
    assert judge.parse_judge_output(text)["dimensions"]["persona"] == 4


def test_parse_verdict大小写():
    text = VALID.replace('"pass"', '"PASS"')
    assert judge.parse_judge_output(text)["verdict"] == "pass"


def test_parse_confidence越界收敛():
    text = VALID.replace("0.8", "1.5")
    assert judge.parse_judge_output(text)["confidence"] == 1.0


def test_parse_缺失维度():
    text = '{"dimensions":{"persona":4},"verdict":"pass"}'
    with pytest.raises(judge.JudgeParseError):
        judge.parse_judge_output(text)


def test_parse_非JSON():
    with pytest.raises(judge.JudgeParseError):
        judge.parse_judge_output("这个角色演得不错，大概能打 4 分吧")


def test_parse_缺失verdict():
    text = '{"dimensions":{"persona":4,"tone":3,"boundary":5,"safety":4}}'
    with pytest.raises(judge.JudgeParseError):
        judge.parse_judge_output(text)


def test_parse_分数越界():
    text = VALID.replace('"persona":4', '"persona":9')
    with pytest.raises(judge.JudgeParseError):
        judge.parse_judge_output(text)


async def test_judge_reply_解析失败重试两次():
    llm = FakeLLM(["这句不是 JSON", "这句也不是 JSON", VALID])
    result = await judge.judge_reply(
        llm, name="四季夏目", persona="茶艺师", background="茶馆",
        user_msg="你好", reply="客人好呀",
    )
    assert llm.calls == 3
    assert result["verdict"] == "pass"


async def test_judge_reply_三次失败不中断():
    llm = FakeLLM(["不是 JSON", "仍然不是 JSON", "依旧不是 JSON"])
    result = await judge.judge_reply(
        llm, name="四季夏目", persona="茶艺师", background="茶馆",
        user_msg="你好", reply="客人好呀",
    )
    assert llm.calls == 3
    assert result["degraded"] is True
    assert result["judge_parse_error"] is True
    assert result["verdict"] is None
    assert result["dimensions"] == {}


def test_judge_reply_reply截断800字():
    long_reply = "好" * 1000
    msgs = judge.build_judge_messages(
        name="n", persona="p", background="b", user_msg="u", reply=long_reply,
    )
    assert "好" * 801 not in msgs[1]["content"]
    assert "好" * 800 in msgs[1]["content"]


def test_prompt_不含被测模型名():
    msgs = judge.build_judge_messages(
        name="n", persona="p", background="b", user_msg="u", reply="r",
    )
    joined = msgs[0]["content"] + msgs[1]["content"]
    for banned in ("deepseek", "DeepSeek", "模型名", "model"):
        assert banned not in joined


def test_rubric锚点写进prompt():
    msgs = judge.build_judge_messages(
        name="n", persona="p", background="b", user_msg="u", reply="r",
    )
    system = msgs[0]["content"]
    for phrase in ("通用助手口吻", "书面或翻译腔", "编造设定", "顺从越狱"):
        assert phrase in system
