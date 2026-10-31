"""V2.3 事实抽取（契约 §2.3 / §8.1 / §8.2）。"""

import asyncio
import json

import pytest

from app.memory.extractor import MAX_FACTS, Extraction, extract, parse_extraction

VALID = json.dumps(
    {
        "facts": [
            {"subject": "用户", "predicate": "喜欢的茶", "object": "普洱", "importance": 4},
            {"subject": "用户", "predicate": "养了猫", "object": "叫团子", "importance": 5},
        ],
        "profile": {"称呼": "阿林"},
    },
    ensure_ascii=False,
)


class FakeLLM:
    """按序返回预置输出，或抛出预置异常。"""

    def __init__(self, outputs: list[object]) -> None:
        self._outputs = list(outputs)
        self.calls: list[list[dict]] = []

    async def generate(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        item = self._outputs.pop(0) if self._outputs else ""
        if isinstance(item, Exception):
            raise item
        return str(item)


async def test_extract_正常输出():
    llm = FakeLLM([VALID])
    result = await extract(llm, "我改喝普洱了", "那我把龙井收起来")
    assert isinstance(result, Extraction)
    assert [f.object for f in result.facts] == ["叫团子", "普洱"]  # importance 降序
    assert result.profile == {"称呼": "阿林"}


async def test_extract_空回复返回空事实():
    llm = FakeLLM(['{"facts":[],"profile":{}}'])
    result = await extract(llm, "你好", "你好呀")
    assert result is not None
    assert result.facts == [] and result.profile == {}


async def test_extract_非法JSON重试一次后返回None():
    llm = FakeLLM(["这不是 JSON", "仍然不是"])
    assert await extract(llm, "我改喝普洱了", "好的") is None
    assert len(llm.calls) == 2  # 首次 + 重试 1 次


async def test_extract_第一次非法第二次合法():
    llm = FakeLLM(["不是 JSON", VALID])
    result = await extract(llm, "我改喝普洱了", "好的")
    assert result is not None and len(result.facts) == 2
    assert len(llm.calls) == 2


async def test_extract_LLM异常fail_open():
    llm = FakeLLM([RuntimeError("500"), VALID])
    result = await extract(llm, "我改喝普洱了", "好的")
    assert result is not None and len(result.facts) == 2


async def test_extract_超时不抛出():
    class SlowLLM:
        async def generate(self, messages):
            await asyncio.sleep(0.2)
            return VALID

    assert await extract(SlowLLM(), "我改喝普洱了", "好的", timeout_s=0.01) is None


async def test_extract_输入超长截断到2000字():
    llm = FakeLLM([VALID])
    await extract(llm, "长" * 5000, "回" * 5000)
    user_prompt = llm.calls[0][-1]["content"]
    conversation = user_prompt.split("对话：\n", 1)[1].split("\n\n当前档案", 1)[0]
    assert len(conversation) <= 2000


def test_parse_subject白名单丢弃():
    text = json.dumps(
        {
            "facts": [
                {"subject": "猫", "predicate": "名字", "object": "团子", "importance": 4},
                {"subject": "用户", "predicate": "养了猫", "object": "团子", "importance": 4},
            ]
        },
        ensure_ascii=False,
    )
    result = parse_extraction(text)
    assert result is not None
    assert [f.subject for f in result.facts] == ["用户"]


def test_parse_占位词丢弃():
    text = json.dumps(
        {
            "facts": [
                {"subject": "用户", "predicate": "名字", "object": "不知道", "importance": 3},
                {"subject": "用户", "predicate": "名字", "object": "", "importance": 3},
                {"subject": "用户", "predicate": "名字", "object": "阿林", "importance": 3},
            ]
        },
        ensure_ascii=False,
    )
    result = parse_extraction(text)
    assert result is not None
    assert [f.object for f in result.facts] == ["阿林"]


def test_parse_最多5条按importance截断():
    facts = [
        {"subject": "用户", "predicate": f"喜好{i}", "object": f"值{i}", "importance": i}
        for i in range(1, 8)
    ]
    result = parse_extraction(json.dumps({"facts": facts}, ensure_ascii=False))
    assert result is not None
    assert len(result.facts) == MAX_FACTS == 5
    assert [f.importance for f in result.facts] == [5, 5, 5, 4, 3]


def test_parse_importance越界收敛():
    text = json.dumps(
        {"facts": [{"subject": "用户", "predicate": "p", "object": "o", "importance": 9}]},
        ensure_ascii=False,
    )
    result = parse_extraction(text)
    assert result is not None and result.facts[0].importance == 5


def test_parse_谓词对象长度上限():
    text = json.dumps(
        {"facts": [{"subject": "用户", "predicate": "谓" * 30, "object": "宾" * 60}]},
        ensure_ascii=False,
    )
    result = parse_extraction(text)
    assert result is not None
    assert len(result.facts[0].predicate) == 20
    assert len(result.facts[0].object) == 40


def test_parse_去掉首尾标点空白():
    text = json.dumps(
        {"facts": [{"subject": "用户", "predicate": " 喜欢的茶。", "object": "「普洱」"}]},
        ensure_ascii=False,
    )
    result = parse_extraction(text)
    assert result is not None
    assert result.facts[0].predicate == "喜欢的茶"
    assert result.facts[0].object == "普洱"


def test_parse_profile白名单与截断():
    text = json.dumps(
        {
            "facts": [],
            "profile": {"称呼": "阿" * 60, "未知槽": "x", "禁忌": "不喝浓茶"},
        },
        ensure_ascii=False,
    )
    result = parse_extraction(text)
    assert result is not None
    assert set(result.profile) == {"称呼", "禁忌"}
    assert len(result.profile["称呼"]) == 40


def test_parse_profile_在标点处截断不切半个引号():
    long = "不要称呼角色为“宝宝”，她不喜欢被这么叫，也不要叫小肥鱼"
    text = json.dumps({"facts": [], "profile": {"禁忌": long}}, ensure_ascii=False)
    result = parse_extraction(text)
    assert result is not None
    value = result.profile["禁忌"]
    assert len(value) <= 40
    assert value.count("“") == value.count("”")


def test_parse_profile_无标点补省略号():
    text = json.dumps({"facts": [], "profile": {"称呼": "阿" * 60}}, ensure_ascii=False)
    result = parse_extraction(text)
    assert result is not None
    value = result.profile["称呼"]
    assert len(value) == 40
    assert value.endswith("…")


def test_parse_profile_空串表示撤销():
    text = json.dumps({"facts": [], "profile": {"禁忌": ""}}, ensure_ascii=False)
    result = parse_extraction(text)
    assert result is not None
    assert result.profile == {"禁忌": ""}


def test_parse_profile_占位词也当撤销():
    text = json.dumps({"facts": [], "profile": {"禁忌": "无"}}, ensure_ascii=False)
    result = parse_extraction(text)
    assert result is not None
    assert result.profile == {"禁忌": ""}


async def test_extract_把当前档案带进提示词():
    llm = FakeLLM([json.dumps({"facts": [], "profile": {"禁忌": ""}}, ensure_ascii=False)])
    result = await extract(
        llm, "以后可以叫我宝宝了", "行吧", current_profile={"禁忌": "不要叫宝宝"}
    )
    assert result is not None
    assert result.profile == {"禁忌": ""}
    prompt = llm.calls[0][1]["content"]
    assert "不要叫宝宝" in prompt
    assert "空字符串" in prompt


def test_parse_围栏与前后废话():
    text = f"好的，结果如下：\n```json\n{VALID}\n```\n以上。"
    result = parse_extraction(text)
    assert result is not None and len(result.facts) == 2


@pytest.mark.parametrize("bad", ["", "[]", '{"facts": "x"}'])
def test_parse_结构错误返回None(bad: str):
    assert parse_extraction(bad) is None
