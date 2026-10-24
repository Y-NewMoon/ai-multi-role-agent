"""V2.3 prompt 注入（契约 §2.8）：三段顺序、空段逐字一致、预算与裁剪。"""

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.nodes.compose_prompt import compose_prompt
from app.agents.nodes.update_memory import update_memory
from app.memory.store import MemoryStore
from langgraph.store.memory import InMemoryStore


def _state(**overrides) -> dict:
    state = {
        "messages": [HumanMessage(content="你好")],
        "agent_id": 1,
        "session_id": 1,
        "agent_name": "四季夏目",
        "persona_prompt": "茶艺师，说话温柔。",
        "background": "江南古镇「一盏春」",
        "system_prompt": "人设：茶艺师。",
        "recalled_memories": [],
        "summary": "",
    }
    state.update(overrides)
    return state


async def _system(state: dict, **kwargs) -> str:
    result = await compose_prompt(state, **kwargs)
    return result["prompt"][0]["content"]


async def test_三段为空时与V22逐字一致():
    state = _state(recalled_memories=["甲"])
    assert await _system(state) == "人设：茶艺师。\n\n【你记得的相关信息】\n- 甲"
    # 显式空字段（extract 关闭时 retrieve_memory 不写这些键）也不得插入字符
    state["user_profile"] = {}
    state["recalled_facts"] = []
    state["l2_portrait"] = ""
    assert await _system(state) == "人设：茶艺师。\n\n【你记得的相关信息】\n- 甲"


async def test_三段顺序固定():
    state = _state(
        summary="摘要内容",
        recalled_memories=["记忆甲"],
        user_profile={"称呼": "阿林"},
        recalled_facts=[
            {"subject": "用户", "predicate": "喜欢的茶", "object": "普洱",
             "importance": 0.9, "updated_at": "2026-10-21T00:00:00+00:00"}
        ],
        l2_portrait="我们之间有过一段茶时光。",
    )
    text = await _system(state)
    order = ["人设：茶艺师。", "【客人档案】", "【本会话摘要】", "【你记得的相关信息】", "【你记得的事】", "【你和这位客人的过往】"]
    positions = [text.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "- 称呼：阿林" in text
    assert "- 用户喜欢的茶=普洱" in text


async def test_召回段不得再加前缀():
    state = _state(recalled_memories=["用户说过：我叫阿飞", "你说过：我给你倒茶"])
    text = await _system(state)
    assert "用户说过：我叫阿飞" in text
    assert "你说过：我给你倒茶" in text
    assert "用户说过：用户说过" not in text


async def test_profile段预算十分之一():
    state = _state(user_profile={"称呼": "阿" * 400})
    text = await _system(state, token_budget=100)
    assert "【客人档案】" not in text or "阿" * 200 not in text


async def test_三段合计不超过四分之一预算():
    state = _state(
        user_profile={"称呼": "阿林", "禁忌": "不喝浓茶"},
        recalled_facts=[
            {"subject": "用户", "predicate": f"p{i}", "object": "o" * 20,
             "importance": 0.5, "updated_at": f"2026-10-{10 + i:02d}T00:00:00+00:00"}
            for i in range(8)
        ],
        l2_portrait="关" * 300,
    )
    text = await _system(state, token_budget=200)
    # 200 * 25% = 50 token：L2 必然被裁掉，facts 也应大幅收缩
    assert "关" * 300 not in text
    assert text.count("- 用户p") <= 4


async def test_第十五轮信息由摘要兜住():
    # D9：summary_trigger_rounds=10 时第 11 轮起滚动摘要；近 10 轮窗口外的第 1 轮
    # 关键内容必须已在摘要里（compose 只看近 10 轮）。
    messages = []
    for i in range(1, 16):
        messages.append(HumanMessage(content=f"第{i}句"))
        messages.append(AIMessage(content=f"回应{i}"))
    state = _state(messages=messages, summary="用户第一轮说：我叫阿飞")
    result = await compose_prompt(state, recent_rounds=10)
    text = result["prompt"][0]["content"]
    assert "【本会话摘要】" in text
    assert "我叫阿飞" in text
    joined = "\n".join(m["content"] for m in result["prompt"])
    assert "第1句" not in joined  # 已掉出近轮窗口
    assert "第15句" in joined


async def test_摘要阈值默认10并触发滚动摘要():
    class SummaryLLM:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []

        async def generate(self, messages: list[dict]) -> str:
            self.calls.append(messages)
            return "用户第一轮说：我叫阿飞"

    store = MemoryStore(InMemoryStore())
    llm = SummaryLLM()
    messages = []
    for i in range(1, 12):
        messages.append(HumanMessage(content=f"第{i}句"))
        messages.append(AIMessage(content=f"回应{i}"))
    state = _state(messages=messages, replies=["回应11"])
    await update_memory(state, store, llm, trigger_rounds=10, chunk_rounds=10)
    assert await store.get_summary(1, 1) == "用户第一轮说：我叫阿飞"


async def test_extract关闭时不调LLM不写新条目():
    class CountingLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, messages: list[dict]) -> str:
            self.calls += 1
            return ""

    store = MemoryStore(InMemoryStore())
    llm = CountingLLM()
    state = _state(replies=["你好呀"])
    await update_memory(state, store, llm, extract_enabled=False)
    assert llm.calls == 0
    assert await store.current_facts(1) == []
    assert await store.get_profile(1) == {}
    assert await store.get_l2(1) == ""
    assert await store.get_l2_counter(1) == 0


async def test_extract关闭时11轮摘要不写L2计数():
    """R8：`l2_enabled` 默认 True，但 extract 关闭时必须强制停（不变量 7）。"""

    class CountingLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, messages: list[dict]) -> str:
            self.calls += 1
            return "摘要桩"

    store = MemoryStore(InMemoryStore())
    llm = CountingLLM()
    messages = []
    for i in range(1, 12):
        messages.append(HumanMessage(content=f"第{i}句"))
        messages.append(AIMessage(content=f"回应{i}"))
    state = _state(messages=messages, replies=["回应11"])

    await update_memory(state, store, llm, trigger_rounds=10, chunk_rounds=10, extract_enabled=False)

    assert llm.calls == 1  # 只有摘要，没有抽取、没有 L2 重合成
    assert await store.get_summary(1, 1) == "摘要桩"
    assert await store.get_l2_counter(1) == 0
    assert await store.get_l2(1) == ""
