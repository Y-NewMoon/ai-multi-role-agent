"""V2.3 L2 层级摘要（契约 §2.7）。"""

from langgraph.store.memory import InMemoryStore

from app.memory.l2 import force_update_l2, maybe_update_l2
from app.memory.store import MemoryStore


def _store() -> MemoryStore:
    return MemoryStore(InMemoryStore())


class FakeLLM:
    def __init__(self, reply: str = "我们之间有过一段安静的茶时光。", *, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[list[dict]] = []

    async def generate(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.reply


async def test_未达阈值不合成():
    store, llm = _store(), FakeLLM()
    assert await maybe_update_l2(store, 1, llm, current_l1="L1", every_merges=3) is None
    assert await maybe_update_l2(store, 1, llm, current_l1="L1", every_merges=3) is None
    assert llm.calls == []
    assert await store.get_l2(1) == ""
    assert await store.get_l2_counter(1) == 2  # 计数照常推进


async def test_达阈值合成并归零():
    store, llm = _store(), FakeLLM()
    await maybe_update_l2(store, 1, llm, current_l1="第一段", every_merges=2)
    result = await maybe_update_l2(store, 1, llm, current_l1="第二段", every_merges=2)

    assert result == "我们之间有过一段安静的茶时光。"
    assert await store.get_l2(1) == result
    assert await store.get_l2_counter(1) == 0
    prompt = llm.calls[0][-1]["content"]
    assert "第二段" in prompt


async def test_合成失败不归零也不写L2():
    store = _store()
    llm = FakeLLM(error=RuntimeError("502"))
    assert await maybe_update_l2(store, 1, llm, current_l1="L1", every_merges=1) is None
    assert await store.get_l2(1) == ""
    assert await store.get_l2_counter(1) == 1  # 不归零，下次重试


async def test_空输出不归零():
    store = _store()
    llm = FakeLLM("   ")
    assert await maybe_update_l2(store, 1, llm, current_l1="L1", every_merges=1) is None
    assert await store.get_l2(1) == ""
    assert await store.get_l2_counter(1) == 1


async def test_超过300字截断():
    store = _store()
    llm = FakeLLM("长" * 500)
    result = await maybe_update_l2(store, 1, llm, current_l1="L1", every_merges=1)
    assert result is not None and len(result) == 300
    assert len(await store.get_l2(1)) == 300


async def test_force跳过计数立即合成():
    store, llm = _store(), FakeLLM()
    assert await store.get_l2_counter(1) == 0
    result = await force_update_l2(store, 1, llm, current_l1="")
    assert result is not None and await store.get_l2(1) == result
    assert len(llm.calls) == 1


async def test_prompt包含现有L2_L1与facts():
    store, llm = _store(), FakeLLM()
    await store.put_l2(1, "他第一次来时点了龙井。")
    await store.put_fact(1, subject="用户", predicate="称呼", obj="阿林", importance=1.0)
    await force_update_l2(store, 1, llm, current_l1="今晚聊了茶")
    prompt = llm.calls[0][-1]["content"]
    assert "他第一次来时点了龙井。" in prompt
    assert "今晚聊了茶" in prompt
    assert "用户称呼=阿林" in prompt


def test_prompt要求场合区分():
    from app.prompts.l2_summary import L2_PROMPT

    assert "在群里" in L2_PROMPT
    assert "私聊与群聊是同一段关系" in L2_PROMPT


async def test_计数与L2按agent隔离():
    store, llm = _store(), FakeLLM()
    await maybe_update_l2(store, 1, llm, current_l1="L1", every_merges=3)
    assert await store.get_l2_counter(1) == 1
    assert await store.get_l2_counter(2) == 0
    assert await store.get_l2(2) == ""
