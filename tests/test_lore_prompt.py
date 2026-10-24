"""T6：compose_prompt 的 lore 段与配额裁剪；recalled_lore 为空时与 V2.0 逐字一致。

- 快照断言：`recalled_lore=[]` 时输出 == V2.0（0a3938c）固定输出（零回归硬证据）；
- 段配额：always-on ≤ 8%、lore 合计 ≤ 20%；
- 全局裁剪顺序（契约 4.7）：① always-on → ② 摘要 → ③ 记忆 → ④ lore → ⑤ 轮次；
- retrieve_lore 节点：retriever 为空 / 关闭 → 空列表且不访问数据库。
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agents.graph import build_graph
from app.agents.nodes.compose_prompt import (
    LORE_HEADER,
    _estimate_tokens,
    _item_tokens,
    compose_prompt,
)
from app.agents.nodes.retrieve_lore import retrieve_lore
from app.lore.retriever import LoreHit, RetrievalResult
from app.memory.store import MemoryStore

# —— V2.0（0a3938c）在 recalled_lore 为空时的固定输出：逐字快照 ——
_V20_SNAPSHOT = [
    {
        "role": "system",
        "content": (
            "你是「一盏春」茶馆的老板，说话慢条斯理。\n\n"
            "【本会话摘要】\n前情提要：这位客人上周来过一次，聊起了茶价。\n\n"
            "【你记得的相关信息】\n- 客人偏爱茉莉花茶\n- 祖父留下的老匾还在墙上"
        ),
    },
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "来了，坐吧。"},
    {"role": "user", "content": "最近生意如何？"},
]


def _base_state() -> dict:
    return {
        "system_prompt": "你是「一盏春」茶馆的老板，说话慢条斯理。",
        "recalled_memories": ["客人偏爱茉莉花茶", "祖父留下的老匾还在墙上"],
        "summary": "前情提要：这位客人上周来过一次，聊起了茶价。",
        "messages": [
            HumanMessage(content="你好"),
            AIMessage(content="来了，坐吧。"),
            HumanMessage(content="最近生意如何？"),
        ],
        "recalled_lore": [],
    }


def _item(ref: str, header: str, content: str, score: float, routes=("vector",)) -> dict:
    return {
        "source_ref": ref,
        "header_path": header,
        "content": content,
        "score": score,
        "routes": list(routes),
    }


# ---- 零回归：recalled_lore 为空 ----


async def test_no_lore_matches_v20_snapshot():
    out = await compose_prompt(_base_state(), token_budget=8000, recent_rounds=10)
    assert out["prompt"] == _V20_SNAPSHOT
    assert LORE_HEADER not in out["prompt"][0]["content"]


async def test_missing_lore_key_also_matches_v20_snapshot():
    state = _base_state()
    del state["recalled_lore"]
    out = await compose_prompt(state, token_budget=8000, recent_rounds=10)
    assert out["prompt"] == _V20_SNAPSHOT


# ---- lore 段格式与顺序 ----


async def test_lore_section_format_and_keeps_input_order():
    lore = [
        _item("d:02", "茶馆 / 历史 / 祖父", "祖父在 1979 年盘下这间铺面。", 0.91),
        _item("d:07", "茶馆 / 招牌 / 一盏春", "招牌是一盏春的茉莉花茶。", 0.42),
    ]
    out = await compose_prompt(
        {**_base_state(), "recalled_lore": lore}, token_budget=8000
    )
    text = out["prompt"][0]["content"]
    assert text.startswith("你是「一盏春」")
    assert LORE_HEADER in text
    # 一行一个 chunk，前缀 `（header_path）content`，按传入（rerank 降序）顺序
    lo = text.index("（茶馆 / 历史 / 祖父）祖父在 1979 年盘下这间铺面。")
    hi = text.index("（茶馆 / 招牌 / 一盏春）招牌是一盏春的茉莉花茶。")
    assert lo < hi
    # lore 段在摘要/记忆之前
    assert text.index(LORE_HEADER) < text.index("【本会话摘要】")


# ---- 段配额 ----


async def test_lore_total_truncated_to_20_percent():
    lore = [
        _item("d:03", "甲", "正文" * 60, 0.5),  # 最低分 → 先丢
        _item("d:01", "乙", "正文" * 60, 0.9),
        _item("d:02", "丙", "正文" * 60, 0.7),
    ]
    cap = int(1000 * 0.20)
    assert sum(_item_tokens(i) for i in lore) > cap
    out = await compose_prompt(
        {**_base_state(), "recalled_lore": lore}, token_budget=1000
    )
    text = out["prompt"][0]["content"]
    assert "（乙）" in text
    assert "（甲）" not in text
    assert "（丙）" not in text
    injected = [i for i in lore if f"（{i['header_path']}）" in text]
    assert sum(_item_tokens(i) for i in injected) <= cap


async def test_always_on_truncated_to_8_percent_by_source_ref_desc():
    lore = [
        _item("d:01", "核", "常" * 57, 1.0, routes=("always_on",)),
        _item("d:02", "次", "常" * 57, 1.0, routes=("always_on",)),
    ]
    cap = int(1000 * 0.08)
    assert sum(_item_tokens(i) for i in lore) > cap  # 两条合计超 8%
    out = await compose_prompt(
        {**_base_state(), "recalled_lore": lore}, token_budget=1000
    )
    text = out["prompt"][0]["content"]
    assert "（核）" in text  # source_ref 小的保留
    assert "（次）" not in text  # source_ref 大的被丢


# ---- 全局裁剪顺序（契约 4.7）----


async def test_global_trim_drops_always_on_first():
    lore = [_item("d:01", "核", "很长的常驻设定" * 20, 1.0, routes=("always_on",))]
    state = {
        "system_prompt": "人设",
        "summary": "摘要",
        "recalled_memories": ["记忆"],
        "messages": [HumanMessage("问"), AIMessage("答")],
        "recalled_lore": lore,
    }
    # 抬高端配额上限，隔离出「全局超预算」这条路径；
    # 预算 60 只够去掉 lore 后的其余段落，故 ② 摘要 / ③ 记忆都不该被丢
    out = await compose_prompt(
        state, token_budget=60, lore_max_ratio=10.0, lore_always_on_ratio=10.0
    )
    text = out["prompt"][0]["content"]
    assert "很长的常驻设定" not in text  # ① always-on 先丢
    assert "【本会话摘要】" in text  # ② 摘要仍在
    assert "【你记得的相关信息】" in text  # ③ 记忆仍在


async def test_global_trim_order_summary_then_memories_then_lore():
    lore = [_item("d:01", "设", "设定" * 30, 0.9)]
    state = {
        "system_prompt": "人设",
        "summary": "摘要" * 30,
        "recalled_memories": ["记忆" * 30],
        "messages": [HumanMessage("问"), AIMessage("答")],
        "recalled_lore": lore,
    }
    out = await compose_prompt(
        state, token_budget=90, lore_max_ratio=10.0, lore_always_on_ratio=10.0
    )
    text = out["prompt"][0]["content"]
    assert "【本会话摘要】" not in text  # ② 先丢摘要
    assert "【你记得的相关信息】" not in text  # ③ 再丢记忆
    assert "（设）设定" in text  # ④ lore 尚未被丢
    assert _estimate_tokens(text) <= 90


async def test_global_trim_drops_lore_before_rounds():
    lore = [_item("d:01", "设", "设定" * 30, 0.9)]
    msgs = []
    for i in range(5):
        msgs.append(HumanMessage(content=f"问{i}"))
        msgs.append(AIMessage(content=f"答{i}"))
    state = {
        "system_prompt": "人设",
        "summary": "",
        "recalled_memories": [],
        "messages": msgs,
        "recalled_lore": lore,
    }
    out = await compose_prompt(
        state, token_budget=20, lore_max_ratio=10.0, lore_always_on_ratio=10.0
    )
    assert "（设）设定" not in out["prompt"][0]["content"]  # ④ lore 在轮次之前被丢
    assert len(out["prompt"]) > 1  # 仍保留至少一条消息（⑤ 保底）


# ---- retrieve_lore 节点 ----


class _ExplodingRetriever:
    async def retrieve(self, **kwargs):  # pragma: no cover - 不该被调用
        raise AssertionError("retriever 不该被访问")


async def test_retrieve_lore_disabled_does_not_touch_db():
    state = {"agent_id": 1, "messages": [HumanMessage(content="问")]}
    assert await retrieve_lore(state, _ExplodingRetriever(), enabled=False) == {
        "recalled_lore": []
    }


async def test_retrieve_lore_none_retriever_returns_empty():
    state = {"agent_id": 1, "messages": [HumanMessage(content="问")]}
    assert await retrieve_lore(state, None) == {"recalled_lore": []}


async def test_retrieve_lore_maps_injected_hits():
    hit = LoreHit(
        source_ref="tea_house:07",
        header_path="茶馆 / 历史 / 祖父",
        content="祖父在 1979 年盘下这间铺面。",
        token_count=18,
        score=0.91,
        routes=("keyword", "vector"),
    )
    result = RetrievalResult(
        query="她祖父开的茶馆",
        rewritten="祖父 茶馆",
        rewrite_used=True,
        routes={"always_on": [], "keyword": ["tea_house:07"], "vector": ["tea_house:07"]},
        fused=[("tea_house:07", 0.032)],
        reranked=[("tea_house:07", 0.91)],
        rerank=True,
        injected=[hit],
        lore_tokens=18,
        lore_cap=1600,
    )

    class _FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def retrieve(self, **kwargs):
            self.calls.append(kwargs)
            return result

    fake = _FakeRetriever()
    out = await retrieve_lore(
        {"agent_id": 7, "messages": [HumanMessage(content="她祖父开的茶馆")]}, fake
    )
    assert out["recalled_lore"] == [hit.as_recalled()]
    assert fake.calls[0]["agent_id"] == 7
    assert fake.calls[0]["query"] == "她祖父开的茶馆"


async def test_retrieve_lore_without_user_message_still_queries_always_on():
    class _FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def retrieve(self, **kwargs):
            self.calls.append(kwargs)
            return RetrievalResult(
                query="", rewritten="", rewrite_used=False,
                routes={"always_on": [], "keyword": [], "vector": []},
                fused=[], reranked=[], rerank=False,
            )

    fake = _FakeRetriever()
    out = await retrieve_lore({"agent_id": 3, "messages": []}, fake)
    assert out == {"recalled_lore": []}
    assert fake.calls[0]["query"] == ""  # 闲聊轮 query 为空串，仍走 always-on 路


# ---- 图集成 ----


class _FakeLLM:
    def __init__(self, reply: str = "好的，客人。") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def generate(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply

    async def astream(self, messages: list[dict]):
        self.calls.append(messages)
        yield self.reply


def _graph_state(message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "agent_id": 1,
        "session_id": 1,
        "agent_name": "四季夏目",
        "persona_prompt": "茶艺师，说话温柔。",
        "background": "江南古镇「一盏春」",
    }


class _StubRetriever:
    def __init__(self, injected: list[LoreHit]) -> None:
        self._injected = injected

    async def retrieve(self, **kwargs):
        return RetrievalResult(
            query=kwargs.get("query", ""),
            rewritten=kwargs.get("query", ""),
            rewrite_used=False,
            routes={"always_on": [], "keyword": [], "vector": []},
            fused=[],
            reranked=[],
            rerank=False,
            injected=list(self._injected),
        )


def _lore_graph(llm, retriever):
    store = InMemoryStore()
    return build_graph(
        llm=llm,
        memory_store=MemoryStore(store),
        checkpointer=InMemorySaver(),
        store=store,
        lore_retriever=retriever,
    )


async def test_graph_without_retriever_injects_no_lore():
    llm = _FakeLLM()
    graph = _lore_graph(llm, None)
    await graph.ainvoke(_graph_state("你好"), config={"configurable": {"thread_id": "L:1"}})
    assert LORE_HEADER not in llm.calls[0][0]["content"]


async def test_graph_with_retriever_injects_lore_block():
    hit = LoreHit(
        source_ref="tea_house:07",
        header_path="茶馆 / 历史 / 祖父",
        content="祖父在 1979 年盘下这间铺面。",
        token_count=18,
        score=0.91,
        routes=("vector",),
    )
    llm = _FakeLLM()
    graph = _lore_graph(llm, _StubRetriever([hit]))
    await graph.ainvoke(_graph_state("她祖父开过茶馆吗"), config={"configurable": {"thread_id": "L:2"}})
    system_content = llm.calls[0][0]["content"]
    assert LORE_HEADER in system_content
    assert "（茶馆 / 历史 / 祖父）祖父在 1979 年盘下这间铺面。" in system_content
