"""图编排单测：用 fake LLM + 内存 checkpointer/store 跑通 5 节点（接缝 3 / 4 / 5）。"""

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agents.graph import build_graph
from app.agents.nodes.generate import normalize_thinkings, split_messages
from app.memory.store import MemoryStore
from app.prompts.roleplay_system import MESSAGE_DELIMITER as D
from app.prompts.roleplay_system import THINK_DELIMITER as T
from app.prompts.roleplay_system import THINK_MAX_CHARS

_DIMS = 1024


class FakeEmbeddings(Embeddings):
    """固定向量：让 InMemoryStore 的向量索引可跑（无需真模型）。"""

    def _vec(self) -> list[float]:
        return [0.1] * _DIMS

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec() for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec()


class FakeLLM:
    """记录收到的 prompt，返回固定回复，替代真实 LLM（M3 起走 astream）。"""

    def __init__(self, reply: str = "好的，客人。") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def generate(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply

    async def astream(self, messages: list[dict]):
        self.calls.append(messages)
        yield self.reply


class _CharLLM(FakeLLM):
    """逐字符产出：验证分隔符/思考标记被拆分在多个流式增量之间仍能正确解析。"""

    async def astream(self, messages: list[dict]):
        self.calls.append(messages)
        for ch in self.reply:
            yield ch


def _state(message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "agent_id": 1,
        "session_id": 1,
        "agent_name": "四季夏目",
        "persona_prompt": "茶艺师，说话温柔。",
        "background": "江南古镇「一盏春」",
    }


def _store() -> InMemoryStore:
    # M2：检索带 query，InMemoryStore 必须显式配置向量索引
    return InMemoryStore(index={"dims": _DIMS, "embed": FakeEmbeddings(), "fields": ["text"]})


def _graph(llm, store):
    return build_graph(
        llm=llm,
        memory_store=MemoryStore(store),
        checkpointer=InMemorySaver(),
        store=store,
    )


async def test_graph_writes_persona_prompt_and_memory():
    llm = FakeLLM("这是我给客人的回话。")
    store = _store()
    graph = _graph(llm, store)

    result = await graph.ainvoke(_state("你好呀"), config={"configurable": {"thread_id": "1:1"}})

    assert result["reply"] == "这是我给客人的回话。"
    # prompt 首条是注入了人设的 system
    assert llm.calls[0][0]["role"] == "system"
    assert "四季夏目" in llm.calls[0][0]["content"]
    # 记忆按 (agent_id, content) 落库：本轮 user + assistant 各一条
    items = await store.asearch(("agent:1", "memories"))
    texts = {item.value["text"] for item in items}
    assert "你好呀" in texts
    assert "这是我给客人的回话。" in texts


async def test_graph_resumes_history_across_turns():
    llm = FakeLLM()
    store = _store()
    graph = _graph(llm, store)
    cfg = {"configurable": {"thread_id": "9:9"}}

    await graph.ainvoke(_state("第一句"), config=cfg)
    await graph.ainvoke(_state("第二句"), config=cfg)

    second_prompt = " ".join(m["content"] for m in llm.calls[-1])
    assert "第一句" in second_prompt  # 历史被 checkpointer 续上
    assert "第二句" in second_prompt


async def test_graph_reuses_recalled_memory_in_prompt():
    llm = FakeLLM()
    store = _store()
    graph = _graph(llm, store)
    cfg = {"configurable": {"thread_id": "5:5"}}

    await graph.ainvoke(_state("我叫阿飞"), config=cfg)
    await graph.ainvoke(_state("你还记得我吗"), config=cfg)

    system_content = llm.calls[-1][0]["content"]
    assert "你记得的相关信息" in system_content  # 上一轮记忆被召回拼进 system


def test_split_messages_caps_and_drops_empty():
    assert split_messages("单条") == ["单条"]
    assert split_messages(f"甲{D}乙{D}丙") == ["甲", "乙", "丙"]
    assert split_messages(f"甲{D}{D}乙") == ["甲", "乙"]  # 空条被丢弃
    assert split_messages(f"1{D}2{D}3{D}4") == ["1", "2", "3\n4"]  # 超上限合并进最后一条
    assert split_messages("") == [""]


async def test_graph_splits_multiple_messages_and_memorizes_each():
    # M5：角色连发多条 → 图产出 replies 列表，且每条各落一条记忆
    llm = FakeLLM(f"第一句{D}第二句")
    store = _store()
    graph = _graph(llm, store)

    result = await graph.ainvoke(_state("你好呀"), config={"configurable": {"thread_id": "3:3"}})

    assert result["replies"] == ["第一句", "第二句"]
    assert result["reply"] == "第一句\n第二句"
    texts = {item.value["text"] for item in await store.asearch(("agent:1", "memories"))}
    assert "第一句" in texts
    assert "第二句" in texts
    assert "你好呀" in texts


async def test_graph_triggers_summary_and_trims_working_memory():
    # 方案 3.10：超过 trigger_rounds 轮 → 压缩最旧 chunk_rounds 轮并裁剪工作记忆窗口
    llm = FakeLLM("回话")
    store = _store()
    graph = build_graph(
        llm=llm,
        memory_store=MemoryStore(store),
        checkpointer=InMemorySaver(),
        store=store,
        summary_trigger_rounds=2,
        summary_chunk_rounds=1,
    )
    cfg = {"configurable": {"thread_id": "7:7"}}

    await graph.ainvoke(_state("第一句"), config=cfg)
    await graph.ainvoke(_state("第二句"), config=cfg)
    await graph.ainvoke(_state("第三句"), config=cfg)

    items = await store.asearch(("agent:1", "memories"))
    summaries = [i for i in items if i.value.get("kind") == "summary"]
    assert summaries, "应写入一条滚动摘要"
    assert summaries[0].key == "summary:1"  # key = summary:{session_id}

    contents = [m.content for m in graph.get_state(cfg).values["messages"]]
    assert "第一句" not in contents  # RemoveMessage 已裁剪最旧一轮
    assert "第三句" in contents


# ===== M6：思考链（内心独白）=====

def test_normalize_thinkings_truncates_and_aligns():
    long = "思" * (THINK_MAX_CHARS + 10)
    assert normalize_thinkings([long], 1) == ["思" * THINK_MAX_CHARS]  # 超长截断
    assert normalize_thinkings(["a", "b", "c"], 2) == ["a", "b"]  # 多余丢弃
    assert normalize_thinkings(["a"], 3) == ["a", "", ""]  # 不足补空
    assert normalize_thinkings([], 2) == ["", ""]
    assert normalize_thinkings(None, 1) == [""]
    assert normalize_thinkings([" 空格 "], 1) == ["空格"]  # 去空白


async def test_graph_parses_thinkings_aligned_to_replies():
    # 交错格式：每条消息后紧跟 THINK 标记 + 该条内心独白，MSG 分隔下一条 → 解析为 thinkings
    llm = FakeLLM(f"第一句{T}心里有点紧张{D}第二句{T}其实挺开心")
    store = _store()
    graph = _graph(llm, store)

    result = await graph.ainvoke(_state("你好呀"), config={"configurable": {"thread_id": "20:20"}})

    assert result["replies"] == ["第一句", "第二句"]
    assert result["thinkings"] == ["心里有点紧张", "其实挺开心"]
    assert T not in result["reply"]  # 思考不得混入正文


async def test_graph_parses_thinkings_when_markers_split_across_deltas():
    # 增量逐字符到达：标记被拆散在多个增量之间也必须正确切分
    llm = _CharLLM(f"甲{T}想甲{D}乙{T}想乙")
    store = _store()
    graph = _graph(llm, store)

    result = await graph.ainvoke(_state("hi"), config={"configurable": {"thread_id": "21:21"}})

    assert result["replies"] == ["甲", "乙"]
    assert result["thinkings"] == ["想甲", "想乙"]


async def test_graph_without_think_section_yields_empty_thinkings():
    llm = FakeLLM("只有这一句")
    store = _store()
    graph = _graph(llm, store)

    result = await graph.ainvoke(_state("hi"), config={"configurable": {"thread_id": "22:22"}})

    assert result["thinkings"] == [""]  # 无 THINK 段 → 静默降级为空（前端不显示按钮）


async def test_graph_aligns_thinkings_count_to_replies():
    # 交错格式下思考天然按条绑定：缺思考的那条补空，多余思考只取首行
    llm = FakeLLM(f"第一句{T}想一{D}第二句")
    store = _store()
    graph = _graph(llm, store)

    result = await graph.ainvoke(_state("hi"), config={"configurable": {"thread_id": "23:23"}})

    assert result["replies"] == ["第一句", "第二句"]
    assert result["thinkings"] == ["想一", ""]


async def test_graph_stream_does_not_leak_think_text():
    # 流式期间思考绝不透出：token 里不得出现 THINK 标记或思考内容
    llm = _CharLLM(f"公开的话{T}不能说的内心话")
    store = _store()
    graph = _graph(llm, store)

    tokens: list[str] = []
    thinkings: list[str] = []
    async for mode, chunk in graph.astream(
        _state("hi"),
        config={"configurable": {"thread_id": "24:24"}},
        stream_mode=["custom", "values"],
    ):
        if mode == "custom" and isinstance(chunk, dict) and "text" in chunk:
            tokens.append(str(chunk["text"]))
        elif mode == "values" and chunk.get("thinkings") is not None:
            thinkings = [str(t) for t in chunk["thinkings"]]

    streamed = "".join(tokens)
    assert "公开的话" in streamed
    assert "不能说的内心话" not in streamed
    assert T not in streamed
    assert thinkings == ["不能说的内心话"]

