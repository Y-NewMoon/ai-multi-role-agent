"""V2.1 检索测试（T4）：RRF 公式、并列排序、预算截断、空库、停用词跳过关键词路、preview 结构。

纯函数用 FakeEmbeddings + NullReranker；真库用例需 `alembic upgrade head`，库不可用时自动跳过。
"""

import asyncio
import hashlib

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import delete, select

from app.agents.nodes.retrieve_lore import _history, retrieve_lore
from app.db.models import Agent, LoreChunk, LoreDoc
from app.lore.indexer import Indexer
from app.lore.retriever import (
    LlmRewriter,
    NullReranker,
    Retriever,
    _fuse,
    _keyword_terms,
    needs_rewrite,
)

_TITLE = "一盏春茶馆"
_HIST = "祖父在民国年间盘下这间铺面，取名一盏春茶馆，从此以茶为生。" * 3
_SIGN = "招牌是一盏春茶馆的茉莉花茶，客人来了先奉一盏。" * 3
_CORE = "本店规矩：进门先问好，落座必奉茶，不谈价钱先叙情分。" * 3


def _content(hist: str = _HIST, sign: str = _SIGN) -> str:
    return f"# 一盏春\n\n## 历史\n{hist}\n\n## 招牌\n{sign}\n"


class _FakeEmbedder(Embeddings):
    def __init__(self, dims: int = 1024) -> None:
        self._dims = dims

    def _vec(self, one: str) -> list[float]:
        digest = hashlib.sha256(one.encode("utf-8")).digest()
        raw = (digest * ((self._dims // len(digest)) + 1))[: self._dims]
        return [b / 255.0 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, one: str) -> list[float]:
        return self._vec(one)


class _EchoRewriter:
    async def rewrite(self, query: str, history: list[str]) -> str:
        return "祖父 茶馆 名字"


class _SlowRewriter:
    async def rewrite(self, query: str, history: list[str]) -> str:
        await asyncio.sleep(0.3)
        return "不该被用到的改写"


class _RecordingRewriter:
    """记录 `rewrite()` 收到的 (query, history)，供节点级测试断言。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def rewrite(self, query: str, history: list[str]) -> str:
        self.calls.append((query, list(history)))
        return "祖父 茶馆 名字"


class _CapturingLlm:
    """无 `_client` 的假 llm：走 `generate` 分支并留存发给模型的 messages。"""

    def __init__(self) -> None:
        self.messages: list[dict] | None = None

    async def generate(self, messages: list[dict]) -> str:
        self.messages = messages
        return "改写结果"


def _retriever(session_factory, **kwargs) -> Retriever:
    params = {"reranker": NullReranker(), "embedder": _FakeEmbedder()}
    params.update(kwargs)
    return Retriever(session_factory=session_factory, **params)


# ---- 纯函数 ----

def test_rrf_formula_and_tie_order() -> None:
    """RRF 公式 `1/(60+rank)`；两路同分时按 source_ref 升序（契约 4.5）。"""
    routes = {"keyword": ["a:01", "a:02"], "vector": ["a:02", "a:01"]}
    fused = _fuse(routes)
    assert [ref for ref, _ in fused] == ["a:01", "a:02"]  # 同分并列 → source_ref 升序
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)

    single = _fuse({"keyword": ["a:01"], "vector": ["a:01", "a:02"]})
    assert single[0][0] == "a:01"
    assert single[0][1] == pytest.approx(2 / 61)
    assert single[1][1] == pytest.approx(1 / 62)


def test_keyword_terms_drops_single_char() -> None:
    """查询侧分词：丢弃 < 2 字的词与标点（契约 4.4）。"""
    assert _keyword_terms("的了。，") == []
    terms = _keyword_terms("祖父的茶馆")
    assert "祖父" in terms and "茶馆" in terms and "的" not in terms


def test_needs_rewrite_rules() -> None:
    """含代词、过短，或上一轮有未消解指代才触发改写（契约 4.6）。"""
    assert needs_rewrite("她祖父开的茶馆叫什么") is True  # 含代词
    assert needs_rewrite("那家店") is True  # 含代词
    assert needs_rewrite("多少钱") is True  # 过短（len<8）
    assert needs_rewrite("祖父开的茶馆叫什么名字") is False  # 无代词、不短、无历史
    # 第三条：当前 query 无代词且够长，仅因上一轮有未消解指代而触发
    assert needs_rewrite("2026 年那一次", ["她说的那家"]) is True
    assert needs_rewrite("2026 年第二次见面", ["她说的那家"]) is True
    assert needs_rewrite("2026 年第二次见面") is False  # 同上但无历史 → 不触发


# ---- 真库 ----

async def _make_agent(session_factory, name: str = "lore-retrieval-test") -> int:
    async with session_factory() as db:
        agent = Agent(name=name, persona_prompt="测试人设")
        db.add(agent)
        await db.flush()
        agent_id = agent.id
        await db.commit()
    return agent_id


async def _cleanup(session_factory, agent_id: int) -> None:
    async with session_factory() as db:
        await db.execute(delete(LoreChunk).where(LoreChunk.agent_id == agent_id))
        await db.execute(delete(LoreDoc).where(LoreDoc.agent_id == agent_id))
        await db.execute(delete(Agent).where(Agent.id == agent_id))
        await db.commit()


async def _import(session_factory, agent_id: int, *, slug: str, title: str, content: str,
                  always_on: bool = False) -> None:
    indexer = Indexer(session_factory=session_factory, embedder=_FakeEmbedder())
    await indexer.upsert_doc(
        agent_id=agent_id, slug=slug, title=title, content=content, always_on=always_on
    )


async def _tokens_of(session_factory, agent_id: int, ref: str) -> int:
    async with session_factory() as db:
        return (
            await db.execute(
                select(LoreChunk.token_count).where(
                    LoreChunk.agent_id == agent_id, LoreChunk.source_ref == ref
                )
            )
        ).scalar_one()


async def test_empty_corpus_returns_empty(real_session_factory) -> None:
    """空库：三路皆空、注入为空、rerank=false（契约 4.4/9.6）。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        result = await _retriever(real_session_factory).retrieve(
            agent_id=agent_id, query="祖父的茶馆"
        )
        assert result.routes == {"always_on": [], "keyword": [], "vector": []}
        assert result.fused == [] and result.reranked == [] and result.injected == []
        assert result.rerank is False and result.lore_tokens == 0
        assert result.lore_cap == int(8000 * 0.20)
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_stopword_query_skips_keyword_route(real_session_factory) -> None:
    """查询只剩单字 → keyword 路直接跳过，不报错（契约 4.4）。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        await _import(
            real_session_factory, agent_id, slug="tea_house", title=_TITLE, content=_content()
        )
        result = await _retriever(real_session_factory).retrieve(agent_id=agent_id, query="的了。")
        assert result.routes["keyword"] == []
        assert result.rewritten == "的了。" and result.rewrite_used is False
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_keyword_route_hits_chinese_word(real_session_factory) -> None:
    """中文词经 jieba + tsv 命中对应 chunk（契约 4.4/§10.1）。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        await _import(
            real_session_factory, agent_id, slug="tea_house", title=_TITLE, content=_content()
        )
        result = await _retriever(real_session_factory, embedder=None).retrieve(
            agent_id=agent_id, query="祖父开的茶馆"
        )
        assert "tea_house:01" in result.routes["keyword"]
        assert result.injected and result.injected[0].source_ref == "tea_house:01"
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_budget_truncation(real_session_factory) -> None:
    """lore 段不超过 20% 预算：只装入能放下的高排名 chunk（契约 4.7）。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        await _import(
            real_session_factory, agent_id, slug="tea_house", title=_TITLE, content=_content()
        )
        hist_tokens = await _tokens_of(real_session_factory, agent_id, "tea_house:01")
        sign_tokens = await _tokens_of(real_session_factory, agent_id, "tea_house:02")
        assert hist_tokens > sign_tokens, "语料应保证历史段更长，便于验证截断"

        budget = 350  # lore_cap = 70，只能放下较短的招牌段
        result = await _retriever(
            real_session_factory, embedder=None, budget_tokens=budget
        ).retrieve(agent_id=agent_id, query="祖父开的茶馆")
        assert result.lore_cap == 70
        assert hist_tokens > result.lore_cap and sign_tokens <= result.lore_cap
        assert [hit.source_ref for hit in result.injected] == ["tea_house:02"]
        assert result.lore_tokens == sign_tokens <= result.lore_cap
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_always_on_takes_budget_first(real_session_factory) -> None:
    """always-on 先占预算（≤8%），且排在注入列表最前（契约 4.4/4.7）。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        await _import(
            real_session_factory, agent_id, slug="tea_house", title=_TITLE, content=_content()
        )
        await _import(
            real_session_factory,
            agent_id,
            slug="core",
            title="本店规矩",
            content=f"# 规矩\n\n## 核心\n{_CORE}\n",
            always_on=True,
        )
        result = await _retriever(real_session_factory).retrieve(
            agent_id=agent_id, query="祖父开的茶馆"
        )
        assert result.routes["always_on"] == ["core:01"]
        assert result.injected[0].source_ref == "core:01"
        assert result.injected[0].routes == ("always_on",)
        assert result.lore_tokens <= result.lore_cap
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_preview_structure(real_session_factory) -> None:
    """集成：preview 结构完整（三路 / 融合 / 重排 / 注入 / 预算）。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        await _import(
            real_session_factory, agent_id, slug="tea_house", title=_TITLE, content=_content()
        )
        result = await _retriever(real_session_factory).retrieve(
            agent_id=agent_id, query="她祖父开的那家茶馆叫什么"
        )
        assert set(result.routes) == {"always_on", "keyword", "vector"}
        assert "tea_house:01" in result.routes["keyword"]
        assert result.fused and result.reranked
        assert result.rerank is False  # NullReranker 降级
        hit = result.injected[0]
        assert hit.source_ref and hit.header_path and hit.content and hit.token_count > 0
        assert hit.as_recalled()["routes"] == list(hit.routes)
        assert result.lore_cap == 1600
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_rewrite_used_and_timeout_fallback(real_session_factory) -> None:
    """改写成功则用改写结果；超时则回退原 query（契约 4.6）。"""
    agent_id = await _make_agent(real_session_factory)
    try:
        used = await _retriever(
            real_session_factory,
            rewriter=_EchoRewriter(),
            rewrite_enabled=True,
            rewrite_timeout=0.5,
        ).retrieve(agent_id=agent_id, query="她祖父开的茶馆叫什么")
        assert used.rewrite_used is True and used.rewritten == "祖父 茶馆 名字"

        fallback = await _retriever(
            real_session_factory,
            rewriter=_SlowRewriter(),
            rewrite_enabled=True,
            rewrite_timeout=0.01,
        ).retrieve(agent_id=agent_id, query="她祖父开的茶馆叫什么")
        assert fallback.rewrite_used is False
        assert fallback.rewritten == "她祖父开的茶馆叫什么"
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_retrieve_lore_history_excludes_current_message(real_session_factory) -> None:
    """节点级（契约 4.6）：`_history` 排除本轮当前用户消息、只留用户消息；
    上一轮用户消息含未消解指代时，第三条件应经 `retrieve_lore` 正常触发改写。"""
    messages = [
        HumanMessage(content="她说的那家茶馆在哪"),
        AIMessage(content="在青溪镇，云栖别院旁。"),
        HumanMessage(content="2026 年第二次见面"),
    ]
    assert _history(messages) == ["她说的那家茶馆在哪"]  # 排除当前消息 + 丢弃 AI 消息
    assert "2026 年第二次见面" not in _history(messages)

    rewriter = _RecordingRewriter()
    retriever = _retriever(
        real_session_factory, rewriter=rewriter, rewrite_enabled=True, rewrite_timeout=0.5
    )
    agent_id = await _make_agent(real_session_factory, name="lore-node-rewrite-test")
    try:
        out = await retrieve_lore({"agent_id": agent_id, "messages": messages}, retriever)
        assert out == {"recalled_lore": []}
        # 当前 query 无代词且够长 → 仅因「上一轮含未消解指代」触发，证明第三条生效
        assert rewriter.calls == [("2026 年第二次见面", ["她说的那家茶馆在哪"])]
    finally:
        await _cleanup(real_session_factory, agent_id)


async def test_rewrite_recent_drops_duplicate_current_query(tmp_path) -> None:
    """`rewrite()` 的「最近对话」不重复当前 query（契约 4.6）。"""
    llm = _CapturingLlm()
    rewriter = LlmRewriter(llm, cache_path=tmp_path / "rewrites.json")
    out = await rewriter.rewrite("她说的那家", ["开场白", "她说的那家"])
    assert out == "改写结果"
    assert llm.messages is not None
    assert llm.messages[1]["content"].count("她说的那家") == 1
