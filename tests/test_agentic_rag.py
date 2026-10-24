"""Agentic RAG 实验结构与跳过语义单测（契约 §2.10 / §3-33）。

覆盖：
- `eval/suites/agentic_rag.jsonl` 20 条、字段齐备、`expect_refs` 全部指向真实 fixture 小节；
- 三策略（static/agentic/hybrid）的图拓扑：static 不出现 `tools`，agentic/hybrid 出现；
- 跳过语义：`skipped=true` 时三个指标为 `null`（不是 0.0）；
- `lore_context` 由 `expect_refs` 对应 chunk 拼接（judge 设定一致性用）。
"""

import json
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langgraph.store.memory import InMemoryStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_agentic_rag as rag  # noqa: E402

SUITE_PATH = ROOT / "eval" / "suites" / "agentic_rag.jsonl"
FIXTURE_DIR = ROOT / "eval" / "fixtures" / "lore_long"
_DIMS = 1024


class _Emb(Embeddings):
    def embed_documents(self, texts):
        return [[0.1] * _DIMS for _ in texts]

    def embed_query(self, text):
        return [0.1] * _DIMS


class _LLM:
    async def generate(self, messages):
        return ""

    async def astream(self, messages):
        yield ""

    async def generate_with_tools(self, messages, tools, *, tool_choice="auto", temperature=0.0):
        from app.llm.base import LLMReply

        return LLMReply(content="ok")


class _Settings:
    prompt_token_budget = 8000
    skill_max_tool_rounds = 3
    embedding_dims = _DIMS


def _session_factory():
    raise AssertionError("图构造期不应访问会话工厂")


def _fixture_refs() -> set[str]:
    refs: set[str] = set()
    for path in sorted(FIXTURE_DIR.glob("*.md")):
        sections = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("## ")
        ]
        refs |= {f"{path.stem}:{i:02d}" for i in range(1, len(sections) + 1)}
    return refs


def _cases() -> list[dict]:
    return [
        json.loads(line)
        for line in SUITE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_suite_has_20_cases_with_schema():
    cases = _cases()
    assert len(cases) == 20
    assert [c["id"] for c in cases] == [f"ar-{i:03d}" for i in range(1, 21)]
    for c in cases:
        assert set(c) >= {"id", "agent_id", "query", "expect_refs", "tags", "difficulty", "synthetic"}
        assert isinstance(c["agent_id"], int)
        assert c["query"] and c["expect_refs"]


def test_expect_refs_point_to_real_fixture_sections():
    valid = _fixture_refs()
    assert len(valid) == 20  # 两篇 × 10 小节
    for c in _cases():
        assert set(c["expect_refs"]) <= valid, c["id"]


def test_metric_keys_and_strategies():
    assert rag.STRATEGIES == ("static", "agentic", "hybrid")
    assert set(rag.METRIC_KEYS) == {
        "setting_consistency", "avg_prompt_tokens", "first_token_latency_ms", "queries",
    }
    assert rag.READ_LORE == ["read_lore"]


def test_graph_topology_per_strategy():
    common = dict(
        llm=_LLM(),
        settings=_Settings(),
        embed=_Emb(),
        reranker=None,
        session_factory=_session_factory,
        store=InMemoryStore(index={"dims": _DIMS, "embed": _Emb(), "fields": ["text"]}),
    )
    tool_retriever = rag.Retriever(session_factory=_session_factory, embedder=_Emb())
    static, static_executor = rag._build_graph_for("static", **common, tool_retriever=tool_retriever)
    assert "tools" not in set(static.get_graph().nodes)  # static：不绑工具，拓扑同 V2.1
    assert static_executor is None  # static 无 executor，每 case 无需 reset
    for name in ("agentic", "hybrid"):
        graph, executor = rag._build_graph_for(name, **common, tool_retriever=tool_retriever)
        assert "tools" in set(graph.get_graph().nodes), name
        assert executor is not None, name  # 有 executor 才能在每 case 前 reset（§2.2-5）


def test_skip_semantics_null_metrics():
    row = rag._null_strategy("no_test_database_url")
    assert row["skipped"] is True and row["skip_reason"] == "no_test_database_url"
    assert row["queries"] == 0
    for key in ("setting_consistency", "avg_prompt_tokens", "first_token_latency_ms"):
        assert row[key] is None  # 跳过时为 null，不得写 0.0


def test_lore_context_built_from_expect_refs():
    contents = {
        "yunqi_tea:01": {"header_path": "创始人", "content": "云栖茶庄由苏云栖创办。"},
        "qingxi_town:04": {"header_path": "老桥", "content": "老桥名唤听雨。"},
    }
    case = {"expect_refs": ["yunqi_tea:01", "qingxi_town:04", "missing:99"]}
    ctx = rag._lore_context(case, contents)
    assert "云栖茶庄由苏云栖创办。" in ctx and "老桥" in ctx
    assert "missing:99" not in ctx  # 缺失 ref 跳过
    assert rag._lore_context({"expect_refs": []}, contents) is None
