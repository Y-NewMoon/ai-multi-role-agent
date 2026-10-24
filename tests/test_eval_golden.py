"""golden 套件在 FakeEmbeddings 下可跑通（全绿）；附套件 id/结构校验与 forbid 负例。"""

import json
import sys
from pathlib import Path

from langgraph.store.memory import InMemoryStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402
from fake_embeddings import FakeEmbeddings  # noqa: E402


def _store():
    from app.config import get_settings

    s = get_settings()
    return InMemoryStore(
        index={"dims": s.embedding_dims, "embed": FakeEmbeddings(s.embedding_dims), "fields": ["text"]}
    )


async def test_golden_全绿():
    cases = run_eval.load_suite("golden")
    result = await run_golden_with_fake(cases)
    assert result["metrics"]["isolation_leaks"] == 0
    assert result["metrics"]["memory_hit_rate"] == 1.0
    assert result["metrics"]["forbid_violations"] == 0
    assert result["failures"] == []


async def run_golden_with_fake(cases):
    return await run_eval.run_golden(_store(), cases, "testrun")


def test_golden_结构与kind():
    cases = run_eval.load_suite("golden")
    for c in cases:
        assert c["kind"] in ("isolation", "memory", "forbid"), c["id"]
        if c["kind"] == "isolation":
            assert set(c) >= {"id", "kind", "probe", "facts_a", "facts_b"}
        elif c["kind"] == "memory":
            assert set(c) >= {"id", "kind", "query", "facts", "expect", "limit"}
        else:
            assert set(c) >= {"id", "kind", "reply", "forbid_any"}


def test_全部套件id唯一():
    ids: list[str] = []
    for path in sorted((ROOT / "eval" / "suites").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                ids.append(json.loads(line)["id"])
    assert len(ids) == len(set(ids)), "套件间存在重复 id"


def test_forbid检测器负例():
    """正例由 golden 用例覆盖（reply 必须零违规），检测能力靠负例断言。"""
    hits = run_eval.check_forbidden(
        "作为一个AI，我无法回答这个问题。", ["作为一个AI", "语言模型", "人工智能"]
    )
    assert hits == ["作为一个AI"]
    assert run_eval.check_forbidden("客人请喝茶", ["作为一个AI"]) == []


def test_select_cases_确定性():
    cases = run_eval.load_suite("golden")
    a = run_eval.select_cases(cases, "test", 5)
    b = run_eval.select_cases(cases, "test", 5)
    assert [c["id"] for c in a] == [c["id"] for c in b]


async def test_retrieval_冒烟():
    if not (ROOT / "eval" / "suites" / "retrieval.jsonl").exists():
        return
    result = run_eval.run_retrieval(run_eval.load_suite("retrieval"))
    for key in ("hit@1", "hit@3", "mrr", "ndcg@10"):
        value = result["metrics"][key]
        assert value is None or 0.0 <= value <= 1.0, (key, value)
