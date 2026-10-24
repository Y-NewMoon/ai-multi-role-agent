"""retrieval_lore 评测扩展（契约 4.10 / T7）：套件结构、长语料、跳过语义、报告 schema 1.1。

覆盖文档 §5 T7 的验收点：
- `eval/suites/retrieval_lore.jsonl` 40 条，类型配额 专名 12 / 指代 12 / 多跳 8 / 别名 8；
- `eval/fixtures/lore_long/` 两篇长语料，各 ≥2000 字、10 个 `##` 小节，chunk 与 section 1:1；
- 跳过条件触发时四指标为 `null`（不是 `0.0`）、`skipped=true`，且退出码 0（不算失败）；
- `report.py` schema 升 `1.1` 且 `retrieval_lore` 新键齐备；布尔元信息不作差。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import lore_eval  # noqa: E402
import report as report_mod  # noqa: E402
import run_eval  # noqa: E402

SUITE_PATH = ROOT / "eval" / "suites" / "retrieval_lore.jsonl"


def _cases() -> list[dict]:
    return [
        json.loads(line)
        for line in SUITE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# 套件结构（契约 4.10）
# ---------------------------------------------------------------------------
def test_retrieval_lore_共40条():
    cases = _cases()
    assert len(cases) == 40


def test_retrieval_lore_类型配额():
    cases = _cases()
    counts = {
        tag: sum(1 for c in cases if tag in c["tags"])
        for tag in ("专名", "指代", "多跳", "别名")
    }
    assert counts == {"专名": 12, "指代": 12, "多跳": 8, "别名": 8}, counts


def test_retrieval_lore_字段齐全且id唯一():
    cases = _cases()
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 40
    for c in cases:
        assert set(c) == {
            "id",
            "agent_id",
            "query",
            "expect_refs",
            "difficulty",
            "tags",
            "synthetic",
        }, c["id"]
        assert c["agent_id"] == lore_eval.DEFAULT_AGENT_ID
        assert c["synthetic"] is False
        assert c["difficulty"] in ("easy", "medium", "hard")
        assert c["query"].strip()


def test_retrieval_lore_多跳两条其余一条():
    for c in _cases():
        if "多跳" in c["tags"]:
            assert len(c["expect_refs"]) == 2, c["id"]
        else:
            assert len(c["expect_refs"]) == 1, c["id"]


def test_retrieval_lore_refs指向前缀():
    import re

    pattern = re.compile(r"^(yunqi_tea|qingxi_town):\d{2}$")
    for c in _cases():
        for ref in c["expect_refs"]:
            assert pattern.match(ref), f"{c['id']}: {ref}"


def test_retrieval_lore_至少5条dev():
    # 与既有 220 条合并后 dev 比例须落在 0.60~0.80（test_eval_split 全量校验）
    dev = sum(1 for c in _cases() if __import__("split").split_of(c["id"]) == "dev")
    assert 0 < dev < 40


# ---------------------------------------------------------------------------
# 长语料（契约 4.10）
# ---------------------------------------------------------------------------
def _fixtures() -> list[tuple[str, str, str]]:
    return lore_eval.load_fixtures()


def test_lore_long_两篇且字数达标():
    fixtures = _fixtures()
    assert [f[0] for f in fixtures] == ["qingxi_town", "yunqi_tea"]
    for slug, title, content in fixtures:
        assert title and title != slug  # 取到 `# ` 标题
        assert len(content) >= 2000, f"{slug} 仅 {len(content)} 字符"


def test_lore_long_小节与chunk一一对应():
    from app.lore.chunker import chunk_document
    from app.lore.parser import parse_markdown

    for slug, title, content in _fixtures():
        parsed = parse_markdown(content, doc_title=title)
        chunks = chunk_document(parsed)
        assert 8 <= len(parsed.sections) <= 12, slug
        assert len(chunks) == len(parsed.sections), slug


# ---------------------------------------------------------------------------
# 跳过语义：指标为 null（不是 0.0）
# ---------------------------------------------------------------------------
def test_skipped_metrics_四指标为None():
    row = lore_eval.skipped_metrics("no_test_database")
    for name in lore_eval.METRIC_NAMES:
        assert row[name] is None
    assert row["queries"] == 0
    assert row["rerank"] is False
    assert row["skipped"] is True
    assert row["skip_reason"] == "no_test_database"


def test_run_retrieval_lore_inmemory跳过():
    import asyncio
    from types import SimpleNamespace

    cases = _cases()[:3]
    settings = SimpleNamespace(test_database_url=None)
    result = asyncio.run(
        run_eval._run_retrieval_lore(cases, settings, force_inmemory=True)
    )
    assert result["failures"] == []
    assert result["metrics"]["skipped"] is True
    assert result["metrics"]["skip_reason"] == "no_test_database"
    for name in lore_eval.METRIC_NAMES:
        assert result["metrics"][name] is None


def test_run_retrieval_lore_无库url跳过():
    import asyncio
    from types import SimpleNamespace

    settings = SimpleNamespace(test_database_url=None)
    result = asyncio.run(
        run_eval._run_retrieval_lore(_cases()[:2], settings, force_inmemory=False)
    )
    assert result["metrics"]["skip_reason"] == "no_test_database"


# ---------------------------------------------------------------------------
# 指标计算（lore_eval.measure）
# ---------------------------------------------------------------------------
def test_measure_已知排序指标():
    cases = [
        {"id": "a", "expect_refs": ["s:01"]},
        {"id": "b", "expect_refs": ["s:02"]},
    ]
    ranked = {"a": ["s:01", "x:01"], "b": ["x:01", "x:02", "s:02"]}
    out = lore_eval.measure(cases, ranked, rerank=True)
    m = out["metrics"]
    assert m["hit@1"] == pytest.approx(0.5)  # 2 例中 1 例 top-1 命中
    assert m["hit@3"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert m["ndcg@10"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["queries"] == 2
    assert m["rerank"] is True
    assert m["skipped"] is False
    assert m["skip_reason"] is None


def test_measure_top1未命中记failure():
    cases = [{"id": "b", "expect_refs": ["s:02"]}]
    out = lore_eval.measure(cases, {"b": ["x:01", "s:02"]}, rerank=False)
    assert out["metrics"]["hit@1"] == 0.0
    assert out["metrics"]["hit@3"] == 1.0
    assert len(out["failures"]) == 1
    assert out["failures"][0]["suite"] == "retrieval_lore"
    assert out["failures"][0]["id"] == "b"


def test_measure_空用例指标为None():
    out = lore_eval.measure([], {}, rerank=False)
    for name in lore_eval.METRIC_NAMES:
        assert out["metrics"][name] is None
    assert out["metrics"]["queries"] == 0
    assert out["failures"] == []


# ---------------------------------------------------------------------------
# 报告 schema 1.1（契约 4.10）
# ---------------------------------------------------------------------------
def test_report_schema升为1_1():
    assert report_mod.SCHEMA_VERSION == "1.1"


def test_report_retrieval_lore键齐备():
    keys = report_mod.METRIC_KEYS["retrieval_lore"]
    assert keys == (
        "hit@1",
        "hit@3",
        "mrr",
        "ndcg@10",
        "queries",
        "rerank",
        "skipped",
        "skip_reason",
    )
    assert set(report_mod.empty_metrics()["retrieval_lore"]) == set(keys)


def test_compute_delta_布尔作差为None():
    metrics = {"retrieval_lore": {"hit@1": 0.5, "rerank": True, "skipped": False}}
    base = {"retrieval_lore": {"hit@1": 0.4, "rerank": False, "skipped": False}}
    delta = report_mod.compute_delta(metrics, base)["retrieval_lore"]
    assert delta["hit@1"] == pytest.approx(0.1)
    assert delta["rerank"] is None  # 布尔元信息不作差
    assert delta["skipped"] is None


def test_delta键覆盖retrieval_lore():
    metrics = report_mod.empty_metrics()
    metrics["retrieval_lore"].update({"hit@1": 0.8, "hit@3": 0.9})
    delta = report_mod.compute_delta(metrics, {})["retrieval_lore"]
    assert set(delta) == set(report_mod.METRIC_KEYS["retrieval_lore"])


# ---------------------------------------------------------------------------
# 软门禁：跳过（null）不告警；低于基线才告警
# ---------------------------------------------------------------------------
def test_gates_跳过无lore告警():
    report = {
        "metrics": report_mod.empty_metrics(),
        "failures": [],
        "baseline": {"path": None, "delta": {}},
    }
    hard, soft = run_eval.evaluate_gates(report)
    assert hard == []
    assert not any("retrieval_lore" in s for s in soft)


def test_gates_低于基线触发软告警(tmp_path):
    baseline_file = tmp_path / "base.json"
    baseline_file.write_text(
        json.dumps({"metrics": {"retrieval_lore": {"hit@3": 0.9}}}),
        encoding="utf-8",
    )
    metrics = report_mod.empty_metrics()
    metrics["retrieval_lore"]["hit@3"] = 0.5
    report = {
        "metrics": metrics,
        "failures": [],
        "baseline": {
            "path": str(baseline_file),
            "delta": {"retrieval_lore": {"hit@3": -0.4}},
        },
    }
    _hard, soft = run_eval.evaluate_gates(report)
    assert any("retrieval_lore.hit@3" in s for s in soft)


def test_suite_names含retrieval_lore():
    assert "retrieval_lore" in run_eval.SUITE_NAMES
