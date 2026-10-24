"""指标函数单测（契约 4.2）：边界值、空输入、k 越界、全命中/全不命中、并列分数。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import metrics  # noqa: E402


# ---- hit_at_k -------------------------------------------------------------
def test_hit_at_k_全命中():
    assert metrics.hit_at_k({"a"}, ["a", "b", "c"], 3) == 1.0


def test_hit_at_k_命中在k外():
    assert metrics.hit_at_k({"c"}, ["a", "b", "c"], 2) == 0.0


def test_hit_at_k_多期望命中一条即算():
    assert metrics.hit_at_k({"x", "b"}, ["a", "b"], 2) == 1.0


def test_hit_at_k_全不命中():
    assert metrics.hit_at_k({"z"}, ["a", "b"], 5) == 0.0


def test_hit_at_k_空输入():
    assert metrics.hit_at_k(set(), [], 3) == 0.0


def test_hit_at_k_k越界():
    with pytest.raises(ValueError):
        metrics.hit_at_k({"a"}, ["a"], 0)
    with pytest.raises(ValueError):
        metrics.hit_at_k({"a"}, ["a"], -1)


# ---- mrr ------------------------------------------------------------------
def test_mrr_首条命中():
    assert metrics.mrr({"a"}, ["a", "b"]) == 1.0


def test_mrr_第三条命中():
    assert metrics.mrr({"c"}, ["a", "b", "c"]) == pytest.approx(1 / 3)


def test_mrr_未命中():
    assert metrics.mrr({"z"}, ["a", "b"]) == 0.0


def test_mrr_空输入():
    assert metrics.mrr({"a"}, []) == 0.0


# ---- ndcg_at_k ------------------------------------------------------------
def test_ndcg_理想排序():
    assert metrics.ndcg_at_k([1.0, 0.0, 0.0], 3) == pytest.approx(1.0)


def test_ndcg_全零返回0():
    assert metrics.ndcg_at_k([0.0, 0.0, 0.0], 3) == 0.0


def test_ndcg_空输入():
    assert metrics.ndcg_at_k([], 10) == 0.0


def test_ndcg_非理想排序():
    value = metrics.ndcg_at_k([0.0, 1.0, 0.0], 3)
    assert 0.0 < value < 1.0


def test_ndcg_并列分数确定():
    assert metrics.ndcg_at_k([1.0, 1.0, 0.0], 3) == metrics.ndcg_at_k([1.0, 1.0, 0.0], 3)


def test_ndcg_k越界():
    with pytest.raises(ValueError):
        metrics.ndcg_at_k([1.0], 0)


# ---- pass_rate ------------------------------------------------------------
def test_pass_rate_空列表返回None():
    assert metrics.pass_rate([]) is None


def test_pass_rate_混合():
    assert metrics.pass_rate([True, False, True, True]) == 0.75


def test_pass_rate_全灭():
    assert metrics.pass_rate([False, False]) == 0.0


# ---- cohens_kappa ---------------------------------------------------------
def test_kappa_完全一致():
    assert metrics.cohens_kappa([1, 2, 3, 2], [1, 2, 3, 2]) == 1.0


def test_kappa_无方差完全一致返回1():
    assert metrics.cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_kappa_长度不等抛错():
    with pytest.raises(ValueError):
        metrics.cohens_kappa([1, 2], [1])


def test_kappa_空输入抛错():
    with pytest.raises(ValueError):
        metrics.cohens_kappa([], [])


def test_kappa_完全相反低于随机():
    value = metrics.cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0])
    assert value < 0


# ---- p50 / p95 ------------------------------------------------------------
def test_p50_最近秩():
    assert metrics.p50([1.0, 2.0, 3.0]) == 2.0


def test_p95_最近秩():
    values = [float(i) for i in range(1, 21)]
    assert metrics.p95(values) == 19.0


def test_p50_p95_空列表():
    assert metrics.p50([]) == 0.0
    assert metrics.p95([]) == 0.0


def test_p50_单元素():
    assert metrics.p50([7.0]) == 7.0
