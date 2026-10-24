"""纯函数指标库（契约 4.2）：无 IO、无随机、无外部依赖。

口径与 V1 方案 §8 一致：
- 串味 = 0 是硬指标（计数在 run_eval 侧，本库只提供 pass_rate 等原语）；
- 记忆命中 = `expect` 全部被召回才算命中（run_eval 沿用 `all(e in recalled)`）。
"""

import math

__all__ = [
    "hit_at_k",
    "mrr",
    "ndcg_at_k",
    "pass_rate",
    "cohens_kappa",
    "p50",
    "p95",
]


def hit_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    """expected 与 retrieved[:k] 有交集即 1.0，否则 0.0。k<=0 抛 ValueError。"""
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k!r}")
    return 1.0 if set(retrieved[:k]) & set(expected) else 0.0


def mrr(expected: set[str], retrieved: list[str]) -> float:
    """首个命中位置的倒数（位置从 1 起）；未命中返回 0.0。"""
    for rank, item in enumerate(retrieved, start=1):
        if item in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """标准 DCG/IDCG，relevances 按检索顺序给；全零返回 0.0。"""
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k!r}")
    rel = [float(r) for r in relevances[:k]]
    if not rel or all(r <= 0 for r in rel):
        return 0.0
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted((float(r) for r in relevances), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def pass_rate(results: list[bool]) -> float | None:
    """空列表返回 None：未测 ≠ 0 命中（报告按项目约定写 null，见 V2.0 §3）。"""
    if not results:
        return None
    return sum(1 for r in results if r) / len(results)


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """长度不等抛 ValueError；完全一致且无方差时返回 1.0。"""
    if len(a) != len(b):
        raise ValueError(f"两列标注长度不等：{len(a)} != {len(b)}")
    if not a:
        raise ValueError("标注为空，无法计算 kappa")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if pe == 1.0:
        return 1.0  # 完全一致且无方差
    return (po - pe) / (1.0 - pe)


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


def p50(values: list[float]) -> float:
    """最近秩法（nearest-rank），空列表返回 0.0。"""
    return _nearest_rank(values, 50)


def p95(values: list[float]) -> float:
    """最近秩法（nearest-rank），空列表返回 0.0。"""
    return _nearest_rank(values, 95)
