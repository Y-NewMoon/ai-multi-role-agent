"""M5 手段①：rerank/embed 并发闸门与排队计时（T5）。

全部用假 reranker，不加载真模型、不访问数据库（只调 `Retriever._rerank`）。
覆盖四条：
1. 闸门生效：容量 1 时 5 个并发 rerank 的“同时在跑数”最大为 1；
2. 排队可观测：`rerank_queue_ms > 0`、`rerank_compute_ms > 0` 且 `rerank_ms ≈ queue + compute`；
3. 关闸门：最大并发 > 1，且 `*_queue_ms == 0`（行为退回改造前）；
4. 异常安全：compute 抛异常时信号量仍被释放（再跑一次不死锁）。
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from app.lore.retriever import Retriever

_FUSED = [("a:01", 1.0), ("b:01", 0.5)]
_ROWS = {
    "a:01": SimpleNamespace(content="甲的设定"),
    "b:01": SimpleNamespace(content="乙的设定"),
}


class _ProbeReranker:
    """线程内统计“同时在跑数”；可选在最初 N 次调用抛异常。"""

    def __init__(self, *, delay_s: float = 0.03, fail_first: int = 0) -> None:
        self._delay_s = delay_s
        self._fail_first = fail_first
        self._lock = threading.Lock()
        self.calls = 0
        self.max_concurrent = 0
        self._current = 0

    def rank(self, query: str, docs: list[str], top_n: int):  # noqa: ANN201
        with self._lock:
            self.calls += 1
            call_no = self.calls
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
        try:
            time.sleep(self._delay_s)
            if call_no <= self._fail_first:
                raise RuntimeError("compute boom")
            return [(i, 1.0 - i * 0.1) for i in range(min(top_n, len(docs)))]
        finally:
            with self._lock:
                self._current -= 1


def _retriever(reranker: _ProbeReranker, *, capacity: int = 1, enabled: bool = True) -> Retriever:
    return Retriever(
        session_factory=object(),
        reranker=reranker,
        rerank_concurrency=capacity,
        gate_enabled=enabled,
    )


def _stages(n: int) -> list[dict]:
    return [{"rerank_ms": None, "rerank_queue_ms": None, "rerank_compute_ms": None} for _ in range(n)]


async def test_gate_limits_rerank_concurrency() -> None:
    """1）容量 1：5 个并发 rerank 同时在跑的最大数 == 1。"""
    probe = _ProbeReranker()
    retriever = _retriever(probe, capacity=1)
    stages = _stages(5)
    await asyncio.gather(
        *[retriever._rerank("q", _FUSED, _ROWS, stage_ms=st) for st in stages]
    )
    assert probe.calls == 5
    assert probe.max_concurrent == 1


async def test_queue_and_compute_timings_add_up() -> None:
    """2）排队与计算都 > 0，且 rerank_ms ≈ queue + compute（±10ms）。"""
    probe = _ProbeReranker()
    retriever = _retriever(probe, capacity=1)
    stages = _stages(5)
    await asyncio.gather(
        *[retriever._rerank("q", _FUSED, _ROWS, stage_ms=st) for st in stages]
    )
    for stage in stages:
        assert stage["rerank_compute_ms"] > 0, stage
    # 后 4 个请求都在等同一个名额
    assert max(stage["rerank_queue_ms"] for stage in stages) > 0
    for stage in stages:
        assert abs(stage["rerank_ms"] - (stage["rerank_queue_ms"] + stage["rerank_compute_ms"])) <= 10, stage


async def test_gate_disabled_restores_old_behavior() -> None:
    """3）关闸门：允许并行（最大并发 > 1），且 queue_ms 恒为 0。"""
    probe = _ProbeReranker(delay_s=0.05)
    retriever = _retriever(probe, capacity=1, enabled=False)
    stages = _stages(5)
    await asyncio.gather(
        *[retriever._rerank("q", _FUSED, _ROWS, stage_ms=st) for st in stages]
    )
    assert probe.max_concurrent > 1
    for stage in stages:
        assert stage["rerank_queue_ms"] == 0, stage
        assert stage["rerank_compute_ms"] > 0, stage


async def test_gate_released_on_compute_error() -> None:
    """4）compute 抛异常：信号量仍释放，后续调用能拿到闸门（不死锁）。"""
    probe = _ProbeReranker(delay_s=0.01, fail_first=1)
    retriever = _retriever(probe, capacity=1)
    with pytest.raises(RuntimeError, match="compute boom"):
        await retriever._rerank("q", _FUSED, _ROWS, stage_ms=_stages(1)[0])

    result, ok = await asyncio.wait_for(
        retriever._rerank("q", _FUSED, _ROWS, stage_ms=_stages(1)[0]), timeout=2.0
    )
    assert ok is True
    assert [ref for ref, _ in result] == ["a:01", "b:01"]
    assert probe.calls == 2
