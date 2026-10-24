"""KeyedLockPool（V2.4 B-02 §2.7）：同键串行 / 异键并行 / 异常取消释放 / 不驻留。"""

from __future__ import annotations

import asyncio
import tracemalloc

import pytest

from app.core.locks import KeyedLockPool


async def test_同键串行():
    pool = KeyedLockPool()
    order: list[tuple[str, str]] = []

    async def worker(tag: str) -> None:
        async with pool.acquire("k"):
            order.append(("in", tag))
            await asyncio.sleep(0.01)
            order.append(("out", tag))

    await asyncio.gather(worker("a"), worker("b"))
    assert order == [("in", "a"), ("out", "a"), ("in", "b"), ("out", "b")]


async def test_不同键并行():
    pool = KeyedLockPool()
    entered = 0
    max_entered = 0

    async def worker(key: str) -> None:
        nonlocal entered, max_entered
        async with pool.acquire(key):
            entered += 1
            max_entered = max(max_entered, entered)
            await asyncio.sleep(0.02)
            entered -= 1

    await asyncio.gather(worker("a"), worker("b"))
    assert max_entered == 2  # 两个不同 key 同时持锁


async def test_持锁期间驻留退出即清零():
    pool = KeyedLockPool()
    async with pool.acquire("thread:1"):
        assert pool.size() == 1
    assert pool.size() == 0


async def test_异常路径释放():
    pool = KeyedLockPool()
    with pytest.raises(RuntimeError):
        async with pool.acquire("k"):
            raise RuntimeError("boom")
    assert pool.size() == 0


async def test_取消路径释放():
    pool = KeyedLockPool()

    async def hold() -> None:
        async with pool.acquire("k"):
            await asyncio.sleep(10)

    task = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    assert pool.size() == 1
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=0.01)
    assert pool.size() == 0


async def test_1000个不同key依次使用后不驻留():
    tracemalloc.start()
    try:
        pool = KeyedLockPool()
        for i in range(1000):
            async with pool.acquire(f"session:{i}"):
                pass
        current, _peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert pool.size() == 0
    assert current < 50 * 1024 * 1024


async def test_引用计数期间不误删():
    pool = KeyedLockPool()
    started = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with pool.acquire("k"):
            started.set()
            await release.wait()

    async def waiter() -> None:
        await started.wait()
        async with pool.acquire("k"):
            pass

    tasks = [asyncio.create_task(holder()), asyncio.create_task(waiter())]
    await started.wait()
    await asyncio.sleep(0.01)
    assert pool.size() == 1  # holder 持锁 + waiter 等待，条目仍在
    release.set()
    await asyncio.gather(*tasks)
    assert pool.size() == 0
