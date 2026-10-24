"""池化连接下的 LangGraph checkpointer（V2.4 R1 返修）。

`langgraph-checkpoint-postgres` 3.1.2 的 `AsyncPostgresSaver._cursor()` 无论底层是单个连接
还是连接池，都无条件 `async with self.lock`；该锁是**实例级**的，于是同进程内所有会话的
checkpoint 读写被完全串行化（20 并发压测实测吞吐封顶 ~1.5 req/s，20 轮总时长 = 20×单轮）。

同版本的 `AsyncPostgresStore._cursor()` 已按“是否池化”分支（池化时用一次性 `asyncio.Lock`），
本类与之对齐：池化时每次 `_cursor` 从池中取独占连接（psycopg_pool 不会并发复用同一连接），
无需实例级锁；同一 thread 的先后顺序由服务层 `KeyedLockPool`（B-02）保证。
若上游修复该分支，删除本类即可（见交接文档 §7 技术债）。
"""

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool


class _NoopAsyncLock:
    """满足 `async with` 协议的空锁（仅用于连接池场景，不提供互斥）。"""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


class PooledAsyncPostgresSaver(AsyncPostgresSaver):
    """池化连接版 saver：关掉基类的实例级全局锁，让 checkpoint I/O 可并发。"""

    def __init__(self, pool: AsyncConnectionPool, *, serde=None) -> None:
        if not isinstance(pool, AsyncConnectionPool):
            raise TypeError("PooledAsyncPostgresSaver 只接受 AsyncConnectionPool")
        super().__init__(pool, serde=serde)
        # 基类声明为 asyncio.Lock；池化下每次 _cursor 各取独立连接，无需互斥
        self.lock = _NoopAsyncLock()  # type: ignore[assignment]
