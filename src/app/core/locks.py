"""按键串行化的锁池（V2.4 · B-02 §2.7）。

- 同 key 串行（`asyncio.Lock`），不同 key 并行；
- 引用计数：`acquire` 进入前 +1，`finally` -1；归零即从字典移除，
  避免"每会话一把锁"长期驻留（1000 会话后 `size()` 回到 0）；
- 异常/取消由 `asynccontextmanager` 的 `finally` 保证释放；
- 单进程语义；多实例部署需分布式锁（§7）。

`max_idle` 仅作防御性上限：正常路径按引用计数归零清理，只有引用计数异常
（例如历史缺陷遗留）时才由它兜底，避免字典无限增长。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class _Entry:
    """一个 key 对应的锁与引用计数。"""

    __slots__ = ("lock", "refs")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.refs = 0


class KeyedLockPool:
    def __init__(self, *, max_idle: int = 128) -> None:
        self._entries: dict[str, _Entry] = {}
        self._max_idle = max(1, int(max_idle))

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[None]:
        """进入即 +1 引用；退出（含异常/取消）释放锁并 -1，归零即 pop。"""
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry()
            self._entries[key] = entry
        entry.refs += 1
        self._evict_idle()
        try:
            async with entry.lock:
                yield
        finally:
            entry.refs -= 1
            if entry.refs <= 0 and self._entries.get(key) is entry:
                # 仅当字典里仍是同一个 entry 时才移除（防极端并发下误删）
                self._entries.pop(key, None)

    def size(self) -> int:
        """当前驻留的 key 数（用于泄漏验收）。"""
        return len(self._entries)

    def _evict_idle(self) -> None:
        """防御性清理：超过上限时移除无引用的条目（正常路径不会存在）。"""
        if len(self._entries) <= self._max_idle:
            return
        for key in [k for k, e in self._entries.items() if e.refs == 0]:
            if len(self._entries) <= self._max_idle:
                break
            self._entries.pop(key, None)
