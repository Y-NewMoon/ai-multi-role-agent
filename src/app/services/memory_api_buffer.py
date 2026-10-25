"""V3 M2：本机记忆接口的服务端滚动缓冲（单文件 JSON）。

结构（交接 §2.6）::

    {"version": 1,
     "agents": {"ext": {"sessions": {"main": {
         "turn_count": 12,
         "last_turn_at": "2026-10-25T10:35:00+08:00",
         "messages": [{"id": "...", "role": "user", "content": "...", "at": "..."}]}}}}}

- 只存**未摘要**消息；摘要 / L2 / facts / profile 在 Store，不复制进缓冲。
- 每条消息都必须有 `id`（用户消息与每条角色回复）；摘要后按 `RemoveMessage.id` 真删。
- 原子写：同目录临时文件 → `os.replace`；全局文件锁 + 每 `(agent,session)` 一把 `asyncio.Lock`。
- 文件损坏 → 改名 `.corrupt-<ts>`、记 error、以空缓冲启动（Store 记忆不丢）。
- 单进程语义，不做跨进程锁。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

TZ = ZoneInfo("Asia/Shanghai")


class MemoryApiBufferError(Exception):
    """缓冲层错误基类。"""


class BufferOverflowError(MemoryApiBufferError):
    """追加前缓冲已达上限。"""


class BufferPersistError(MemoryApiBufferError):
    """缓冲写盘失败。"""


def build_turn_messages(
    *,
    user_text: str,
    user_id: str,
    replies: list[str],
    at: str,
) -> list[dict[str, Any]]:
    """构造本轮待追加消息：用户消息带 `at`，角色回复不带 `at`（M1 D4）。"""
    messages: list[dict[str, Any]] = [
        {"id": user_id, "role": "user", "content": user_text, "at": at}
    ]
    for index, reply in enumerate(replies):
        messages.append(
            {"id": f"{user_id}:a{index}", "role": "assistant", "content": reply}
        )
    return messages


class MemoryApiBuffer:
    """单进程缓冲：结构读写 + 原子持久化。"""

    def __init__(self, path: str | Path, *, max_messages: int = 200) -> None:
        self._path = Path(path)
        self._max_messages = max(1, int(max_messages))
        self._data: dict[str, Any] = self._empty()
        self._loaded = False
        self._file_lock = asyncio.Lock()
        self._session_locks: dict[tuple[str, str], asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    @property
    def max_messages(self) -> int:
        return self._max_messages

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"version": 1, "agents": {}}

    def session_lock(self, agent: str, session: str) -> asyncio.Lock:
        """每 `(agent, session)` 一把锁；调用方持有期间完成读-改-写。"""
        key = (str(agent), str(session))
        lock = self._session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[key] = lock
        return lock

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    async def load(self) -> None:
        async with self._file_lock:
            self._load_unlocked()

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        self._loaded = True
        path = self._path
        if not path.exists():
            self._data = self._empty()
            return
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else self._empty()
            if not isinstance(data, dict) or not isinstance(data.get("agents"), dict):
                raise ValueError("缓冲根结构非法（缺少 agents 对象）")
            data.setdefault("version", 1)
            data["agents"] = {
                str(agent): {
                    "sessions": dict(agent_slot.get("sessions") or {})
                    if isinstance(agent_slot.get("sessions"), dict)
                    else {},
                }
                if isinstance(agent_slot, dict)
                else {"sessions": {}}
                for agent, agent_slot in data["agents"].items()
            }
            self._data = data
        except Exception as exc:  # noqa: BLE001 - 坏文件不能阻止服务启动
            stamp = datetime.now(TZ).strftime("%Y%m%d%H%M%S")
            corrupt = path.with_name(f"{path.name}.corrupt-{stamp}")
            try:
                os.replace(path, corrupt)
            except OSError:
                logger.exception("缓冲损坏文件隔离失败：%s", path.name)
            logger.error(
                "记忆接口缓冲文件损坏，已隔离为 %s，以空缓冲启动：%s",
                corrupt.name,
                exc,
            )
            self._data = self._empty()

    # ------------------------------------------------------------------
    # 会话槽
    # ------------------------------------------------------------------
    def slot(self, agent: str, session: str, *, create: bool = True) -> dict[str, Any] | None:
        """取会话槽；`create=False` 且不存在时返回 None。"""
        self._ensure_loaded()
        agents = self._data.setdefault("agents", {})
        agent_slot = agents.get(str(agent))
        if agent_slot is None:
            if not create:
                return None
            agent_slot = {"sessions": {}}
            agents[str(agent)] = agent_slot
        sessions = agent_slot.setdefault("sessions", {})
        session_slot = sessions.get(str(session))
        if session_slot is None:
            if not create:
                return None
            session_slot = {"turn_count": 0, "last_turn_at": None, "messages": []}
            sessions[str(session)] = session_slot
        session_slot.setdefault("turn_count", 0)
        session_slot.setdefault("last_turn_at", None)
        session_slot.setdefault("messages", [])
        return session_slot

    def messages(self, agent: str, session: str) -> list[dict[str, Any]]:
        slot = self.slot(agent, session)
        return list(slot["messages"]) if slot is not None else []

    def turn_count(self, agent: str, session: str) -> int:
        slot = self.slot(agent, session, create=False)
        return int(slot["turn_count"]) if slot is not None else 0

    def size(self, agent: str, session: str | None = None) -> int:
        """`session` 提供 → 该会话消息数；省略 → 该 agent 全部会话之和。"""
        self._ensure_loaded()
        agent_slot = self._data.get("agents", {}).get(str(agent))
        if not agent_slot:
            return 0
        sessions = agent_slot.get("sessions", {})
        if session is not None:
            slot = sessions.get(str(session))
            return len(slot.get("messages", [])) if slot else 0
        return sum(len(s.get("messages", [])) for s in sessions.values())

    def last_turn_at(self, agent: str, session: str | None = None) -> str | None:
        """`session` 提供 → 该会话的时间；省略 → 该 agent 所有会话的最大值。"""
        self._ensure_loaded()
        agent_slot = self._data.get("agents", {}).get(str(agent))
        if not agent_slot:
            return None
        sessions = agent_slot.get("sessions", {})
        if session is not None:
            slot = sessions.get(str(session))
            return slot.get("last_turn_at") if slot else None
        stamps = [s.get("last_turn_at") for s in sessions.values() if s.get("last_turn_at")]
        return max(stamps) if stamps else None

    def has_message(self, agent: str, session: str, message_id: str) -> bool:
        return any(m.get("id") == message_id for m in self.messages(agent, session))

    def ensure_capacity(self, agent: str, session: str, *, incoming: int = 1) -> None:
        """追加前检查：`current + incoming` 超上限 → `BufferOverflowError`（不静默删）。

        R3：一轮可能追加多条（1 条用户 + N 条回复），只比 `current` 会让多回复轮
        静默突破上限；调用方须传本轮实际追加条数。
        """
        slot = self.slot(agent, session, create=False)
        current = len(slot.get("messages", [])) if slot else 0
        if current + max(0, int(incoming)) > self._max_messages:
            raise BufferOverflowError(
                f"缓冲追加后将达 {current + int(incoming)} 条，超过上限 "
                f"{self._max_messages}，拒绝追加（agent={agent}）"
            )

    def remove_message_ids(self, agent: str, session: str, ids: list[str]) -> tuple[int, list[str]]:
        """按 id 删除；返回 (删除条数, 未找到的 id)。"""
        wanted = [str(i) for i in ids]
        slot = self.slot(agent, session, create=False)
        if slot is None:
            return 0, wanted
        wanted_set = set(wanted)
        kept = [m for m in slot["messages"] if str(m.get("id")) not in wanted_set]
        removed = len(slot["messages"]) - len(kept)
        found = {str(m.get("id")) for m in slot["messages"]} & wanted_set
        slot["messages"] = kept
        return removed, [i for i in wanted if i not in found]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    async def save(self) -> None:
        async with self._file_lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.tmp-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except Exception as exc:  # noqa: BLE001 - 统一包装成缓冲层错误
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise BufferPersistError(f"缓冲写盘失败：{exc}") from exc
