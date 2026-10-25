"""V3 M2：服务端滚动缓冲单测（原子写 / 重载 / 损坏隔离 / 上限 / 消息 id / turn_count）。"""

from __future__ import annotations

import json

import pytest

from app.services.memory_api_buffer import (
    BufferOverflowError,
    MemoryApiBuffer,
    build_turn_messages,
)


async def test_load_missing_file_starts_empty(tmp_path):
    buffer = MemoryApiBuffer(tmp_path / "buffer.json", max_messages=10)
    await buffer.load()
    assert buffer.size("ext") == 0
    assert buffer.last_turn_at("ext") is None
    assert buffer.turn_count("ext", "main") == 0


async def test_save_is_atomic_and_reloads(tmp_path):
    path = tmp_path / "buffer.json"
    buffer = MemoryApiBuffer(path, max_messages=10)
    await buffer.load()
    slot = buffer.slot("ext", "main")
    slot["messages"].extend(
        build_turn_messages(user_text="今天有点累", user_id="m-1", replies=["早点休息。"], at="2026-10-25T10:35:00+08:00")
    )
    slot["turn_count"] = 1
    slot["last_turn_at"] = "2026-10-25T10:35:00+08:00"
    await buffer.save()

    # 不残留临时文件；文件是合法 JSON
    assert [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name] == []
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["agents"]["ext"]["sessions"]["main"]["turn_count"] == 1

    reloaded = MemoryApiBuffer(path, max_messages=10)
    await reloaded.load()
    assert reloaded.turn_count("ext", "main") == 1
    assert reloaded.size("ext", "main") == 2
    assert reloaded.last_turn_at("ext", "main") == "2026-10-25T10:35:00+08:00"
    assert [m["id"] for m in reloaded.messages("ext", "main")] == ["m-1", "m-1:a0"]


async def test_corrupt_file_is_quarantined(tmp_path):
    path = tmp_path / "buffer.json"
    path.write_text("{ not json", encoding="utf-8")

    buffer = MemoryApiBuffer(path, max_messages=10)
    await buffer.load()

    assert not path.exists()
    assert list(tmp_path.glob("buffer.json.corrupt-*")), "坏文件应被改名隔离"
    assert buffer.size("ext") == 0


async def test_overflow_raises_without_dropping(tmp_path):
    buffer = MemoryApiBuffer(tmp_path / "buffer.json", max_messages=2)
    await buffer.load()
    buffer.ensure_capacity("ext", "main")  # 空槽，正常
    slot = buffer.slot("ext", "main")
    slot["messages"].extend(build_turn_messages(user_text="第一轮", user_id="m-1", replies=["回复。"], at="2026-10-25T10:00:00+08:00"))

    slot["messages"].extend(build_turn_messages(user_text="第二轮", user_id="m-2", replies=["回复。"], at="2026-10-25T10:01:00+08:00"))

    with pytest.raises(BufferOverflowError):
        buffer.ensure_capacity("ext", "main")
    assert buffer.size("ext", "main") == 4  # 不静默删（追加方须先检查）


async def test_capacity_counts_incoming_messages(tmp_path):
    """R3：上限按本轮实际追加条数判断，多回复轮不得静默突破。"""
    buffer = MemoryApiBuffer(tmp_path / "buffer.json", max_messages=3)
    await buffer.load()
    slot = buffer.slot("ext", "main")
    slot["messages"].extend(
        build_turn_messages(user_text="第一轮", user_id="m-1", replies=[], at="2026-10-25T10:00:00+08:00")
    )

    buffer.ensure_capacity("ext", "main", incoming=1)  # 2 + 1 ≤ 3
    with pytest.raises(BufferOverflowError):
        buffer.ensure_capacity("ext", "main", incoming=3)  # 2 + 3 > 3
    assert buffer.size("ext", "main") == 1


def test_build_turn_messages_ids_and_at_rules():
    messages = build_turn_messages(
        user_text="今天有点累",
        user_id="m-0001",
        replies=["那就早点休息。", "别熬夜。"],
        at="2026-10-25T10:35:00+08:00",
    )
    assert [m["id"] for m in messages] == ["m-0001", "m-0001:a0", "m-0001:a1"]
    assert messages[0]["role"] == "user"
    assert messages[0]["at"] == "2026-10-25T10:35:00+08:00"
    assert all(m["role"] == "assistant" for m in messages[1:])
    assert all("at" not in m for m in messages[1:])  # M1 D4：AI 回复不带 at


async def test_turn_count_and_agent_aggregate(tmp_path):
    buffer = MemoryApiBuffer(tmp_path / "buffer.json", max_messages=100)
    await buffer.load()
    for session, stamp, turns in [("main", "2026-10-25T10:00:00+08:00", 2), ("alt", "2026-10-26T10:00:00+08:00", 3)]:
        slot = buffer.slot("ext", session)
        slot["turn_count"] = turns
        slot["last_turn_at"] = stamp
        slot["messages"].extend(
            message
            for i in range(turns)
            for message in build_turn_messages(
                user_text=f"{session}-{i}", user_id=f"{session}-{i}", replies=["回复。"], at=stamp
            )
        )
    await buffer.save()

    assert buffer.size("ext", "main") == 4
    assert buffer.size("ext", "alt") == 6
    assert buffer.size("ext") == 10  # 省略 session → 聚合
    assert buffer.last_turn_at("ext") == "2026-10-26T10:00:00+08:00"  # 取最大
    assert buffer.turn_count("ext", "alt") == 3


async def test_remove_message_ids_reports_missing(tmp_path):
    buffer = MemoryApiBuffer(tmp_path / "buffer.json", max_messages=100)
    await buffer.load()
    slot = buffer.slot("ext", "main")
    slot["messages"].extend(build_turn_messages(user_text="你好", user_id="m-1", replies=["你也好。"], at="2026-10-25T10:00:00+08:00"))

    removed, missing = buffer.remove_message_ids("ext", "main", ["m-1:a0", "not-found"])
    assert removed == 1
    assert missing == ["not-found"]
    assert [m["id"] for m in buffer.messages("ext", "main")] == ["m-1"]


async def test_has_message(tmp_path):
    buffer = MemoryApiBuffer(tmp_path / "buffer.json", max_messages=10)
    await buffer.load()
    slot = buffer.slot("ext", "main")
    slot["messages"].extend(build_turn_messages(user_text="你好", user_id="m-1", replies=[], at="2026-10-25T10:00:00+08:00"))
    assert buffer.has_message("ext", "main", "m-1")
    assert not buffer.has_message("ext", "main", "m-2")
