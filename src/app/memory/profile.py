"""V2.3 用户档案 profile（契约 §2.5）。

4 个白名单槽位常驻注入（≤10% 预算），**不参与向量召回**。
槽位固定：称呼 / 禁忌 / 关系阶段 / 重要日期。
"""

from __future__ import annotations

import logging

from app.memory.store import _PROFILE_SLOTS, MemoryStore

logger = logging.getLogger(__name__)

PROFILE_SLOTS = _PROFILE_SLOTS
PROFILE_HEADER = "【客人档案】"


async def update_profile(
    store: MemoryStore, agent_id: int | str, profile: dict[str, str]
) -> dict[str, str]:
    """只写白名单槽位；同槽位覆盖；返回写入后的完整档案。"""
    for slot, value in (profile or {}).items():
        if slot not in PROFILE_SLOTS:
            logger.warning("忽略未知 profile 槽位：%r", slot)
            continue
        await store.put_profile(agent_id, slot, str(value or ""))
    return await store.get_profile(agent_id)


async def render_profile(profile: dict[str, str]) -> str:
    """渲染档案注入段；空档案返回空串。"""
    lines = [
        f"- {slot}：{profile[slot]}"
        for slot in PROFILE_SLOTS
        if str(profile.get(slot) or "").strip()
    ]
    if not lines:
        return ""
    return f"{PROFILE_HEADER}\n" + "\n".join(lines)
