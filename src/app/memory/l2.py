"""V2.3 L2 层级摘要（契约 §2.7）。

每成功滚动合并一次 L1（≈10 轮）计数 +1；每攒够 `every_merges` 次（≈30 轮）把
「现有 L2 + 当前 L1 + 高 importance facts」重合成一段 ≤300 字的关系档案。
计数按 agent 存（`l2:counter` 不掺 session）→ 私聊/群聊推进同一条 L2。
"""

from __future__ import annotations

import asyncio
import logging

from app.llm.base import LLMProvider
from app.memory.store import MemoryStore
from app.prompts.l2_summary import L2_PROMPT, L2_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _facts_block(facts: list[dict]) -> str:
    lines = [f"- {f.get('text', '')}" for f in facts if f.get("text")]
    return "\n".join(lines) if lines else "（无）"


async def _synthesize(
    store: MemoryStore,
    agent_id: int | str,
    llm: LLMProvider,
    *,
    current_l1: str,
    max_chars: int,
) -> str | None:
    existing = await store.get_l2(agent_id)
    facts = await store.current_facts(agent_id, limit=8)
    messages = [
        {"role": "system", "content": L2_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": L2_PROMPT.format(
                existing_l2=existing or "（空）",
                current_l1=current_l1 or "（空）",
                facts=_facts_block(facts),
            ),
        },
    ]
    try:
        raw = await asyncio.wait_for(llm.generate(messages), timeout=10.0)
    except Exception as exc:  # noqa: BLE001 - 合成失败不得阻断回复（D8）
        logger.warning("L2 合成失败（agent=%s）：%s", agent_id, exc)
        return None
    text = (raw or "").strip()
    if not text:
        logger.warning("L2 合成返回空文本（agent=%s），计数不归零", agent_id)
        return None
    text = text[:max_chars]
    await store.put_l2(agent_id, text)
    await store.reset_l2_counter(agent_id)
    return text


async def maybe_update_l2(
    store: MemoryStore,
    agent_id: int | str,
    llm: LLMProvider,
    *,
    current_l1: str,
    every_merges: int = 3,
    max_chars: int = 300,
) -> str | None:
    """一次 L1 滚动合并之后调用：计数 +1；达到 every_merges → 重合成 L2 并返回。"""
    merges = await store.bump_l2_counter(agent_id)
    if merges < max(1, every_merges):
        return None
    return await _synthesize(
        store, agent_id, llm, current_l1=current_l1, max_chars=max_chars
    )


async def force_update_l2(
    store: MemoryStore,
    agent_id: int | str,
    llm: LLMProvider,
    *,
    current_l1: str = "",
    max_chars: int = 300,
) -> str | None:
    """不看计数，立即重合成（`compact_memory.py --force` 与测试用）。"""
    return await _synthesize(
        store, agent_id, llm, current_l1=current_l1, max_chars=max_chars
    )
