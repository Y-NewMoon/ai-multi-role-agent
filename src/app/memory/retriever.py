"""记忆召回（方案 3.2 语义记忆 + V2.3 加权召回 §2.6）。

- `retrieve`：V2.1/V2.2 既有列表召回（`golden.memory_hit_rate` 依赖，语义不变）。
- `retrieve_weighted`：V2.3 应用层加权重排（overscan=50），并给召回文本加
  「用户说过 / 你说过」角色前缀，解决「分不清谁说的」。
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from app.memory.store import MemoryStore


async def retrieve(
    memory_store: MemoryStore,
    agent_id: int | str,
    query: str | None = None,
    limit: int = 5,
) -> list[str]:
    return await memory_store.search(agent_id, query=query, limit=limit)


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _recency(created_at: object, now: datetime, halflife_days: float) -> float:
    created = _parse_dt(created_at)
    if created is None:
        return 0.5  # 旧数据缺 created_at：给中性值，避免整批旧记忆全灭
    delta_days = max(0.0, (now - created).total_seconds() / 86400.0)
    if halflife_days <= 0:
        return 1.0
    return math.exp(-delta_days / halflife_days)


def _cosine(score: object) -> float:
    """Store 的 `SearchItem.score` 即 cosine similarity（后端统一为 1 - distance）。"""
    if score is None:
        return 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _label(text: str, role: object) -> str:
    """渲染角色前缀（标注只在渲染时加，不进向量、不影响 score）。"""
    if role == "user":
        return f"用户说过：{text}"
    if role == "assistant":
        return f"你说过：{text}"
    return text


async def retrieve_weighted(
    memory_store: MemoryStore,
    agent_id: int | str,
    *,
    query: str | None,
    limit: int = 5,
    overscan: int = 50,
    w_cosine: float = 0.6,
    w_recency: float = 0.25,
    w_importance: float = 0.15,
    halflife_days: float = 14.0,
    now: datetime | None = None,
) -> list[str]:
    """加权召回：`0.6*cos + 0.25*recency + 0.15*importance`。

    - 过滤顺序：kind → 群聊他人发言 → superseded（见 `MemoryStore.search_scored`）；
    - 排序：score 降序 → updated_at 降序 → key 升序（确定性）；
    - `query is None` → 退化为 `current_facts` + 最近记忆（不含加权）；
    - `memory_weighted_recall=false` 时调用方传 `w_cosine=1`、其余 0，标注规则不变。
    """
    if query is None:
        facts = await memory_store.current_facts(agent_id, limit=limit)
        out = [str(f.get("text") or "") for f in facts if f.get("text")]
        for text in await memory_store.search(agent_id, query=None, limit=limit):
            if text not in out:
                out.append(text)
        return out[:limit]

    candidates = await memory_store.search_scored(
        agent_id, query=query, limit=limit, overscan=overscan
    )
    now = now or datetime.now(UTC)
    scored: list[tuple[float, dict]] = []
    for item in candidates:
        cos = _cosine(item.get("score"))
        importance = float(item.get("importance", 0.5))
        recency = _recency(item.get("created_at"), now, halflife_days)
        score = w_cosine * cos + w_recency * recency + w_importance * importance
        scored.append((score, item))

    scored.sort(key=lambda pair: str(pair[1].get("key") or ""))
    scored.sort(key=lambda pair: str(pair[1].get("updated_at") or ""), reverse=True)
    scored.sort(key=lambda pair: -pair[0])

    return [
        _label(str(item.get("text") or ""), item.get("role"))
        for _, item in scored[: max(0, limit)]
    ]
