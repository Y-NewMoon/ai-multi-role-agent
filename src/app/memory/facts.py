"""V2.3 事实与冲突（契约 §2.4 / §2.8）。

- `record_facts`：逐条落 Store；单条失败只记 log，不影响其它条。
- `render_facts`：把当前事实渲染成注入段（≤8 条，importance 降序）。
- 冲突判定只看 `(subject, predicate)` 精确相等（同义归一等 V2.4/V2.5）。
"""

from __future__ import annotations

import logging

from app.memory.extractor import Extraction
from app.memory.store import FactWrite, MemoryStore

logger = logging.getLogger(__name__)

FACTS_HEADER = "【你记得的事】"
DEFAULT_RENDER_LIMIT = 8


async def record_facts(
    store: MemoryStore,
    agent_id: int | str,
    extraction: Extraction,
    *,
    source_message_id: str | None,
    turn_id: str,
    mode: str = "overwrite",
) -> list[FactWrite]:
    """把抽出的事实逐条写入 Store；返回每条写入结果。"""
    writes: list[FactWrite] = []
    for fact in extraction.facts:
        try:
            write = await store.put_fact(
                agent_id,
                subject=fact.subject,
                predicate=fact.predicate,
                obj=fact.object,
                importance=fact.importance / 5.0,  # LLM 1~5 分 /5
                source_message_id=source_message_id,
                turn_id=turn_id,
                mode=mode,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 - 单条失败不影响其它条
            logger.warning(
                "写入事实失败（agent=%s, %s/%s）：%s",
                agent_id,
                fact.subject,
                fact.predicate,
                exc,
            )
            continue
        writes.append(write)
    return writes


def _fact_line(fact: dict, *, note: bool) -> str:
    text = str(fact.get("text") or "")
    if not text:
        text = f"{fact.get('subject', '')}{fact.get('predicate', '')}={fact.get('object', '')}"
    suffix = f"（你以前说{fact.get('object', '')}）" if note else ""
    return f"- {text}{suffix}"


async def render_facts(facts: list[dict], *, limit: int = DEFAULT_RENDER_LIMIT) -> str:
    """渲染 facts 注入段（空列表 → 空串，调用方不得插入任何字符）。

    `keep_both_with_note` 模式下同一 `(subject, predicate)` 会有多条 active：
    最新一条正常渲染，**最近一次**被取代的那条带「（你以前说 X）」注，更早的不注入。
    """
    if not facts or limit <= 0:
        return ""

    groups: dict[tuple[str, str], list[dict]] = {}
    for fact in facts:
        key = (str(fact.get("subject") or ""), str(fact.get("predicate") or ""))
        groups.setdefault(key, []).append(fact)

    selected: list[tuple[dict, bool]] = []
    for group in groups.values():
        group.sort(key=lambda f: str(f.get("updated_at") or ""), reverse=True)
        selected.append((group[0], False))
        if len(group) > 1:
            selected.append((group[1], True))

    # 先按 updated_at 降序，再按 importance 降序（稳定排序保证同分时新的在前）
    selected.sort(key=lambda pair: str(pair[0].get("updated_at") or ""), reverse=True)
    selected.sort(key=lambda pair: -float(pair[0].get("importance", 0.5)))
    selected = selected[:limit]

    lines = [_fact_line(fact, note=note) for fact, note in selected]
    return f"{FACTS_HEADER}\n" + "\n".join(lines)
