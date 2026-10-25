"""节点：从该角色 namespace 召回语义记忆，并取出本会话滚动摘要 + V2.3 三段。

M2：以本轮最后一条用户消息为 query 做 top-k 向量召回（方案 3.2 语义记忆）；
摘要（方案 3.10）单独取出，供 compose_prompt 高优先级注入。

V2.3（§2.9）：`memory_extract_enabled` 为真时额外取 profile / current_facts / L2，
并改用加权召回（`memory_weighted_recall=false` 时退化为纯余弦）。
开关关闭时不取、不做任何额外调用，行为与 V2.2 完全一致。
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage

from app.agents.state import RolePlayState
from app.memory.retriever import retrieve, retrieve_weighted
from app.memory.store import MemoryStore
from app.observability.trace import current_trace

logger = logging.getLogger(__name__)


def _latest_user_query(messages: list) -> str | None:
    """取本轮最后一条用户消息作为检索 query；无则返回 None（退化为列表召回）。"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            text = str(m.content).strip()
            if text:
                return text
    return None


async def retrieve_memory(
    state: RolePlayState,
    memory_store: MemoryStore,
    *,
    limit: int = 5,
    extract_enabled: bool = False,
    weighted: bool = True,
    profile_enabled: bool = True,
    l2_enabled: bool = True,
    w_cosine: float = 0.6,
    w_recency: float = 0.25,
    w_importance: float = 0.15,
    halflife_days: float = 14.0,
    overscan: int = 50,
    now: datetime | None = None,
    debug: bool = False,
) -> dict:
    """召回记忆；`now`/`debug` 为 M2 加性参数，图内调用不传，行为与 M1 一致。"""
    messages = state.get("messages", [])
    query = _latest_user_query(messages)
    agent_id = state["agent_id"]

    result: dict = {}
    debug_out: dict | None = {"pool": 0, "top": []} if debug else None
    if extract_enabled:
        recalled = await retrieve_weighted(
            memory_store,
            agent_id,
            query=query,
            limit=limit,
            overscan=overscan,
            w_cosine=w_cosine if weighted else 1.0,
            w_recency=w_recency if weighted else 0.0,
            w_importance=w_importance if weighted else 0.0,
            halflife_days=halflife_days,
            now=now,
            debug_out=debug_out,
        )
        result["recalled_memories"] = recalled
        result["recalled_facts"] = await memory_store.current_facts(agent_id, limit=8)
        result["user_profile"] = (
            await memory_store.get_profile(agent_id) if profile_enabled else {}
        )
        result["l2_portrait"] = (await memory_store.get_l2(agent_id)) if l2_enabled else ""
    else:
        result["recalled_memories"] = await retrieve(memory_store, agent_id, query=query, limit=limit)

    result["summary"] = await memory_store.get_summary(agent_id, state["session_id"])
    if debug:
        result["recalled_debug"] = debug_out or {"pool": 0, "top": []}
    # V2.4：只记录召回条数；lore 由 retrieve_lore 补全
    recorder = current_trace()
    if recorder is not None:
        recorder.memory_count(len(result.get("recalled_memories") or []))
        # C-03 观察项：默认关闭，开启时每轮多一次 asearch(limit=1000)；失败 fail-open
        if _memory_stats_enabled():
            try:
                stats = await memory_store.stats(agent_id)
            except Exception as exc:  # noqa: BLE001 - 旁路统计不得影响主流程
                logger.warning("trace memory_stats 统计失败（已忽略）：%s", exc)
            else:
                recorder.memory_snapshot(**stats)
    return result


def _memory_stats_enabled() -> bool:
    try:
        from app.config import get_settings

        return bool(get_settings().trace_memory_stats)
    except Exception:  # noqa: BLE001 - 配置不可读时按关闭处理
        return False
