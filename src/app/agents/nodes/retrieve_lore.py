"""节点：世界书设定召回（V2.1 契约 4.8）。

以本轮最后一条用户消息为 query，经三路召回 + RRF + rerank + 预算，
把最终注入的 lore chunk 写入 `state.recalled_lore`，供 compose_prompt 拼进 system。

- `retriever` 为 None 或 `enabled=False`：直接返回 `{"recalled_lore": []}`（**不访问数据库**）；
- 无用户消息（自主闲聊轮）：query 取空串，仍走 always-on 路，保证核心设定常驻；
- 无 lore 文档时，retriever 返回空注入 → 行为与 V2.0 完全一致。
"""

import time

from langchain_core.messages import BaseMessage, HumanMessage

from app.agents.nodes.retrieve_memory import _latest_user_query
from app.agents.state import RolePlayState
from app.lore.retriever import Retriever
from app.observability.trace import current_trace


def _history(messages: list[BaseMessage]) -> list[str]:
    """本轮之前的用户消息纯文本（**不含本轮当前用户消息**），供 query 改写参考（契约 4.6）。

    图状态里 `messages[-1]` 即本轮用户输入；若不排除，`needs_rewrite` 第三条
    「上一轮有未消解指代」会退化为第一条「本轮含代词」而永不独立触发。
    只保留用户消息，使 `history[-1]` 恰为"上一轮用户消息"。
    """
    user_texts = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
    return user_texts[:-1][-4:]


async def retrieve_lore(
    state: RolePlayState,
    retriever: Retriever | None = None,
    *,
    enabled: bool = True,
) -> dict:
    if not enabled or retriever is None:
        return {"recalled_lore": []}

    messages = state.get("messages", [])
    query = _latest_user_query(messages) or ""
    started = time.perf_counter()
    result = await retriever.retrieve(
        agent_id=state["agent_id"],
        query=query,
        history=_history(messages),
    )
    # V2.4：检索旁路记录（不改检索算法）。Retriever 未暴露改写耗时，改写轮以
    # 本次 retrieve 总耗时作为上界（§5 偏差已说明）。
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    recorder = current_trace()
    if recorder is not None:
        recorder.retrieval(
            lore_ids=[hit.source_ref for hit in result.injected],
            memory_count=int(recorder.retrieval_stats.get("memory_count") or 0),
            rerank_top=len(result.reranked) if result.rerank and result.reranked else None,
            rewrite_used=bool(result.rewrite_used),
            rewrite_ms=elapsed_ms if result.rewrite_used else 0,
        )
    return {"recalled_lore": [hit.as_recalled() for hit in result.injected]}
