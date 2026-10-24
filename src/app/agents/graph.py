"""LangGraph 编排：load_persona → retrieve_memory → retrieve_lore → compose_prompt → generate → update_memory。

接缝 3（方案 3.11A#3）：build_graph() 是**可实例化工厂**，每次返回独立图，不是全局单例
——群聊时为每个角色各起一张图。
接缝 5：依赖（LLM / Store / checkpointer）全部由外部注入，图内节点零业务表访问。
M2：预算/摘要参数由外部注入（来自 Settings），便于测试与调参。
V2.1：新增 retrieve_lore 节点（世界书设定召回）；retriever 为空或关闭时不访问数据库。
V2.2：`skill_registry` 非空时追加工具循环 `generate → (条件) → tools → generate`，最多
`max_tool_rounds` 轮；`skill_registry=None` 时**保持 V2.1 拓扑**（不出现 `tools` 节点）。
V2.2：`lore_static_mode` 分派检索静态注入策略（§2.10 三策略）：
- `full`：V2.1 原路径（always-on + keyword + vector + RRF + rerank + 预算）；
- `always_on`：只召回 always-on chunk（`source_ref` 升序、8% 预算），不走向量/关键词/rerank；
- `off`：直接返回空，不访问数据库。
检索**算法本体未改**：`always_on` 复用 `Retriever` 既有的 always-on 取数与预算方法。
"""

import time

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from app.agents.nodes.compose_prompt import compose_prompt
from app.agents.nodes.generate import generate
from app.agents.nodes.load_persona import load_persona
from app.agents.nodes.retrieve_lore import retrieve_lore
from app.agents.nodes.retrieve_memory import retrieve_memory
from app.agents.nodes.tools import execute_tools
from app.agents.nodes.update_memory import update_memory
from app.agents.state import RolePlayState
from app.llm.base import LLMProvider
from app.lore.retriever import Retriever
from app.memory.store import MemoryStore
from app.observability.trace import current_trace
from app.skills.executor import ToolExecutor
from app.skills.registry import SkillRegistry


def _traced(name: str, fn):
    """节点 trace 包装：只记录 `(name, ms, ok)`，返回值/异常原样透传（§2.3）。"""

    async def wrapped(state: RolePlayState) -> dict:
        started = time.perf_counter()
        ok = True
        try:
            return await fn(state)
        except BaseException:  # noqa: BLE001 - 记录失败后原样抛出
            ok = False
            raise
        finally:
            recorder = current_trace()
            if recorder is not None:
                recorder.node(
                    name, int((time.perf_counter() - started) * 1000), ok
                )

    return wrapped


def should_continue(state: RolePlayState, *, max_tool_rounds: int = 3) -> str:
    """有 tool_calls 且未达轮数上限 → `tools`；否则 `update_memory`（契约 §2.4）。"""
    if state.get("tool_calls") and int(state.get("tool_rounds") or 0) < max_tool_rounds:
        return "tools"
    return "update_memory"


async def _always_on_lore(retriever: Retriever, agent_id: int) -> list[dict]:
    """`lore_static_mode="always_on"`：只召回 always-on chunk（§2.10 hybrid 策略）。

    复用 `Retriever` 既有的 always-on 取数（`_route_always_on` / `_load_rows`）与预算
    （`_fit_budget`）方法，**不改其算法**；只跑 always-on 一路，不走向量/关键词/rerank，
    也不触发 query 改写——检索细节由 `read_lore` 工具另行提供。
    """
    async with retriever._session_factory() as db:  # noqa: SLF001 - 复用既有取数逻辑
        refs = await retriever._route_always_on(db, agent_id)  # noqa: SLF001
    rows = await retriever._load_rows(agent_id=agent_id, refs=set(refs))  # noqa: SLF001
    budget = getattr(retriever, "_budget_tokens", 8000)
    injected, _ = retriever._fit_budget(  # noqa: SLF001
        routes={"always_on": refs, "keyword": [], "vector": []},
        reranked=[],
        rows=rows,
        always_cap=int(budget * getattr(retriever, "_always_on_ratio", 0.08)),
        lore_cap=int(budget * getattr(retriever, "_lore_ratio", 0.20)),
    )
    return [hit.as_recalled() for hit in injected]


def build_graph(
    *,
    llm: LLMProvider,
    memory_store: MemoryStore,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    token_budget: int = 8000,
    recent_rounds: int = 10,
    summary_trigger_rounds: int = 10,
    summary_chunk_rounds: int = 5,
    reply_max_messages: int = 3,
    reply_think_retry_enabled: bool = True,
    lore_retriever: Retriever | None = None,
    lore_enabled: bool = True,
    lore_max_ratio: float = 0.20,
    lore_always_on_ratio: float = 0.08,
    lore_static_mode: str = "full",
    skill_registry: SkillRegistry | None = None,
    tool_executor: ToolExecutor | None = None,
    allowed_skills: list[str] | None = None,
    max_tool_rounds: int = 3,
    # V2.3 记忆升级（全部默认关闭/中性，保证既有调用方行为不变）
    memory_extract_enabled: bool = False,
    memory_extract_min_chars: int = 8,
    memory_extract_timeout_s: float = 8.0,
    memory_fact_mode: str = "overwrite",
    memory_profile_enabled: bool = True,
    memory_profile_ratio: float = 0.10,
    memory_weighted_recall: bool = True,
    memory_w_cosine: float = 0.6,
    memory_w_recency: float = 0.25,
    memory_w_importance: float = 0.15,
    memory_recency_halflife_days: float = 14.0,
    memory_recall_overscan: int = 50,
    memory_recall_limit: int = 5,
    memory_l2_enabled: bool = True,
    memory_l2_ratio: float = 0.05,
    memory_l2_every_merges: int = 3,
    memory_l2_max_chars: int = 300,
):
    """构造并编译一张 StateGraph。"""
    workflow = StateGraph(RolePlayState)

    async def _load_persona(state: RolePlayState) -> dict:
        return await load_persona(state)

    async def _retrieve_memory(state: RolePlayState) -> dict:
        return await retrieve_memory(
            state,
            memory_store,
            limit=memory_recall_limit,
            extract_enabled=memory_extract_enabled,
            weighted=memory_weighted_recall,
            profile_enabled=memory_profile_enabled,
            l2_enabled=memory_l2_enabled,
            w_cosine=memory_w_cosine,
            w_recency=memory_w_recency,
            w_importance=memory_w_importance,
            halflife_days=memory_recency_halflife_days,
            overscan=memory_recall_overscan,
        )

    async def _retrieve_lore(state: RolePlayState) -> dict:
        # V2.2 §2.10：按 lore_static_mode 分派；off/always_on 均不走向量/关键词/rerank
        if not lore_enabled or lore_retriever is None or lore_static_mode == "off":
            return {"recalled_lore": []}
        if lore_static_mode == "always_on":
            return {
                "recalled_lore": await _always_on_lore(
                    lore_retriever, int(state["agent_id"])
                )
            }
        return await retrieve_lore(state, lore_retriever, enabled=True)

    async def _compose_prompt(state: RolePlayState) -> dict:
        return await compose_prompt(
            state,
            token_budget=token_budget,
            recent_rounds=recent_rounds,
            lore_max_ratio=lore_max_ratio,
            lore_always_on_ratio=lore_always_on_ratio,
            profile_ratio=memory_profile_ratio,
            l2_ratio=memory_l2_ratio,
            skill_registry=skill_registry,
            allowed_skills=allowed_skills,
        )

    async def _generate(state: RolePlayState) -> dict:
        return await generate(
            state,
            llm,
            max_messages=reply_max_messages,
            skill_registry=skill_registry,
            max_tool_rounds=max_tool_rounds,
            allowed_skills=allowed_skills,
            think_retry_enabled=reply_think_retry_enabled,
        )

    async def _update_memory(state: RolePlayState) -> dict:
        return await update_memory(
            state,
            memory_store,
            llm,
            trigger_rounds=summary_trigger_rounds,
            chunk_rounds=summary_chunk_rounds,
            extract_enabled=memory_extract_enabled,
            extract_min_chars=memory_extract_min_chars,
            extract_timeout_s=memory_extract_timeout_s,
            fact_mode=memory_fact_mode,
            # 不变量 7：全局开关关闭时不得写 profile/L2 计数器（否则 prompt/Store 都变了）
            profile_enabled=memory_extract_enabled and memory_profile_enabled,
            l2_enabled=memory_extract_enabled and memory_l2_enabled,
            l2_every_merges=memory_l2_every_merges,
            l2_max_chars=memory_l2_max_chars,
        )

    workflow.add_node("load_persona", _traced("load_persona", _load_persona))
    workflow.add_node("retrieve_memory", _traced("retrieve_memory", _retrieve_memory))
    workflow.add_node("retrieve_lore", _traced("retrieve_lore", _retrieve_lore))
    workflow.add_node("compose_prompt", _traced("compose_prompt", _compose_prompt))
    workflow.add_node("generate", _traced("generate", _generate))
    workflow.add_node("update_memory", _traced("update_memory", _update_memory))

    workflow.add_edge(START, "load_persona")
    workflow.add_edge("load_persona", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "retrieve_lore")
    workflow.add_edge("retrieve_lore", "compose_prompt")
    workflow.add_edge("compose_prompt", "generate")

    if skill_registry is not None:
        executor = tool_executor or ToolExecutor(
            skill_registry, max_tool_rounds=max_tool_rounds
        )

        async def _tools(state: RolePlayState) -> dict:
            return await execute_tools(state, executor, allowed_skills=allowed_skills)

        workflow.add_node("tools", _traced("tools", _tools))

        def _route(state: RolePlayState) -> str:
            return should_continue(state, max_tool_rounds=max_tool_rounds)

        workflow.add_conditional_edges(
            "generate", _route, {"tools": "tools", "update_memory": "update_memory"}
        )
        workflow.add_edge("tools", "generate")
    else:
        # V2.1 拓扑：不添加 tools / 条件路由，直接收尾
        workflow.add_edge("generate", "update_memory")

    workflow.add_edge("update_memory", END)

    return workflow.compile(checkpointer=checkpointer, store=store)
