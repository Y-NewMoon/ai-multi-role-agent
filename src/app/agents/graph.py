"""LangGraph 编排：load_persona → retrieve_memory → compose_prompt → generate → update_memory。

接缝 3（方案 3.11A#3）：build_graph() 是**可实例化工厂**，每次返回独立图，不是全局单例
——群聊时为每个角色各起一张图。
接缝 5：依赖（LLM / Store / checkpointer）全部由外部注入，图内节点零业务表访问。
M2：预算/摘要参数由外部注入（来自 Settings），便于测试与调参。
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from app.agents.nodes.compose_prompt import compose_prompt
from app.agents.nodes.generate import generate
from app.agents.nodes.load_persona import load_persona
from app.agents.nodes.retrieve_memory import retrieve_memory
from app.agents.nodes.update_memory import update_memory
from app.agents.state import RolePlayState
from app.llm.base import LLMProvider
from app.memory.store import MemoryStore


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
):
    """构造并编译一张 StateGraph。"""
    workflow = StateGraph(RolePlayState)

    async def _load_persona(state: RolePlayState) -> dict:
        return await load_persona(state)

    async def _retrieve_memory(state: RolePlayState) -> dict:
        return await retrieve_memory(state, memory_store)

    async def _compose_prompt(state: RolePlayState) -> dict:
        return await compose_prompt(state, token_budget=token_budget, recent_rounds=recent_rounds)

    async def _generate(state: RolePlayState) -> dict:
        return await generate(state, llm, max_messages=reply_max_messages)

    async def _update_memory(state: RolePlayState) -> dict:
        return await update_memory(
            state,
            memory_store,
            llm,
            trigger_rounds=summary_trigger_rounds,
            chunk_rounds=summary_chunk_rounds,
        )

    workflow.add_node("load_persona", _load_persona)
    workflow.add_node("retrieve_memory", _retrieve_memory)
    workflow.add_node("compose_prompt", _compose_prompt)
    workflow.add_node("generate", _generate)
    workflow.add_node("update_memory", _update_memory)

    workflow.add_edge(START, "load_persona")
    workflow.add_edge("load_persona", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "compose_prompt")
    workflow.add_edge("compose_prompt", "generate")
    workflow.add_edge("generate", "update_memory")
    workflow.add_edge("update_memory", END)

    return workflow.compile(checkpointer=checkpointer, store=store)
