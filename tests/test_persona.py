"""端到端人设贯通测试（真库）：AgentService 建档 → SessionService 建会话 → ChatService 对话。

验证 M3 的接缝：会话以其 `agent_id` 关联角色，角色的 persona_prompt 最终进入 LLM prompt。
"""

from sqlalchemy import delete

from app.agents.graph import build_graph
from app.core.errors import NotFoundError
from app.db.models import Agent, Message
from app.db.models import Session as ChatSession
from app.memory.namespace import memory_namespace
from app.memory.store import MemoryStore
from app.schemas.agent import AgentCreate
from app.schemas.session import SessionCreate
from app.services.agent_service import AgentService
from app.services.chat_service import ChatService
from app.services.session_service import SessionService

_PERSONA = "你是一位只谈茶道的江南茶艺师，自称「小满」。"


class _RecordingLLM:
    """固定回复并记录收到的 prompt（M3 起 generate 节点走 astream）。"""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def generate(self, messages):
        self.calls.append(messages)
        return "好的"

    async def astream(self, messages):
        self.calls.append(messages)
        yield "好的"


async def test_agent_persona_reaches_llm_prompt(
    real_session_factory, real_store, real_checkpointer
):
    ms = MemoryStore(real_store)
    agent_service = AgentService(
        session_factory=real_session_factory, memory_store=ms, checkpointer=real_checkpointer
    )
    session_service = SessionService(
        session_factory=real_session_factory, memory_store=ms, checkpointer=real_checkpointer
    )

    agent = await agent_service.create(AgentCreate(name="小满", persona_prompt=_PERSONA))
    session = await session_service.create(SessionCreate(agent_id=agent.id))
    thread = f"{session.id}:{agent.id}"

    llm = _RecordingLLM()
    chat_service = ChatService(
        graph=build_graph(
            llm=llm, memory_store=ms, checkpointer=real_checkpointer, store=real_store
        ),
        session_factory=real_session_factory,
        memory_store=ms,
    )
    try:
        resp = await chat_service.chat(session_id=session.id, message="在吗")

        assert resp.agent_id == agent.id  # 以 session.agent_id 为准
        assert llm.calls[-1][0]["role"] == "system"
        assert _PERSONA in llm.calls[-1][0]["content"]  # 人设进入 system prompt
    finally:
        async with real_session_factory() as db:
            await db.execute(delete(Message).where(Message.session_id == session.id))
            await db.execute(delete(ChatSession).where(ChatSession.id == session.id))
            await db.execute(delete(Agent).where(Agent.id == agent.id))
            await db.commit()
        await real_checkpointer.adelete_thread(thread)
        for item in await real_store.asearch(memory_namespace(agent.id), limit=1000):
            await real_store.adelete(item.namespace, item.key)


async def test_chat_missing_session_raises_not_found(real_session_factory, real_store, real_checkpointer):
    ms = MemoryStore(real_store)
    chat_service = ChatService(
        graph=build_graph(
            llm=_RecordingLLM(),
            memory_store=ms,
            checkpointer=real_checkpointer,
            store=real_store,
        ),
        session_factory=real_session_factory,
        memory_store=ms,
    )
    try:
        await chat_service.chat(session_id=10**9, message="在吗")
    except NotFoundError:
        return
    raise AssertionError("应抛出 NotFoundError")
