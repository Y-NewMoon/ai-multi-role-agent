"""FastAPI 入口与生命周期。

启动顺序（方案 3.8）：alembic upgrade head（离线）→ 应用启动时 LangGraph setup()（幂等）。
Windows 提示：psycopg 异步需 Selector 事件循环，请用 scripts/run_server.py 启动。
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agents.graph import build_graph
from app.api.v1 import api_router
from app.config import get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.db.base import create_engine, create_session_factory
from app.llm.factory import create_embedding_provider, create_llm_provider
from app.memory.store import MemoryStore
from app.services.agent_service import AgentService
from app.services.chat_service import ChatService
from app.services.group_service import GroupService
from app.services.session_service import SessionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()

    # 头像静态目录（M5）：先确保存在，供 StaticFiles 与上传接口写入
    Path(settings.avatar_dir).mkdir(parents=True, exist_ok=True)

    # LangGraph checkpointer + store 共用同一连接池（psycopg3 异步）
    pool = AsyncConnectionPool(
        settings.database_url,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    await pool.wait()

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    # 向量索引：dims 与 EMBEDDING_DIMS 一致（方案 3.6，建表后不可改）
    embed = create_embedding_provider(settings)
    store = AsyncPostgresStore(
        pool,
        index={"dims": settings.embedding_dims, "embed": embed, "fields": ["text"]},
    )
    await store.setup()

    engine = create_engine(settings.sqlalchemy_url)
    session_factory = create_session_factory(engine)

    memory_store = MemoryStore(store)
    llm = create_llm_provider(settings)

    def graph_factory():
        """每角色一张图（接缝 3）：群聊调度器按 agent_id 复用各自实例。"""
        return build_graph(
            llm=llm,
            memory_store=memory_store,
            checkpointer=checkpointer,
            store=store,
            token_budget=settings.prompt_token_budget,
            recent_rounds=settings.prompt_recent_rounds,
            summary_trigger_rounds=settings.summary_trigger_rounds,
            summary_chunk_rounds=settings.summary_chunk_rounds,
            reply_max_messages=settings.reply_max_messages,
        )

    graph = graph_factory()

    app.state.settings = settings
    app.state.graph = graph
    app.state.chat_service = ChatService(
        graph=graph, session_factory=session_factory, memory_store=memory_store
    )
    app.state.agent_service = AgentService(
        session_factory=session_factory, memory_store=memory_store, checkpointer=checkpointer
    )
    app.state.session_service = SessionService(
        session_factory=session_factory, memory_store=memory_store, checkpointer=checkpointer
    )
    app.state.group_service = GroupService(
        session_factory=session_factory,
        memory_store=memory_store,
        llm=llm,
        graph_factory=graph_factory,
        checkpointer=checkpointer,
        max_members=settings.group_max_members,
        max_turns=settings.group_max_turns,
        idle_max_messages=settings.idle_group_max_messages,
        idle_turn_max_messages=settings.idle_group_turn_max_messages,
    )

    yield

    await pool.close()
    await engine.dispose()


app = FastAPI(title="AI 多角色扮演智能体", lifespan=lifespan)
app.include_router(api_router)
register_exception_handlers(app)

# 头像静态服务（M5）：check_dir=False 允许目录在 lifespan 启动后再创建
_settings = get_settings()
app.mount(
    _settings.avatar_url_prefix,
    StaticFiles(directory=_settings.avatar_dir, check_dir=False),
    name="avatars",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
