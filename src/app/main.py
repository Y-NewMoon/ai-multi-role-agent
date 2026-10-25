"""FastAPI 入口与生命周期。

启动顺序（方案 3.8）：alembic upgrade head（离线）→ 应用启动时 LangGraph setup()（幂等）。
Windows 提示：psycopg 异步需 Selector 事件循环，请用 scripts/run_server.py 启动。
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agents.graph import build_graph
from app.api.v1 import api_router
from app.config import get_settings
from app.core.clock import fixed_now_raw, validate_fixed_now
from app.core.error_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.db.base import create_engine, create_session_factory
from app.db.checkpointer import PooledAsyncPostgresSaver
from app.llm.factory import create_embedding_provider, create_llm_provider
from app.lore.indexer import Indexer
from app.lore.retriever import BgeReranker, LlmRewriter, Retriever
from app.lore.service import LoreService
from app.memory.store import MemoryStore
from app.observability.llm import InstrumentedLLM
from app.observability.trace import TraceStore, configure_trace_store
from app.services.agent_service import AgentService
from app.services.chat_service import ChatService
from app.services.group_service import GroupService
from app.services.reminder_service import ReminderService
from app.services.session_service import SessionService
from app.skills.builtin import build_default_registry
from app.skills.builtin.web_search import HttpSearchProvider
from app.skills.executor import ToolExecutor


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()

    # V3 M1：冻结时钟非法值启动即失败（不把错误带进请求链路）
    validate_fixed_now()
    frozen = fixed_now_raw()
    logging.getLogger(__name__).info(
        "EVAL_FIXED_NOW=%s（固定时钟%s）", frozen, "已生效" if frozen else "未设置"
    )

    # V2.4：trace 文件写入（按天 JSONL）；写失败 fail-open，不影响请求
    trace_store = TraceStore(
        directory=settings.trace_dir,
        enabled=settings.trace_enabled,
        sample_rate=settings.trace_sample_rate,
    )
    configure_trace_store(trace_store)

    # 头像静态目录（M5）：先确保存在，供 StaticFiles 与上传接口写入
    Path(settings.avatar_dir).mkdir(parents=True, exist_ok=True)

    # LangGraph checkpointer + store 共用同一连接池（psycopg3 异步）
    # R1：池大小需覆盖并发 SSE 数，否则 checkpoint/记忆查询在取连接处串行排队
    pool = AsyncConnectionPool(
        settings.database_url,
        open=False,
        min_size=max(1, settings.db_pool_min_size),
        max_size=max(1, settings.db_pool_min_size, settings.db_pool_max_size),
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    await pool.wait()

    checkpointer = PooledAsyncPostgresSaver(pool)
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

    memory_store = MemoryStore(store, fact_limit=settings.memory_fact_limit)
    # V2.4：全局信号量 + 超时兜底 + token/成本/trace 聚合（只包装，不改 provider 语义）
    # R3：fake 压测不得把 trace 标成真实模型/成本 → model="fake"、单价 0、usage_source="fake"
    is_fake = settings.llm_provider == "fake"
    llm = InstrumentedLLM(
        create_llm_provider(settings),
        model="fake" if is_fake else settings.llm_model,
        max_concurrency=settings.llm_max_concurrency,
        timeout_s=settings.llm_timeout_s,
        fallback=settings.llm_timeout_fallback,
        input_price_per_1m=0.0 if is_fake else settings.llm_price_input_cny_per_1m,
        output_price_per_1m=0.0 if is_fake else settings.llm_price_output_cny_per_1m,
        usage_source="fake" if is_fake else "estimator",
    )

    # 世界书（V2.1）：索引器与检索器共用同一嵌入器；rerank 懒加载、失败自动降级
    lore_indexer = Indexer(session_factory=session_factory, embedder=embed)
    lore_retriever = Retriever(
        session_factory=session_factory,
        embedder=embed,
        reranker=BgeReranker(),
        rewriter=LlmRewriter(llm),
        rewrite_enabled=settings.lore_query_rewrite,
        budget_tokens=settings.prompt_token_budget,
        lore_ratio=settings.lore_max_ratio,
        always_on_ratio=settings.lore_always_on_ratio,
        top_k=settings.lore_top_k,
    )

    # V2.2 Skill 工具层：read_lore 复用同一检索器但关闭 query 改写（D7：杜绝工具内再调 LLM）
    lore_tool_retriever = Retriever(
        session_factory=session_factory,
        embedder=embed,
        reranker=BgeReranker(),
        rewriter=None,
        rewrite_enabled=False,
        budget_tokens=settings.prompt_token_budget,
        lore_ratio=settings.lore_max_ratio,
        always_on_ratio=settings.lore_always_on_ratio,
        top_k=settings.lore_top_k,
    )
    # 5 个内置工具；搜索后端缺 URL/KEY 时 web_search 返回 `unavailable`（不阻塞启动）
    skill_registry = build_default_registry(
        search_provider=HttpSearchProvider(
            url=settings.web_search_url, api_key=settings.web_search_api_key
        ),
        session_factory=session_factory,
        lore_retriever=lore_tool_retriever,
    )
    tool_executor = ToolExecutor(skill_registry, max_tool_rounds=settings.skill_max_tool_rounds)

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
            reply_think_retry_enabled=settings.reply_think_retry_enabled,
            lore_retriever=lore_retriever,
            lore_enabled=settings.lore_enabled,
            lore_max_ratio=settings.lore_max_ratio,
            lore_always_on_ratio=settings.lore_always_on_ratio,
            lore_static_mode=settings.lore_static_mode,
            skill_registry=skill_registry,
            tool_executor=tool_executor,
            max_tool_rounds=settings.skill_max_tool_rounds,
            # V2.3 记忆升级
            memory_extract_enabled=settings.memory_extract_enabled,
            memory_extract_min_chars=settings.memory_extract_min_chars,
            memory_extract_timeout_s=settings.memory_extract_timeout_s,
            memory_fact_mode=settings.memory_fact_mode,
            memory_profile_enabled=settings.memory_profile_enabled,
            memory_profile_ratio=settings.memory_profile_ratio,
            memory_weighted_recall=settings.memory_weighted_recall,
            memory_w_cosine=settings.memory_w_cosine,
            memory_w_recency=settings.memory_w_recency,
            memory_w_importance=settings.memory_w_importance,
            memory_recency_halflife_days=settings.memory_recency_halflife_days,
            memory_recall_overscan=settings.memory_recall_overscan,
            memory_recall_limit=settings.memory_recall_limit,
            memory_l2_enabled=settings.memory_l2_enabled,
            memory_l2_ratio=settings.memory_l2_ratio,
            memory_l2_every_merges=settings.memory_l2_every_merges,
            memory_l2_max_chars=settings.memory_l2_max_chars,
        )

    graph = graph_factory()

    app.state.settings = settings
    app.state.trace_store = trace_store
    app.state.llm = llm
    app.state.graph = graph
    app.state.chat_service = ChatService(
        graph=graph,
        session_factory=session_factory,
        memory_store=memory_store,
        tool_executor=tool_executor,
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
        tool_executor=tool_executor,
    )
    app.state.lore_service = LoreService(
        session_factory=session_factory,
        indexer=lore_indexer,
        retriever=lore_retriever,
    )
    # V2.2：工具注册表（agents API 白名单校验用）与提醒服务（HITL API）
    app.state.skill_registry = skill_registry
    app.state.reminder_service = ReminderService(session_factory=session_factory)

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
async def health() -> dict[str, object]:
    # llm_provider 供 load_test 启动前校验“只压 fake”，不得据此做业务分支；
    # fake_llm_calls 为 provider 实际调用计数，供压测核对 delta（R9）
    # R8/R9：同时暴露压测 server 的真实库/嵌入/世界书/trace 配置（不含连接串与密钥）
    provider = getattr(getattr(app.state, "llm", None), "_provider", None)
    calls = getattr(provider, "calls", None)
    return {
        "status": "ok",
        "llm_provider": _settings.llm_provider,
        "fake_llm_calls": calls if isinstance(calls, int) else None,
        "database": _database_name(_settings.database_url),
        "embedding_provider": _settings.embedding_provider,
        "embedding_model": (
            _settings.embedding_model if _settings.embedding_provider == "local" else None
        ),
        "lore_enabled": _settings.lore_enabled,
        "lore_static_mode": _settings.lore_static_mode,
        # R1：压测必须能核对“同负载”，图开关一律随 /health 落进 manifest
        "memory_extract_enabled": _settings.memory_extract_enabled,
        "memory_profile_enabled": _settings.memory_profile_enabled,
        "memory_l2_enabled": _settings.memory_l2_enabled,
        "memory_weighted_recall": _settings.memory_weighted_recall,
        "trace_enabled": _settings.trace_enabled,
        "trace_dir": _settings.trace_dir,
        "trace_sample_rate": _settings.trace_sample_rate,
    }


def _database_name(url: str) -> str | None:
    """只取库名，避免把账号/密码写进 manifest。"""
    if not url or "/" not in url:
        return None
    return url.rsplit("/", 1)[-1].split("?", 1)[0] or None
