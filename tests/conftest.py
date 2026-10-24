"""pytest 全局夹具。

- Windows：psycopg 异步需要 SelectorEventLoop，覆盖 pytest-asyncio 的事件循环策略。
- 真库夹具（方案 3.8）：隔离测试跑独立测试库 `roleplay_test`，
  用真 `AsyncPostgresStore` + pgvector 索引，不用 `InMemoryStore` 兜底。
"""

import asyncio
import hashlib
import sys

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings
from app.db.base import create_engine, create_session_factory
from app.observability.trace import configure_trace_store

if sys.platform == "win32":  # 兜底：非 pytest-asyncio 路径也走 Selector 策略
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(autouse=True)
def _isolate_trace_store():
    """R2：单测绝不写默认 `data/traces`。

    `trace._get_store()` 在未注入时会按 settings **惰性构造**默认目录的 TraceStore；
    单测一旦走真实的 trace 生命周期（chat/group 用例）就会把测试记录混进当日报告。
    这里显式注入 None（等价于关闭写入），需要真实 store 的用例自己在用例内注入临时目录。
    """
    configure_trace_store(None)
    yield
    configure_trace_store(None)


def pytest_asyncio_loop_factories(config, item):
    """Windows 下 psycopg 异步必须用 SelectorEventLoop（pytest-asyncio 推荐钩子）。

    该钩子一旦定义，pytest-asyncio 便要求返回非空映射，故非 Windows
    平台显式返回默认工厂（等价于不覆盖）。
    """
    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}
    return {"default": asyncio.new_event_loop}


class _FakeEmbeddings(Embeddings):
    """确定性向量（按文本哈希），用于真库测试——不加载真模型，但走真 pgvector 索引。"""

    def __init__(self, dims: int) -> None:
        self._dims = dims

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (digest * ((self._dims // len(digest)) + 1))[: self._dims]
        return [b / 255.0 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
async def real_pool():
    """真库 psycopg 连接池；测试库不可用时跳过（真库类夹具的公共前置）。"""
    settings = get_settings()
    url = settings.test_database_url
    if not url:
        pytest.skip("未配置 TEST_DATABASE_URL，跳过真库测试")

    pool = AsyncConnectionPool(
        url,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    try:
        await pool.open()
        await pool.wait(timeout=5)
    except Exception as exc:  # noqa: BLE001 - 环境不可用时跳过，避免拖垮其它单测
        await pool.close()
        pytest.skip(f"测试库不可用，跳过真库测试：{exc}")

    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def real_store(real_pool):
    """真库 AsyncPostgresStore（含 pgvector 索引，建出 store_vectors）。"""
    settings = get_settings()
    store = AsyncPostgresStore(
        real_pool,
        index={
            "dims": settings.embedding_dims,
            "embed": _FakeEmbeddings(settings.embedding_dims),
            "fields": ["text"],
        },
    )
    await store.setup()
    yield store


@pytest.fixture
async def real_checkpointer(real_pool):
    """真库 AsyncPostgresSaver（与 store 共用同一连接池，与 main.py 装配一致）。"""
    saver = AsyncPostgresSaver(real_pool)
    await saver.setup()
    yield saver


@pytest.fixture
async def real_session_factory(real_pool):
    """真库业务表会话工厂（agents / sessions / messages）。"""
    engine = create_engine(get_settings().test_sqlalchemy_url)
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()
