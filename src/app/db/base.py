"""SQLAlchemy 基类与引擎/会话工厂。

注意：Alembic 只管理业务表（agents / sessions / messages）；
LangGraph 的 tables 由 AsyncPostgresSaver/Store.setup() 建，不纳入 Alembic（方案 3.8）。
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """业务模型基类。"""


def create_engine(sqlalchemy_url: str, *, echo: bool = False) -> AsyncEngine:
    return create_async_engine(sqlalchemy_url, echo=echo, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
