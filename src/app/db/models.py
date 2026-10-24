"""业务数据模型（方案 3.4）。"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Agent(Base):
    """角色档案（档案记忆）。"""

    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    avatar: Mapped[str | None] = mapped_column(String(255))
    persona_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    background: Mapped[str | None] = mapped_column(Text)
    # V2.2：本角色可用的工具白名单（Skill 名）；空数组 = 不绑定任何工具（D5 渐进披露）
    allowed_skills: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]"), default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Session(Base):
    """会话。"""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # V1 无鉴权，预留字段，不参与逻辑分支（方案 3.4）
    user_id: Mapped[str | None] = mapped_column(String(64))
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    agent: Mapped[Agent] = relationship("Agent")


class Message(Base):
    """展示用消息快照（真相源是 checkpointer，见方案 3.7）。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_idle: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"), default=False
    )
    # 角色该条发言的「内心独白」（思考链）；用户消息/无思考时为空。仅供前端按需展示。
    thinking: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ===== 群聊（M4 · ADR-0004 §3.5：纯新增 3 表，不动 sessions）=====


class GroupSession(Base):
    """群会话（V1：一个群 = 一个群会话）。"""

    __tablename__ = "group_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class GroupMember(Base):
    """群成员：仅存角色，用户不入表、不占额度（ADR-0004 §3.5）。"""

    __tablename__ = "group_members"

    group_session_id: Mapped[int] = mapped_column(
        ForeignKey("group_sessions.id"), primary_key=True
    )
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True
    )
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class GroupMessage(Base):
    """群聊展示用消息快照（agent_id 为空表示用户发言）。"""

    __tablename__ = "group_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_session_id: Mapped[int] = mapped_column(
        ForeignKey("group_sessions.id"), nullable=False
    )
    agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_idle: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"), default=False
    )
    # 该发言角色对应的「内心独白」（思考链）；用户发言/无思考时为空。
    thinking: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ===== 世界书知识库（V2.1 · ADR-0009：纯新增 2 表，不动既有模型）=====


class LoreDoc(Base):
    """角色设定文档（世界书）：一篇 Markdown 设定，同 slug 即更新。"""

    __tablename__ = "lore_docs"
    __table_args__ = (
        UniqueConstraint("agent_id", "slug", name="uq_lore_docs_agent_slug"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    # 文档稳定标识，如 tea_house（API 校验 ^[a-z0-9_]{1,40}$）
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    # 每次内容变化 +1
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1"), default=1
    )
    # 整篇 sha256
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    # 核心设定：常驻注入
    always_on: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"), default=False
    )
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'active'"), default="active"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class LoreChunk(Base):
    """设定切分后的检索单元（vector / tsv / 关键词三路可召回）。"""

    __tablename__ = "lore_chunks"
    __table_args__ = (
        UniqueConstraint(
            "agent_id", "source_ref", name="uq_lore_chunks_agent_source_ref"
        ),
        Index("ix_lore_chunks_agent_status", "agent_id", "status"),
        Index("ix_lore_kw", "keywords", postgresql_using="gin"),
        Index("ix_lore_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_lore_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[int] = mapped_column(
        ForeignKey("lore_docs.id", ondelete="CASCADE"), nullable=False
    )
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    # 1 起，随切分顺序
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    # slug:NN，稳定外部 id（评测引用此值，不引用自增 id）
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    # "茶馆 / 历史 / 祖父"
    header_path: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]"), default=list
    )
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'active'"), default="active"
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))
    tsv: Mapped[str | None] = mapped_column(TSVECTOR)


# ===== 提醒（V2.2 · HITL：工具只落 pending，用户 API 确认后才 active）=====


class Reminder(Base):
    """提醒事项：pending/active/cancelled/fired（V2.2 不做定时投递，fired 留给 V2.4）。"""

    __tablename__ = "reminders"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','active','cancelled','fired')",
            name="ck_reminders_status",
        ),
        UniqueConstraint("idempotency_key", name="uq_reminders_idempotency_key"),
        Index("ix_reminders_agent_status", "agent_id", "status"),
        Index(
            "ix_reminders_due",
            "remind_at",
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    remind_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'pending'"), default="pending"
    )
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
