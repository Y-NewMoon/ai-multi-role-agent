"""业务数据模型（方案 3.4）。"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func, text
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
