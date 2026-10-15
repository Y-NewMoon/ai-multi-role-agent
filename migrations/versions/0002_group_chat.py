"""add group chat tables: group_sessions / group_members / group_messages

Revision ID: 0002_group_chat
Revises: 0001_init
Create Date: 2026-10-12

纯新增（ADR-0004 §3.5）：不改动 agents / sessions / messages。
只创建业务表；LangGraph 的 checkpoints/store 表由 setup() 管理（方案 3.8）。
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_group_chat"
down_revision: str | None = "0001_init"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "group_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "group_members",
        sa.Column(
            "group_session_id",
            sa.Integer(),
            sa.ForeignKey("group_sessions.id"),
            primary_key=True,
        ),
        sa.Column(
            "agent_id",
            sa.Integer(),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "joined_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "group_messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "group_session_id",
            sa.Integer(),
            sa.ForeignKey("group_sessions.id"),
            nullable=False,
        ),
        sa.Column(
            "agent_id",
            sa.Integer(),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("group_messages")
    op.drop_table("group_members")
    op.drop_table("group_sessions")
