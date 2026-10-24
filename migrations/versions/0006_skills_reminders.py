"""skills & reminders: agents.allowed_skills / reminders

Revision ID: 0006_skills_reminders
Revises: 0005_lore_tables
Create Date: 2026-10-20

V2.2 Skill 工具层（契约 §2.8）：给 agents 加allowed_skills 白名单列（TEXT[]，默认 '{}'），
并新增 reminders 表（pending/active/cancelled/fired）与 2 个索引。
纯加性变更：不动任何既有表/列/索引；旧 agent 的 allowed_skills 默认 '{}'，行为零变化。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006_skills_reminders"
down_revision: str | None = "0005_lore_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agents",
        sa.Column(
            "allowed_skills",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
    )

    op.create_table(
        "reminders",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "agent_id",
            sa.Integer(),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("remind_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('pending','active','cancelled','fired')",
            name="ck_reminders_status",
        ),
        sa.UniqueConstraint("idempotency_key", name="uq_reminders_idempotency_key"),
    )
    op.create_index(
        "ix_reminders_agent_status", "reminders", ["agent_id", "status"]
    )
    op.create_index(
        "ix_reminders_due",
        "reminders",
        ["remind_at"],
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index("ix_reminders_due", table_name="reminders")
    op.drop_index("ix_reminders_agent_status", table_name="reminders")
    op.drop_table("reminders")
    op.drop_column("agents", "allowed_skills")
