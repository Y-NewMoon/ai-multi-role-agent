"""add is_idle flag to messages / group_messages

Revision ID: 0003_idle_messages
Revises: 0002_group_chat
Create Date: 2026-10-14

自主闲聊标记：messages / group_messages 新增 is_idle 列，
用于区分用户触发的正常回复与角色自主闲聊（便于「清空闲聊」）。
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_idle_messages"
down_revision: str | None = "0002_group_chat"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "is_idle",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "group_messages",
        sa.Column(
            "is_idle",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("group_messages", "is_idle")
    op.drop_column("messages", "is_idle")
