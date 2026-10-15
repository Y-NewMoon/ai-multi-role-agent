"""add thinking column to messages / group_messages

Revision ID: 0004_message_thinking
Revises: 0003_idle_messages
Create Date: 2026-10-14

角色思考链：messages / group_messages 各新增 thinking 列（可空），
存该条角色发言对应的「内心独白」，供前端按需（点击按钮）展示。
旧数据为 NULL → 前端不显示思考按钮。
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_message_thinking"
down_revision: str | None = "0003_idle_messages"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("thinking", sa.Text(), nullable=True))
    op.add_column("group_messages", sa.Column("thinking", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("group_messages", "thinking")
    op.drop_column("messages", "thinking")
