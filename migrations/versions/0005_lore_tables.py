"""lore tables: lore_docs / lore_chunks

Revision ID: 0005_lore_tables
Revises: 0004_message_thinking
Create Date: 2026-10-18

V2.1 世界书知识库：新增 2 表 + 4 索引（ADR-0009），不动任何既有表。
vector 扩展由 LangGraph store.setup() 建表时已创建，此处 IF NOT EXISTS 兜底，
保证独立库（如新建测试库）也能直接 upgrade head。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005_lore_tables"
down_revision: str | None = "0004_message_thinking"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "lore_docs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "agent_id",
            sa.Integer(),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column(
            "always_on", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'active'")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("agent_id", "slug", name="uq_lore_docs_agent_slug"),
    )

    op.create_table(
        "lore_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "doc_id",
            sa.Integer(),
            sa.ForeignKey("lore_docs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("agent_id", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=False),
        sa.Column("header_path", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column(
            "keywords",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'active'")
        ),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("tsv", postgresql.TSVECTOR(), nullable=True),
        sa.UniqueConstraint(
            "agent_id", "source_ref", name="uq_lore_chunks_agent_source_ref"
        ),
    )

    # 索引逐字对齐契约 4.1
    op.execute("CREATE INDEX ix_lore_chunks_agent_status ON lore_chunks (agent_id, status)")
    op.execute("CREATE INDEX ix_lore_kw ON lore_chunks USING gin (keywords)")
    op.execute("CREATE INDEX ix_lore_tsv ON lore_chunks USING gin (tsv)")
    op.execute(
        "CREATE INDEX ix_lore_hnsw ON lore_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    # 先删索引再删表（契约 4.1）
    op.execute("DROP INDEX IF EXISTS ix_lore_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_lore_tsv")
    op.execute("DROP INDEX IF EXISTS ix_lore_kw")
    op.execute("DROP INDEX IF EXISTS ix_lore_chunks_agent_status")
    op.drop_table("lore_chunks")
    op.drop_table("lore_docs")
