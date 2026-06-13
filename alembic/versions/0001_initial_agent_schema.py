"""Initial agent-runtime schema: agent_runs, agent_steps (AGT-2).

Creates the two agent-runtime-owned tables with the columns, types, and
indexes specified in atlas-docs/03 §1.7, plus the two Postgres enum types
(`agent_status_enum`, `step_type_enum`). Targets Azure Database for
PostgreSQL via the psycopg3 sync driver (ADR-010).

NOTE: applying this migration to Azure PG is deferred — deliver models +
migration DDL + offline SQLite smoke only (no live PG in this iteration).

Revision ID: 0001
Revises:
Create Date: 2026-06-07

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Dialect-specific `postgresql.ENUM` with `create_type=False`: the types are
# created once explicitly in upgrade() so per-column references do not re-emit
# `CREATE TYPE`. Alembic's create_table honors `create_type=False` only on the
# dialect-specific ENUM, not generic `sa.Enum`.
agent_status_enum = postgresql.ENUM(
    "created",
    "running",
    "completed",
    "failed",
    "capped",
    name="agent_status_enum",
    create_type=False,
)
step_type_enum = postgresql.ENUM(
    "llm_call",
    "tool_call",
    name="step_type_enum",
    create_type=False,
)


def upgrade() -> None:
    # Create the shared enum types once. `.create(checkfirst=True)` is
    # idempotent online; offline (`--sql`) it emits a single `CREATE TYPE` per
    # enum. The per-column `postgresql.ENUM(..., create_type=False)` references
    # reuse the type without re-emitting it.
    bind = op.get_bind()
    agent_status_enum.create(bind, checkfirst=True)
    step_type_enum.create(bind, checkfirst=True)

    op.create_table(
        "agent_runs",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("agent_name", sa.Text(), nullable=False),
        sa.Column("agent_version", sa.Text(), nullable=False),
        sa.Column(
            "status",
            agent_status_enum,
            nullable=False,
            server_default=sa.text("'created'"),
        ),
        sa.Column("token_budget", sa.Integer(), nullable=True),
        sa.Column(
            "tokens_used",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_agent_runs_status",
        "agent_runs",
        ["status", sa.text("started_at DESC")],
    )
    op.create_index(
        "idx_agent_runs_agent_name",
        "agent_runs",
        ["agent_name", sa.text("started_at DESC")],
    )

    op.create_table(
        "agent_steps",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "agent_run_id",
            sa.Uuid(),
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("type", step_type_enum, nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.UniqueConstraint("agent_run_id", "idx", name="agent_steps_run_idx_unique"),
    )
    op.create_index(
        "idx_agent_steps_agent_run_id",
        "agent_steps",
        ["agent_run_id", "idx"],
    )


def downgrade() -> None:
    op.drop_index("idx_agent_steps_agent_run_id", table_name="agent_steps")
    op.drop_table("agent_steps")

    op.drop_index("idx_agent_runs_agent_name", table_name="agent_runs")
    op.drop_index("idx_agent_runs_status", table_name="agent_runs")
    op.drop_table("agent_runs")

    bind = op.get_bind()
    step_type_enum.drop(bind, checkfirst=True)
    agent_status_enum.drop(bind, checkfirst=True)
