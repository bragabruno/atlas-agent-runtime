"""SQLAlchemy 2.0 typed models for agent_runs and agent_steps (AGT-2).

Mirrors the DDL in atlas-docs/03 §1.7. Models use SQLAlchemy 2.0 typed
`Mapped[...]` columns and form the schema source of truth Alembic diffs
against (ADR-010). Column types compile to the atlas-docs Postgres DDL in
production (JSONB, UUID, TIMESTAMPTZ) while remaining valid under SQLite for
the offline schema smoke test (JSON renders as TEXT, UUID as CHAR(32),
DateTime as DATETIME). Per ADR-016 nothing outside `app.persistence`
imports these. See atlas-docs/03 §1.7.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.base import Base


class AgentStatusEnum(enum.Enum):
    """Agent run lifecycle state — the `agent_status_enum` Postgres type."""

    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"
    capped = "capped"


class StepTypeEnum(enum.Enum):
    """Step type within an agent run — the `step_type_enum` Postgres type."""

    llm_call = "llm_call"
    tool_call = "tool_call"


class AgentRun(Base):
    """One execution of a named agent version (atlas-docs/03 §1.7).

    Persists resource caps, accumulated token usage, and lifecycle status.
    Runs that breach a hard cap are stored with status = 'capped'.
    """

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    agent_name: Mapped[str] = mapped_column(Text, nullable=False)
    agent_version: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[AgentStatusEnum] = mapped_column(
        Enum(AgentStatusEnum, name="agent_status_enum"),
        nullable=False,
        default=AgentStatusEnum.created,
        server_default=text("'created'"),
    )
    token_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_agent_runs_status", "status", text("started_at DESC")),
        Index("idx_agent_runs_agent_name", "agent_name", text("started_at DESC")),
    )


class AgentStep(Base):
    """One LLM call or tool call within an agent run (atlas-docs/03 §1.7).

    `idx` is the zero-based step index within the run; the unique constraint
    on (agent_run_id, idx) ensures no duplicate step positions.
    """

    __tablename__ = "agent_steps"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[StepTypeEnum] = mapped_column(
        Enum(StepTypeEnum, name="step_type_enum"),
        nullable=False,
    )
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("agent_run_id", "idx", name="agent_steps_run_idx_unique"),
        Index("idx_agent_steps_agent_run_id", "agent_run_id", "idx"),
    )
