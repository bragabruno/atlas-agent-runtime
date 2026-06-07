"""Persistence DAL — agent_runs + agent_steps CRUD (AGT-6).

All methods accept an injected SQLAlchemy `Session` so callers can:
- use the real engine in production (PostgreSQL via asyncpg or psycopg).
- use an in-memory SQLite session in tests (fully offline, no network).

Design notes:
- Session lifecycle is the caller's responsibility — the DAL never commits or
  rolls back; it only adds/updates rows.  Commit when *all* operations for a
  unit of work succeed (or rollback on error).
- Resumability: `get_last_step_idx` returns the highest persisted step index
  for a run, allowing a restarted runner to skip already-completed steps.
- Per ADR-016 nothing outside `app.persistence` imports SQLAlchemy types.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.persistence.tables import AgentRun, AgentStatusEnum, AgentStep, StepTypeEnum


def create_run(
    session: Session,
    *,
    agent_name: str,
    agent_version: str,
    token_budget: int,
) -> AgentRun:
    """Insert a new `AgentRun` row with status ``created`` and return it.

    The run ``id`` is generated client-side (uuid4) so callers have it
    immediately without a round-trip.
    """
    run = AgentRun(
        id=uuid.uuid4(),
        agent_name=agent_name,
        agent_version=agent_version,
        status=AgentStatusEnum.created,
        token_budget=token_budget,
        tokens_used=0,
        started_at=datetime.now(tz=UTC),
        ended_at=None,
    )
    session.add(run)
    return run


def update_run_status(
    session: Session,
    run: AgentRun,
    *,
    status: AgentStatusEnum,
    tokens_used: int | None = None,
    ended_at: datetime | None = None,
) -> None:
    """Mutate *run* in-place; session tracks the dirty object automatically."""
    run.status = status
    if tokens_used is not None:
        run.tokens_used = tokens_used
    if ended_at is not None:
        run.ended_at = ended_at


def append_step(
    session: Session,
    *,
    run_id: uuid.UUID,
    idx: int,
    step_type: StepTypeEnum,
    payload: dict[str, object],
    tokens: int,
    latency_ms: int,
) -> AgentStep:
    """Insert one `AgentStep` row and return it.

    `idx` must be unique within the run; the DB unique constraint
    (agent_run_id, idx) guards against duplicate inserts.
    """
    step = AgentStep(
        id=uuid.uuid4(),
        agent_run_id=run_id,
        idx=idx,
        type=step_type,
        payload=payload,
        tokens=tokens,
        latency_ms=latency_ms,
    )
    session.add(step)
    return step


def get_run(session: Session, run_id: uuid.UUID) -> AgentRun | None:
    """Fetch an `AgentRun` by primary key, or ``None`` if absent."""
    return session.get(AgentRun, run_id)


def get_last_step_idx(session: Session, run_id: uuid.UUID) -> int | None:
    """Return the highest ``idx`` persisted for *run_id*, or ``None`` if no steps exist.

    Used by a resuming runner to determine which step to restart from::

        last = get_last_step_idx(session, run_id)
        next_idx = (last + 1) if last is not None else 0
    """
    from sqlalchemy import func as sqla_func
    from sqlalchemy import select

    stmt = select(sqla_func.max(AgentStep.idx)).where(AgentStep.agent_run_id == run_id)
    result = session.execute(stmt).scalar()
    # scalar() returns None when no rows match
    return result  # type: ignore[return-value]


def list_steps(session: Session, run_id: uuid.UUID) -> list[AgentStep]:
    """Return all steps for *run_id* ordered by ``idx`` ascending."""
    from sqlalchemy import select

    stmt = select(AgentStep).where(AgentStep.agent_run_id == run_id).order_by(AgentStep.idx)
    return list(session.execute(stmt).scalars())
