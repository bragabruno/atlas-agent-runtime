"""AGT-2 schema smoke tests.

Verifies:
- Models import cleanly and register on Base.metadata.
- SQLite create_all succeeds (offline schema smoke; no Azure PG required).
- Both tables exist with correct column sets.
- FK from agent_steps.agent_run_id → agent_runs.id is present.
- Expected indexes exist on both tables.

Note: applying this migration to Azure PG is deferred — these tests cover the
SQLAlchemy layer only (atlas-docs/03 §1.7, ADR-010).
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import create_engine, inspect

import app.persistence.tables as tables  # registers models on Base.metadata
from app.persistence.base import Base
from app.persistence.tables import AgentRun, AgentStep

_ = tables  # referenced to satisfy strict unused-import check


def _engine() -> sa.Engine:
    """In-memory SQLite engine for offline schema smoke."""
    return create_engine("sqlite:///:memory:")


def test_models_registered_on_metadata() -> None:
    """Base.metadata must contain both tables after import."""
    assert "agent_runs" in Base.metadata.tables
    assert "agent_steps" in Base.metadata.tables


def test_create_all_succeeds() -> None:
    """create_all must complete without error on SQLite."""
    engine = _engine()
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    assert "agent_runs" in table_names
    assert "agent_steps" in table_names


def test_agent_runs_columns() -> None:
    """agent_runs must have all columns from atlas-docs/03 §1.7."""
    engine = _engine()
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    col_names = {c["name"] for c in inspector.get_columns("agent_runs")}
    expected = {
        "id",
        "agent_name",
        "agent_version",
        "status",
        "token_budget",
        "tokens_used",
        "started_at",
        "ended_at",
    }
    assert expected <= col_names


def test_agent_steps_columns() -> None:
    """agent_steps must have all columns from atlas-docs/03 §1.7."""
    engine = _engine()
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    col_names = {c["name"] for c in inspector.get_columns("agent_steps")}
    expected = {
        "id",
        "agent_run_id",
        "idx",
        "type",
        "payload",
        "tokens",
        "latency_ms",
    }
    assert expected <= col_names


def test_agent_steps_fk_to_agent_runs() -> None:
    """agent_steps.agent_run_id must FK to agent_runs.id."""
    engine = _engine()
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    fks = inspector.get_foreign_keys("agent_steps")
    assert any(
        fk["referred_table"] == "agent_runs" and "agent_run_id" in fk["constrained_columns"]
        for fk in fks
    ), f"Expected FK agent_steps.agent_run_id → agent_runs.id, got: {fks}"


def test_agent_runs_indexes_present() -> None:
    """agent_runs must have indexes on (status) and (agent_name)."""
    engine = _engine()
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    idx_cols: set[str] = set()
    for idx in inspector.get_indexes("agent_runs"):
        idx_cols.update(c for c in idx["column_names"] if c is not None)
    assert "status" in idx_cols, f"Missing status index, got index cols: {idx_cols}"
    assert "agent_name" in idx_cols, f"Missing agent_name index, got index cols: {idx_cols}"


def test_agent_steps_index_present() -> None:
    """agent_steps must have an index on (agent_run_id)."""
    engine = _engine()
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    idx_cols: set[str] = set()
    for idx in inspector.get_indexes("agent_steps"):
        idx_cols.update(c for c in idx["column_names"] if c is not None)
    assert "agent_run_id" in idx_cols, f"Missing agent_run_id index, got: {idx_cols}"


def test_orm_model_classes_importable() -> None:
    """AgentRun and AgentStep ORM classes must be importable and have correct tablenames."""
    assert AgentRun.__tablename__ == "agent_runs"
    assert AgentStep.__tablename__ == "agent_steps"
