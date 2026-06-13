"""Engine + Session factory for the agent-runtime persistence layer (AGT-16).

The DAL (`app.persistence.dal`) accepts an injected `Session`; this module is
the one place that constructs the engine and session factory from a DSN, so
the composition root (`app.main`) stays thin. Importing `tables` here registers
the ORM models on `Base.metadata` so `create_all` sees them.

Per ADR-016 only `app.persistence` imports SQLAlchemy construction APIs.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.persistence import tables as _tables  # noqa: F401  # pyright: ignore[reportUnusedImport]
from app.persistence.base import Base


def build_engine(url: str) -> Engine:
    """Construct a synchronous SQLAlchemy engine for *url*.

    Local compose uses ``postgresql+psycopg://atlas:atlas@postgres:5432/atlas``
    (psycopg 3, already a pinned dependency). The runner calls the DAL with a
    sync `Session`, so a sync engine is the correct match.
    """
    return create_engine(url, future=True)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a `sessionmaker` bound to *engine*.

    ``expire_on_commit=False`` so the API can read run attributes after the
    commit when building the response.
    """
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False, future=True)


def create_all(engine: Engine) -> None:
    """Create the agent_runs/agent_steps tables if absent (idempotent).

    Used at local startup so the service is self-contained; production schema is
    managed by the Alembic migrations under ``alembic/``.
    """
    Base.metadata.create_all(engine)
