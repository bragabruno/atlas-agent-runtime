"""FastAPI dependency providers — DI wiring for the agent-run trigger surface.

Controllers declare what they need via ``Depends(...)`` instead of constructing
collaborators, so the gateway client, persistence session factory, agent-spec
loader, and background scheduler are wired in one place — the composition root
for the request scope (ADR-016, mirroring atlas-gateway's `app/api/deps.py`).

Offline-by-default: with no environment overrides, `get_run_service` returns a
`RunService` wired with `MockGatewayClient` (no network / keys) and an in-memory
SQLite engine (no Postgres). The engine uses a `StaticPool` with
``check_same_thread=False`` so the row created by ``POST`` is visible to the
background run and to a later ``GET`` — all of which use independent sessions,
possibly on different threads (FastAPI's `BackgroundTasks` / `TestClient`
threadpool). This is the exact "offline test via Mock" path AGT-16 requires.

Overrides (env vars only — never secrets):
- ``ATLAS_AGENTS_DIR``  : directory of ``<agent>.yaml`` specs (default:
  the repo's ``tests/fixtures``, which ships ``regdoc-qa.yaml``).
- ``ATLAS_DATABASE_URL``: SQLAlchemy URL for a real store (default: shared
  in-memory SQLite). When set, the schema is assumed to be managed by Alembic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.loop.gateway_client import GatewayClient, MockGatewayClient
from app.persistence.base import Base
from app.services.run_service import RunService, SpecLoader, make_dir_spec_loader

#: Repo root (…/app/api/deps.py → repo root), used to locate the default agents
#: directory so the offline default works from any CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Default agents directory — the shipped fixtures hold ``regdoc-qa.yaml``.
_DEFAULT_AGENTS_DIR = _REPO_ROOT / "tests" / "fixtures"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine.

    Default (no ``ATLAS_DATABASE_URL``): a single shared in-memory SQLite engine
    whose schema is created once from `Base.metadata`. ``StaticPool`` +
    ``check_same_thread=False`` keep one connection shared across threads so the
    run row persists across the POST → background → GET request boundary. When a
    real URL is configured, the schema is owned by Alembic and not created here.
    """
    url = os.environ.get("ATLAS_DATABASE_URL")
    if url is None:
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        return engine
    return create_engine(url)


def get_gateway_client() -> GatewayClient:
    """Return the offline-default gateway client (the "via Mock" requirement)."""
    return MockGatewayClient()


@lru_cache(maxsize=1)
def get_spec_loader() -> SpecLoader:
    """Return the agent-spec loader resolving ``<agents_dir>/<agent>.yaml``."""
    agents_dir = Path(os.environ.get("ATLAS_AGENTS_DIR", str(_DEFAULT_AGENTS_DIR)))
    return make_dir_spec_loader(agents_dir)


def _session_factory() -> Session:
    """Build a fresh `Session` bound to the process-wide engine."""
    return Session(get_engine())


def get_run_service(
    background_tasks: BackgroundTasks,
    client: Annotated[GatewayClient, Depends(get_gateway_client)],
    spec_loader: Annotated[SpecLoader, Depends(get_spec_loader)],
) -> RunService:
    """Build a `RunService` for the current request.

    The background scheduler is bound to *this* request's ``BackgroundTasks`` so
    the run executes after the 202 response is sent; the session factory and
    Mock client come from the offline-default providers above.
    """
    return RunService(
        client=client,
        session_factory=_session_factory,
        spec_loader=spec_loader,
        schedule=background_tasks.add_task,
    )
