"""FastAPI application entrypoint for the Atlas agent runtime (AGT-16, ADR-020).

Composition root: builds the gateway client and (optionally) the DB session
factory once at startup and stores them on ``app.state`` for the routes to read
via ``app.api.deps``. Persistence is config-gated on ``ATLAS_DATABASE_URL`` —
with no DSN the service still serves runs (POST), it just doesn't persist them.

Served by uvicorn on :8000 — ``uvicorn app.main:app --host 0.0.0.0 --port 8000``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.gateway_client import HttpGatewayClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.gateway_client = HttpGatewayClient(
        base_url=settings.gateway_url,
        api_key=settings.gateway_api_key,
    )

    if settings.database_url:
        from app.persistence.session import build_engine, build_session_factory, create_all

        engine = build_engine(settings.database_url)
        create_all(engine)
        app.state.engine = engine
        app.state.session_factory = build_session_factory(engine)
    else:
        app.state.engine = None
        app.state.session_factory = None

    yield

    if app.state.engine is not None:
        app.state.engine.dispose()


app = FastAPI(title="Atlas Agent Runtime", version="0.1.0", lifespan=lifespan)
app.include_router(router)
