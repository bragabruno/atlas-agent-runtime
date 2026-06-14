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
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.gateway_client import HttpGatewayClient


def _resolve_app_version() -> str:
    # `pip install -e .` not run yet (e.g. `python -m app.main` from a clean
    # checkout); fall through to the pyproject pin so the spec still renders.
    try:
        return version("atlas-agent-runtime")
    except PackageNotFoundError:
        return "0.1.0"


_APP_VERSION = _resolve_app_version()

_API_DESCRIPTION = """\
The **Atlas Agent Runtime** is the FastAPI trigger surface (ADR-020) over the
hand-rolled bounded agent loop (ADR-006). Runs are synchronous — the loop
enforces hard iteration / token / wall-time caps from the agent's YAML spec
so a request cannot block indefinitely.

### Persistence (config-gated)
With `ATLAS_DATABASE_URL` set, runs + steps are persisted and
`GET /v1/agent/runs/{run_id}` returns their status. Without it, runs still
execute (`POST` works) but `run_id` is `null` and `GET` returns 503.

### Upstream errors
The runtime calls the gateway over HTTP; gateway 429 propagates as 429,
anything else upstream becomes 502.

### Errors
4xx/5xx responses use the `ErrorEnvelope` schema (`{detail, code?}`).
"""

_OPENAPI_TAGS = [
    {
        "name": "agent-runs",
        "description": (
            "Trigger and inspect agent runs. The bounded loop honors the caps "
            "defined in the agent YAML spec (`agents/<agent_name>.yaml`)."
        ),
    },
    {
        "name": "health",
        "description": "Liveness probe. Used by Kubernetes and the local-compose stack.",
    },
]


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


def create_app(settings: Settings) -> FastAPI:
    """Construct the FastAPI app for ``settings``.

    Extracted from module scope so the docs gate (``environment != "prod"``)
    is unit-testable without monkey-patching. Module-level ``app`` below
    instantiates this with the runtime ``Settings()`` — production wiring
    is unchanged.
    """
    docs_enabled = settings.environment != "prod"
    return FastAPI(
        title="Atlas Agent Runtime",
        version=_APP_VERSION,
        description=_API_DESCRIPTION,
        contact={"name": "Bruno Braga", "email": "contact@bragdev.com"},
        license_info={"name": "Internal — Atlas (Enhesa mirror)"},
        servers=[
            {"url": "http://localhost:8083", "description": "Local debug (VS Code launcher)"},
            {
                "url": "http://atlas-agent-runtime.atlas-platform.svc:8000",
                "description": "In-cluster (AKS)",
            },
        ],
        openapi_tags=_OPENAPI_TAGS,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        swagger_ui_parameters=(
            {
                "displayRequestDuration": True,
                "tryItOutEnabled": True,
                "docExpansion": "list",
            }
            if docs_enabled
            else None
        ),
        lifespan=lifespan,
    )


app = create_app(get_settings())
app.include_router(router)
