"""FastAPI application entrypoint for the Atlas agent runtime (AGT-16, ADR-020).

`create_app` is the application factory: it builds the FastAPI app with
documentation-grade OpenAPI metadata, gates `/docs` + `/openapi.json` off in
prod (``environment != "prod"``), and mounts the v1 agent-run router plus the
`/healthz` liveness probe. ``app = create_app()`` is the module-level instance
for ``uvicorn app.main:app``.

The trigger surface is thin: controllers (`app.api.v1`) map HTTP ↔ `RunService`
(`app.services`), which drives the hand-rolled bounded loop (`AgentRunner`).
Runs execute in the background and are polled over HTTP — the POST returns 202
immediately. Inbound auth is bearer-key (BRA-878); `/healthz` stays open. The
run-trigger contract is published as ``openapi.json`` via
`scripts/export_openapi.py`.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.v1.runs import router as runs_router
from app.config import Settings, get_settings

_TITLE = "Atlas Agent Runtime"
_VERSION = "0.1.0"

_API_DESCRIPTION = """\
The **Atlas Agent Runtime** is the FastAPI trigger surface (ADR-020) over the
hand-rolled bounded agent loop (ADR-006). A run is started with
`POST /v1/agent/runs` (returns `202` immediately) and polled with
`GET /v1/agent/runs/{run_id}`; the loop enforces hard iteration / token /
wall-time caps from the agent's YAML spec so a run cannot block indefinitely.

### Auth
Both trigger routes require `Authorization: Bearer <key>` (BRA-878); `/healthz`
stays open. The local/compose default key is `dev-key`.

### Status vocabulary
`status` is the API projection of the persisted run state: `running` while in
flight, `succeeded` on natural completion, `failed` on a cap breach or any
other error.
"""

_OPENAPI_TAGS = [
    {
        "name": "agent-runs",
        "description": (
            "Trigger and poll agent runs. The bounded loop honors the caps "
            "defined in the agent YAML spec (`<agents_dir>/<agent>.yaml`)."
        ),
    },
    {
        "name": "health",
        "description": "Liveness probe. Used by Kubernetes and the local-compose stack.",
    },
]


async def healthz() -> dict[str, str]:
    """Liveness probe — returns a static OK payload. No auth, no dependencies."""
    return {"status": "ok"}


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct the FastAPI app for ``settings`` (defaults to runtime settings).

    The ``settings`` arg is optional so the offline default path (`create_app()`)
    works with zero wiring, while the docs gate (``environment != "prod"``) stays
    unit-testable by passing an explicit ``Settings(environment=...)``.
    """
    settings = settings if settings is not None else get_settings()
    docs_enabled = settings.environment != "prod"
    app = FastAPI(
        title=_TITLE,
        version=_VERSION,
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
    )
    app.include_router(runs_router)
    app.add_api_route(
        "/healthz",
        healthz,
        methods=["GET"],
        tags=["health"],
        summary="Liveness probe",
    )
    return app


app = create_app()
