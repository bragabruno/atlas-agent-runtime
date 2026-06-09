"""FastAPI application entrypoint for the atlas-agent-runtime trigger surface.

`create_app` is the application factory (title / version / OpenAPI metadata)
that mounts the v1 router; ``app = create_app()`` is the module-level instance
for ``uvicorn app.main:app``. Mirrors atlas-gateway's `app/main.py` (AGT-16,
ADR-020). The run-trigger contract is published as ``openapi.json`` via
`scripts/export_openapi.py`.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.v1.runs import router as runs_router

_TITLE = "Atlas Agent Runtime"
_VERSION = "0.1.0"
_DESCRIPTION = (
    "Trigger surface for the Atlas hand-rolled agent loop: start an agent run "
    "and poll it over HTTP (ADR-020)."
)


async def healthz() -> dict[str, str]:
    """Liveness probe — returns a static OK payload."""
    return {"status": "ok"}


def create_app() -> FastAPI:
    """Build and return the FastAPI app with the v1 agent-run router mounted."""
    app = FastAPI(title=_TITLE, version=_VERSION, description=_DESCRIPTION)
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
