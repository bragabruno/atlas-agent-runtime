"""AGT-16 — agent-run trigger endpoints (thin HTTP layer, ADR-020).

Two routes implement the documented contract:

- ``POST /v1/agent/runs`` → start a run; returns **202** with ``{run_id,
  status}``. The body is ``{agent, user_message}``.
- ``GET /v1/agent/runs/{run_id}`` → poll a run; returns ``{run_id, status,
  result|null, error|null}``. An unknown id → **404**.

All orchestration lives in `RunService` (injected via `app.api.deps`); this
controller only maps HTTP ↔ the service and translates domain errors to HTTP:

- `AgentNotFoundError` → 422 (the requested agent has no spec).
- `RunNotFoundError`   → 404 (no run for the polled id).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_run_service
from app.api.schemas import AgentRunResponse, StartAgentRunRequest
from app.services.run_service import AgentNotFoundError, RunNotFoundError, RunService

router = APIRouter(prefix="/v1/agent", tags=["agent-runs"])


@router.post(
    "/runs",
    response_model=AgentRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an agent run",
)
def start_agent_run(
    req: StartAgentRunRequest,
    service: Annotated[RunService, Depends(get_run_service)],
) -> AgentRunResponse:
    """Start a run for ``req.agent`` and return its ``run_id`` (202)."""
    try:
        return service.start_run(agent=req.agent, user_message=req.user_message)
    except AgentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get(
    "/runs/{run_id}",
    response_model=AgentRunResponse,
    summary="Poll an agent run",
)
def get_agent_run(
    run_id: str,
    service: Annotated[RunService, Depends(get_run_service)],
) -> AgentRunResponse:
    """Poll a run by id; unknown id → 404."""
    try:
        return service.get_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
