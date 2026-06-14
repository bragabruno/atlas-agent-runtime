"""Agent-runtime HTTP routes (AGT-16, ADR-020).

- ``POST /v1/agent/runs``         — trigger an agent run, return its result.
- ``GET  /v1/agent/runs/{run_id}``— fetch a persisted run's status + steps.
- ``GET  /healthz``               — liveness probe.

Execution model
---------------
A run is executed synchronously within the POST request. The bounded agent loop
(`AgentRunner`) enforces the hard caps from the resolved `AgentSpec`, so a run
cannot block indefinitely. When persistence is configured (``ATLAS_DATABASE_URL``)
the run + every step are stored, so the GET endpoint can return status afterward;
the run id is obtained by pre-creating the row and adopting it through the
runner's ``resume_run_id`` hook (no change to the engine).

When persistence is disabled the run still executes and POST returns its result
with ``run_id=null``; GET then returns 503.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, NoReturn

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker

from app.agentspec.model import AgentSpec, load_spec
from app.api.deps import gateway_client_dep, session_factory_dep, settings_dep
from app.api.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    ErrorEnvelope,
    RunStatusResponse,
    StepView,
)
from app.config import Settings
from app.loop.errors import CapBreachError
from app.loop.gateway_client import GatewayClient
from app.loop.runner import AgentRunner
from app.tools.registry.registry import ToolRegistry

router = APIRouter()


def _resolve_spec(settings: Settings, agent_name: str) -> AgentSpec:
    """Load the agent spec for *agent_name*, or 404 if there is no such YAML.

    The path is anchored to ``settings.agents_dir`` after symlink resolution so
    that ``agent_name`` cannot escape the directory (defense in depth on top of
    the schema-layer regex that already rejects path separators and ``..``).
    The 404 detail is fixed — echoing ``agent_name`` would let an attacker probe
    the filesystem via the schema-permitted character set.
    """
    base = Path(settings.agents_dir).resolve()
    candidate = (base / f"{agent_name}.yaml").resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="agent not found")
    return load_spec(candidate)


def _raise_for_upstream(exc: httpx.HTTPStatusError) -> NoReturn:
    """Map an upstream gateway HTTP error onto an explicit response.

    A gateway 429 (rate limit / budget) propagates as 429 so callers can back
    off; anything else is a 502 — the run failed because of the upstream, not
    this service. Without this mapping the raw httpx exception surfaced as an
    opaque 500 (found under Locust load when the gateway throttled the shared
    dev key).
    """
    status = exc.response.status_code
    if status == 429:
        raise HTTPException(
            status_code=429, detail="upstream gateway rate-limited the run"
        ) from exc
    raise HTTPException(status_code=502, detail=f"upstream gateway error: HTTP {status}") from exc


@router.get(
    "/healthz",
    tags=["health"],
    summary="Liveness probe",
    description='Returns `{"status": "ok"}`. No dependencies; safe to poll.',
    response_model=dict[str, str],
)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/v1/agent/runs",
    response_model=CreateRunResponse,
    status_code=201,
    tags=["agent-runs"],
    summary="Trigger an agent run",
    description=(
        "Synchronously executes the bounded agent loop for the named agent and "
        "returns its result. The agent spec (`agents/<agent_name>.yaml`) sets the "
        "hard caps the loop enforces. When `ATLAS_DATABASE_URL` is set, the run "
        "and every step are persisted and `run_id` is the UUID for the new row; "
        "otherwise `run_id` is `null` (the run still executes)."
    ),
    responses={
        404: {
            "model": ErrorEnvelope,
            "description": "Unknown agent — no matching YAML in `ATLAS_AGENTS_DIR`.",
        },
        422: {
            "model": ErrorEnvelope,
            "description": "Cap breach (iterations / tokens / wall-time) from the loop.",
        },
        429: {
            "model": ErrorEnvelope,
            "description": "Upstream gateway rate-limited or budget-exhausted.",
        },
        502: {
            "model": ErrorEnvelope,
            "description": "Upstream gateway returned a non-429 HTTP error.",
        },
    },
)
async def create_run(
    body: CreateRunRequest,
    settings: Annotated[Settings, Depends(settings_dep)],
    gateway: Annotated[GatewayClient, Depends(gateway_client_dep)],
    session_factory: Annotated[sessionmaker[Session] | None, Depends(session_factory_dep)],
) -> CreateRunResponse:
    spec = _resolve_spec(settings, body.agent_name)
    registry = ToolRegistry(spec)

    # --- No persistence configured: run and return, run_id is null. ---
    if session_factory is None:
        runner = AgentRunner(spec=spec, client=gateway, registry=registry)
        try:
            result = await runner.run(user_message=body.user_message)
        except CapBreachError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            _raise_for_upstream(exc)
        content = result.responses[-1].content if result.responses else ""
        return CreateRunResponse(
            run_id=None,
            agent_name=spec.agent_name,
            agent_version=spec.agent_version,
            status="completed",
            iterations=result.iterations,
            tokens_used=result.tokens_used,
            elapsed_s=result.elapsed_s,
            content=content,
            responses=[r.content for r in result.responses],
        )

    # --- Persistence configured: pre-create the run row so we own its id, then
    #     let the runner adopt it via resume_run_id (0 steps → starts fresh). ---
    from app.persistence.dal import create_run as dal_create_run
    from app.persistence.dal import get_run

    session = session_factory()
    try:
        run = dal_create_run(
            session,
            agent_name=spec.agent_name,
            agent_version=spec.agent_version,
            token_budget=spec.token_budget,
        )
        session.flush()  # make the row persistent so the runner's get_run finds it
        run_id = run.id

        runner = AgentRunner(
            spec=spec,
            client=gateway,
            registry=registry,
            session=session,
            resume_run_id=run_id,
        )
        try:
            result = await runner.run(user_message=body.user_message)
            session.commit()
        except httpx.HTTPStatusError as exc:
            session.rollback()  # discard the pre-created run row
            _raise_for_upstream(exc)
        except CapBreachError:
            # The runner persisted status=capped before re-raising.
            session.commit()
            capped = get_run(session, run_id)
            return CreateRunResponse(
                run_id=str(run_id),
                agent_name=spec.agent_name,
                agent_version=spec.agent_version,
                status="capped",
                iterations=0,
                tokens_used=capped.tokens_used if capped else 0,
                elapsed_s=0.0,
                content="",
                responses=[],
            )

        content = result.responses[-1].content if result.responses else ""
        return CreateRunResponse(
            run_id=str(run_id),
            agent_name=spec.agent_name,
            agent_version=spec.agent_version,
            status="completed",
            iterations=result.iterations,
            tokens_used=result.tokens_used,
            elapsed_s=result.elapsed_s,
            content=content,
            responses=[r.content for r in result.responses],
        )
    finally:
        session.close()


@router.get(
    "/v1/agent/runs/{run_id}",
    response_model=RunStatusResponse,
    tags=["agent-runs"],
    summary="Get a persisted run's status",
    description=(
        "Returns the persisted run + its ordered steps. Only available when "
        "`ATLAS_DATABASE_URL` is configured — otherwise this endpoint returns 503."
    ),
    responses={
        400: {"model": ErrorEnvelope, "description": "`run_id` is not a valid UUID."},
        404: {"model": ErrorEnvelope, "description": "No run with this id."},
        503: {
            "model": ErrorEnvelope,
            "description": "Run persistence is not configured (`ATLAS_DATABASE_URL` unset).",
        },
    },
)
async def get_run_status(
    run_id: str,
    session_factory: Annotated[sessionmaker[Session] | None, Depends(session_factory_dep)],
) -> RunStatusResponse:
    if session_factory is None:
        raise HTTPException(
            status_code=503,
            detail="run persistence is not configured (no ATLAS_DATABASE_URL)",
        )
    try:
        rid = uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid run id") from exc

    from app.persistence.dal import get_run, list_steps

    session = session_factory()
    try:
        run = get_run(session, rid)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        steps = list_steps(session, rid)
        return RunStatusResponse(
            run_id=str(run.id),
            agent_name=run.agent_name,
            agent_version=run.agent_version,
            status=run.status.value,
            token_budget=run.token_budget,
            tokens_used=run.tokens_used,
            started_at=run.started_at.isoformat(),
            ended_at=run.ended_at.isoformat() if run.ended_at else None,
            steps=[
                StepView(
                    idx=s.idx,
                    type=s.type.value,
                    tokens=s.tokens,
                    latency_ms=s.latency_ms,
                    payload=s.payload,
                )
                for s in steps
            ],
        )
    finally:
        session.close()
