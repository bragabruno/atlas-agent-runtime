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
    """Load the agent spec for *agent_name*, or 404 if there is no such YAML."""
    path = Path(settings.agents_dir) / f"{agent_name}.yaml"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"unknown agent: {agent_name}")
    return load_spec(path)


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


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/agent/runs", response_model=CreateRunResponse, status_code=201)
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


@router.get("/v1/agent/runs/{run_id}", response_model=RunStatusResponse)
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
