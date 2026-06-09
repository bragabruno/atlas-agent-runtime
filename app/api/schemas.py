"""Request/response Pydantic models for the agent-run trigger surface (AGT-16).

These are the *wire* types — the HTTP contract documented in the README and
published in `openapi.json`. They are deliberately separate from the domain
(`AgentSpec`, `RunResult`) and the persistence (`AgentRun`) types: the API layer
owns its own shapes so the contract can stay stable while internals change
(ADR-016 layering, ADR-020).

Status vocabulary (`RunStatus`) is the API-facing projection of the persisted
`AgentStatusEnum`: the five DB lifecycle states collapse to the three a caller
polls for — ``running`` while in flight, ``succeeded`` on natural completion,
``failed`` on a cap breach or any other error. The mapping lives in
`app.services.run_service`.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class RunStatus(str, enum.Enum):
    """API-facing run status returned to polling clients.

    Projection of `app.persistence.tables.AgentStatusEnum`:
    ``created``/``running`` → ``running``; ``completed`` → ``succeeded``;
    ``failed``/``capped`` → ``failed``.
    """

    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class StartAgentRunRequest(BaseModel):
    """Body of ``POST /v1/agent/runs`` — start a run for a named agent."""

    model_config = {"extra": "forbid"}

    agent: str = Field(..., min_length=1, description="Agent name/id to run (resolves a spec)")
    user_message: str = Field(..., min_length=1, description="The user's input for the run")


class AgentRunResponse(BaseModel):
    """Response for both the start (202) and the poll (200) endpoints.

    On start, ``result``/``error`` are ``null`` and ``status`` is ``running``.
    On poll, a terminal run carries either ``result`` (succeeded) or ``error``
    (failed); both stay ``null`` while still ``running``.
    """

    run_id: str = Field(..., description="Server-generated run identifier (UUID)")
    status: RunStatus = Field(..., description="Current lifecycle status")
    result: str | None = Field(default=None, description="Final agent answer when succeeded")
    error: str | None = Field(default=None, description="Explicit failure reason when failed")
