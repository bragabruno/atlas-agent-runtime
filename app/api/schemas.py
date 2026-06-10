"""Request/response models for the agent-runtime API (AGT-16)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    """Body for ``POST /v1/agent/runs``."""

    agent_name: str = Field(..., min_length=1, description="Name of the agent YAML spec to run")
    user_message: str = Field(..., min_length=1, description="The user's question/instruction")


class CreateRunResponse(BaseModel):
    """Result of a triggered run.

    The run is executed synchronously (the mock gateway completes in one
    iteration); ``run_id`` is ``None`` only when persistence is disabled
    (no ``ATLAS_DATABASE_URL``).
    """

    run_id: str | None
    agent_name: str
    agent_version: str
    status: str = Field(..., description="completed | capped")
    iterations: int
    tokens_used: int
    elapsed_s: float
    content: str = Field(..., description="Final assistant message content")
    responses: list[str] = Field(default_factory=list, description="All assistant turn contents")


class StepView(BaseModel):
    """One persisted step within a run."""

    idx: int
    type: str
    tokens: int
    latency_ms: int
    payload: dict  # type: ignore[type-arg]


class RunStatusResponse(BaseModel):
    """Persisted run state for ``GET /v1/agent/runs/{run_id}``."""

    run_id: str
    agent_name: str
    agent_version: str
    status: str
    token_budget: int | None
    tokens_used: int
    started_at: str
    ended_at: str | None
    steps: list[StepView] = Field(default_factory=list)
