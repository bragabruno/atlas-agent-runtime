"""Request/response models for the agent-runtime API (AGT-16).

`Field(..., description=...)` and `json_schema_extra={"example": ...}` are
populated so the exported OpenAPI spec (mirroring the gateway's GW-21 pattern,
see ADR-014) is documentation-grade — Python codegen consumers pick the
descriptions up as docstrings on the generated client.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CreateRunRequest(BaseModel):
    """Body for ``POST /v1/agent/runs``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent_name": "regdoc-qa",
                "user_message": "What does REACH say about DEHP authorisation?",
            }
        }
    )

    agent_name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description=(
            "Name of the agent YAML spec to run — resolved as "
            "`<ATLAS_AGENTS_DIR>/<agent_name>.yaml`. Strict kebab-case "
            "(`[a-z0-9]+(-[a-z0-9]+)*`); path separators and `..` are rejected "
            "at the schema layer. Returns 404 if no matching YAML exists."
        ),
    )
    user_message: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="The user's question/instruction passed to the agent loop.",
    )


class CreateRunResponse(BaseModel):
    """Result of a triggered run.

    The run is executed synchronously (the mock gateway completes in one
    iteration); ``run_id`` is ``None`` only when persistence is disabled
    (no ``ATLAS_DATABASE_URL``).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "9f7d2c1e-0b2a-4f1e-9c8a-1234567890ab",
                "agent_name": "regdoc-qa",
                "agent_version": "0.1.0",
                "status": "completed",
                "iterations": 1,
                "tokens_used": 142,
                "elapsed_s": 0.382,
                "content": "DEHP requires authorisation under REACH Annex XIV …",
                "responses": ["DEHP requires authorisation under REACH Annex XIV …"],
            }
        }
    )

    run_id: str | None = Field(
        description=(
            "Persistent run id (UUID) when `ATLAS_DATABASE_URL` is set; `null` "
            "when persistence is disabled (the run still executed)."
        ),
    )
    agent_name: str = Field(description="Agent name from the resolved YAML spec.")
    agent_version: str = Field(description="Agent version from the resolved YAML spec.")
    status: str = Field(
        ...,
        description="Terminal run status — `completed` or `capped` (hit a hard cap).",
    )
    iterations: int = Field(description="Number of loop iterations executed.")
    tokens_used: int = Field(description="Total tokens consumed across the run.")
    elapsed_s: float = Field(description="Wall-clock seconds for the run.")
    content: str = Field(..., description="Final assistant message content.")
    responses: list[str] = Field(
        default_factory=list,
        description="All assistant turn contents in order (length = iterations).",
    )


class StepView(BaseModel):
    """One persisted step within a run."""

    idx: int = Field(description="Zero-based step index within the run.")
    type: str = Field(description="Step type — `llm`, `tool`, `system`.")
    tokens: int = Field(description="Tokens consumed by this step.")
    latency_ms: int = Field(description="Step latency in milliseconds.")
    payload: dict[str, Any] = Field(
        description="JSON payload mirroring the persisted agent step record.",
    )


class RunStatusResponse(BaseModel):
    """Persisted run state for ``GET /v1/agent/runs/{run_id}``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "9f7d2c1e-0b2a-4f1e-9c8a-1234567890ab",
                "agent_name": "regdoc-qa",
                "agent_version": "0.1.0",
                "status": "completed",
                "token_budget": 8192,
                "tokens_used": 142,
                "started_at": "2026-06-13T10:00:00+00:00",
                "ended_at": "2026-06-13T10:00:00.382+00:00",
                "steps": [
                    {
                        "idx": 0,
                        "type": "llm",
                        "tokens": 142,
                        "latency_ms": 382,
                        "payload": {"model": "mock", "finish_reason": "stop"},
                    }
                ],
            }
        }
    )

    run_id: str = Field(description="Persistent run id (UUID).")
    agent_name: str = Field(description="Agent name from the YAML spec.")
    agent_version: str = Field(description="Agent version from the YAML spec.")
    status: str = Field(description="Run status — `running`, `completed`, `capped`, `failed`.")
    token_budget: int | None = Field(
        description="Cap from the agent spec, or `null` if no cap is defined.",
    )
    tokens_used: int = Field(description="Tokens consumed so far / at completion.")
    started_at: str = Field(description="ISO-8601 UTC timestamp when the run started.")
    ended_at: str | None = Field(
        description="ISO-8601 UTC timestamp when the run ended; `null` while running.",
    )
    steps: list[StepView] = Field(
        default_factory=list,
        description="Persisted steps in execution order.",
    )


class ErrorEnvelope(BaseModel):
    """Typed error body for non-2xx responses.

    FastAPI's default HTTPException emits `{"detail": ...}`. This envelope is
    the documented superset so codegen produces a single typed error model
    (mirrors the gateway's `ErrorEnvelope`).
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"detail": "run not found", "code": "not_found"}}
    )

    detail: str = Field(description="Human-readable error message.")
    code: str | None = Field(
        default=None,
        description="Optional machine-readable error code.",
    )
