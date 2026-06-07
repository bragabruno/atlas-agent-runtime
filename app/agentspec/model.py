"""AgentSpec — Pydantic v2 model for agent YAML declarations (AGT-1).

Validates every field on construction; invalid specs raise `ValidationError`
with explicit messages — never silently coerced. Call `load_spec` to parse a
YAML file from disk.

Schema mirrors the table in atlas-agent-runtime/README.md § Agent Definition Schema.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class AgentSpec(BaseModel):
    """Validated representation of one agent YAML file.

    All fields are required — there are no optional caps (absence would be a
    silent failure mode, and silent failures are disallowed per ADR-006).
    """

    model_config = {"frozen": True, "extra": "forbid"}

    agent_name: str = Field(..., min_length=1, description="Unique name for the agent")
    agent_version: str = Field(..., min_length=1, description="Semver version string")
    system_prompt_ref: str = Field(..., min_length=1, description="Relative path to system prompt")
    model_alias: str = Field(
        ..., min_length=1, description="Logical model name resolved by the Gateway"
    )
    tool_whitelist: list[str] = Field(
        ..., description="Exact tool names the agent is allowed to call"
    )
    max_iterations: int = Field(..., gt=0, description="Maximum LLM→tool cycles")
    token_budget: int = Field(..., gt=0, description="Cumulative token cap across all LLM calls")
    timeout_s: int = Field(..., gt=0, description="Wall-time limit in seconds for the full run")

    @field_validator("agent_version")
    @classmethod
    def _validate_semver(cls, v: str) -> str:
        """Reject versions that don't look like semver (MAJOR.MINOR.PATCH)."""
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"agent_version must be MAJOR.MINOR.PATCH semver, got: {v!r}")
        return v

    @field_validator("tool_whitelist")
    @classmethod
    def _validate_tool_whitelist(cls, v: list[str]) -> list[str]:
        """Each tool name must be a non-empty string."""
        for tool in v:
            if not tool or not tool.strip():
                raise ValueError("tool_whitelist entries must be non-empty strings")
        return v


def load_spec(path: Path | str) -> AgentSpec:
    """Parse and validate a YAML agent spec from *path*.

    Raises:
        FileNotFoundError: if *path* does not exist.
        yaml.YAMLError: if the file is not valid YAML.
        pydantic.ValidationError: if any field is missing or invalid.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Agent spec must be a YAML mapping, got {type(raw).__name__}: {path}")
    return AgentSpec.model_validate(raw)
