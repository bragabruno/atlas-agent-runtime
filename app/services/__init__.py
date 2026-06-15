"""Service layer — use-case orchestration over the agent-runtime internals.

Public surface: `RunService` (AGT-16) and its domain errors. The HTTP layer
(`app.api`) depends on these, not on `AgentRunner`/DAL directly (ADR-016).
"""

from __future__ import annotations

from app.services.run_service import (
    AgentNotFoundError,
    RunNotFoundError,
    RunService,
    make_dir_spec_loader,
)

__all__ = [
    "RunService",
    "AgentNotFoundError",
    "RunNotFoundError",
    "make_dir_spec_loader",
]
