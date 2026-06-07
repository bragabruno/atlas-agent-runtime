"""Persistence layer: SQLAlchemy models + DAL for agent_runs / agent_steps (ADR-010, AGT-6).

Public surface: tables (`AgentRun`, `AgentStep`, `AgentStatusEnum`, `StepTypeEnum`),
DAL functions (`create_run`, `update_run_status`, `append_step`, `get_run`,
`get_last_step_idx`, `list_steps`), and the declarative `Base`.
"""

from __future__ import annotations

from app.persistence.base import Base
from app.persistence.dal import (
    append_step,
    create_run,
    get_last_step_idx,
    get_run,
    list_steps,
    update_run_status,
)
from app.persistence.tables import AgentRun, AgentStatusEnum, AgentStep, StepTypeEnum

__all__ = [
    "Base",
    "AgentRun",
    "AgentStep",
    "AgentStatusEnum",
    "StepTypeEnum",
    "create_run",
    "update_run_status",
    "append_step",
    "get_run",
    "get_last_step_idx",
    "list_steps",
]
