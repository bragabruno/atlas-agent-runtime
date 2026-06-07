"""AgentSpec — Pydantic model for YAML-defined agent declarations (AGT-1).

Public surface: `AgentSpec`, `load_spec`.
"""

from __future__ import annotations

from app.agentspec.model import AgentSpec, load_spec

__all__ = ["AgentSpec", "load_spec"]
