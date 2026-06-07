"""Agent loop — AgentRunner with hard caps (AGT-3).

Public surface: `AgentRunner`, `CapBreachError`, `GatewayClient`,
`LLMRequest`, `LLMResponse`.
"""

from __future__ import annotations

from app.loop.errors import CapBreachError
from app.loop.gateway_client import GatewayClient, LLMRequest, LLMResponse
from app.loop.runner import AgentRunner

__all__ = [
    "AgentRunner",
    "CapBreachError",
    "GatewayClient",
    "LLMRequest",
    "LLMResponse",
]
