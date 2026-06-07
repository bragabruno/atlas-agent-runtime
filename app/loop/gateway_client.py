"""GatewayClient — thin Protocol that decouples the agent loop from transport (AGT-3).

The loop depends on this interface, not on the concrete HTTP implementation.
Offline tests inject a `FakeGatewayClient`; production code uses
`HttpGatewayClient` (backed by `httpx`).

Mirrors the provider Pattern from atlas-gateway (structural Protocol, no
registry needed here). See atlas-docs ADR-006 + ADR-016.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """The minimal payload sent to the Gateway for one LLM call."""

    model_config = {"frozen": True}

    model_alias: str
    system_prompt: str
    messages: list[dict[str, str]]
    max_tokens: int | None = None


class LLMResponse(BaseModel):
    """Normalized response from the Gateway for one LLM call."""

    model_config = {"frozen": True}

    content: str
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class GatewayClient(Protocol):
    """Structural protocol satisfied by any gateway client adapter.

    The agent loop calls only `chat`; it never imports a concrete class.
    """

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Send one LLM request and return the normalized response."""
        ...
