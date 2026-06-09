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


class MockGatewayClient:
    """Deterministic, offline gateway client (the "via Mock" default — AGT-16).

    Returns a single ``finish_reason='stop'`` echo response that is a pure
    function of the conversation, so a run started over the HTTP trigger surface
    completes to a terminal ``succeeded`` state with **no** network, API keys, or
    real Gateway. Mirrors the role of atlas-gateway's `MockProvider`.

    The reply echoes the last user message; token counts are a cheap
    whitespace-word estimate so persisted ``tokens_used`` is non-zero and stable.
    Tests that need multi-turn / tool-call / cap-breach scenarios inject their
    own scripted fake instead.
    """

    async def chat(self, request: LLMRequest) -> LLMResponse:
        last = request.messages[-1]["content"] if request.messages else ""
        reply = f"[mock:{request.model_alias}] echo: {last}"
        return LLMResponse(
            content=reply,
            finish_reason="stop",
            input_tokens=max(1, len(last.split())),
            output_tokens=max(1, len(reply.split())),
        )
