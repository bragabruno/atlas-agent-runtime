"""AgentRunner — bounded asyncio loop with hard caps (AGT-3).

Enforces three independent hard caps drawn from `AgentSpec`:
  - ``max_iterations``: total LLM→(optional tool) cycles.
  - ``token_budget``: cumulative tokens across all LLM calls.
  - ``timeout_s``: wall-time for the entire run.

A breach on **any** cap immediately raises `CapBreachError` with an explicit
message naming the cap — never silent. See atlas-docs ADR-006.

The gateway is accessed through an injected `GatewayClient` — the runner
has no import of any concrete HTTP client, making it fully testable offline.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.agentspec.model import AgentSpec
from app.loop.errors import CapBreachError
from app.loop.gateway_client import GatewayClient, LLMRequest, LLMResponse


@dataclass
class RunResult:
    """Summary of a completed agent run (no cap breach).

    Callers can inspect ``iterations``, ``tokens_used``, and the ordered
    ``responses`` list to reconstruct what happened without reading the DB.
    """

    iterations: int
    tokens_used: int
    elapsed_s: float
    responses: list[LLMResponse] = field(default_factory=list)


class AgentRunner:
    """Executes an agent loop bounded by the hard caps in `AgentSpec`.

    Usage::

        runner = AgentRunner(spec=spec, client=client)
        result = await runner.run(user_message="What does clause 4.2 require?")

    The `client` must satisfy the `GatewayClient` protocol; pass a
    `FakeGatewayClient` from tests or an `HttpGatewayClient` in production.
    """

    def __init__(self, *, spec: AgentSpec, client: GatewayClient) -> None:
        self._spec = spec
        self._client = client

    async def run(self, *, user_message: str) -> RunResult:
        """Execute the agent loop and return a `RunResult`.

        Raises:
            CapBreachError: if ``max_iterations``, ``token_budget``, or
                ``timeout_s`` is exceeded before the run completes naturally.
        """
        spec = self._spec
        messages: list[dict[str, str]] = [{"role": "user", "content": user_message}]
        responses: list[LLMResponse] = []
        tokens_used = 0
        iterations = 0
        started = time.monotonic()

        async def _one_iteration() -> LLMResponse:
            """Call the gateway for one LLM turn, updating shared counters."""
            nonlocal tokens_used, iterations

            # --- wall-time cap (checked before each call) ---
            elapsed = time.monotonic() - started
            if elapsed >= spec.timeout_s:
                raise CapBreachError("timeout_s", spec.timeout_s, "s")

            # --- iteration cap (checked before each call) ---
            if iterations >= spec.max_iterations:
                raise CapBreachError("max_iterations", spec.max_iterations)

            request = LLMRequest(
                model_alias=spec.model_alias,
                system_prompt=spec.system_prompt_ref,
                messages=messages,
                max_tokens=None,
            )
            response = await self._client.chat(request)
            iterations += 1

            # --- token cap (checked after each call, before continuing) ---
            tokens_used += response.total_tokens
            if tokens_used > spec.token_budget:
                raise CapBreachError("token_budget", spec.token_budget)

            return response

        # Wrap the loop body in asyncio.wait_for so the OS-level timeout fires
        # even when a gateway call hangs without yielding.
        remaining = spec.timeout_s - (time.monotonic() - started)
        try:
            while True:
                remaining = spec.timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise CapBreachError("timeout_s", spec.timeout_s, "s")

                try:
                    response = await asyncio.wait_for(
                        _one_iteration(),
                        timeout=remaining,
                    )
                except TimeoutError:
                    raise CapBreachError("timeout_s", spec.timeout_s, "s") from None

                responses.append(response)
                messages.append({"role": "assistant", "content": response.content})

                if response.finish_reason == "stop":
                    break

        except CapBreachError:
            raise  # re-raise without wrapping

        elapsed_s = time.monotonic() - started
        return RunResult(
            iterations=iterations,
            tokens_used=tokens_used,
            elapsed_s=elapsed_s,
            responses=responses,
        )
