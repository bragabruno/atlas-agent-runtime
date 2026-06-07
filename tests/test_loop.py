"""AGT-3 AgentRunner tests — fully offline, no live gateway required.

The `FakeGatewayClient` is deterministic: callers configure how many tokens
each response should report and whether the finish_reason is 'stop' or keeps
looping.  Every test injects the fake so the loop stays entirely in-process.

Covers:
- Normal bounded run completes (finish_reason='stop') and returns RunResult.
- Iteration cap: loop stops at max_iterations with CapBreachError naming cap.
- Token cap: loop stops when cumulative tokens exceed token_budget.
- Wall-time cap: loop stops when timeout_s elapses (uses very short timeout).
- CapBreachError carries the correct cap name, limit, and a human-readable msg.
- GatewayClient is a structural Protocol — FakeGatewayClient satisfies it.
- RunResult accumulates correct iteration count and token sum.
- Loop handles a single-iteration run (max_iterations=1).
"""

from __future__ import annotations

import asyncio

import pytest

from app.agentspec.model import AgentSpec
from app.loop.errors import CapBreachError
from app.loop.gateway_client import GatewayClient, LLMRequest, LLMResponse
from app.loop.runner import AgentRunner, RunResult

# ---------------------------------------------------------------------------
# FakeGatewayClient — deterministic offline stand-in for the real HTTP client
# ---------------------------------------------------------------------------


class FakeGatewayClient:
    """Offline gateway client that returns pre-configured responses in order.

    After the response list is exhausted it raises `StopIteration` so tests
    that expect the loop to finish naturally can verify iteration counts.
    If ``delay_s`` is set each call sleeps that long (used for timeout tests).
    """

    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        delay_s: float = 0.0,
    ) -> None:
        self._responses = list(responses)
        self._index = 0
        self._delay_s = delay_s

    async def chat(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._index >= len(self._responses):
            raise IndexError("FakeGatewayClient ran out of configured responses")
        resp = self._responses[self._index]
        self._index += 1
        return resp


def _stop_response(*, input_tokens: int = 100, output_tokens: int = 100) -> LLMResponse:
    """Return a terminal response (finish_reason='stop')."""
    return LLMResponse(
        content="Answer here.",
        finish_reason="stop",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _continue_response(*, input_tokens: int = 100, output_tokens: int = 100) -> LLMResponse:
    """Return a non-terminal response (simulates tool-call or multi-turn)."""
    return LLMResponse(
        content="Thinking...",
        finish_reason="tool_use",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _spec(
    *,
    max_iterations: int = 10,
    token_budget: int = 100_000,
    timeout_s: int = 60,
) -> AgentSpec:
    return AgentSpec(
        agent_name="test-agent",
        agent_version="1.0.0",
        system_prompt_ref="prompts/test.txt",
        model_alias="atlas-default",
        tool_whitelist=["doc_search"],
        max_iterations=max_iterations,
        token_budget=token_budget,
        timeout_s=timeout_s,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_fake_client_satisfies_gateway_protocol() -> None:
    """FakeGatewayClient must structurally satisfy GatewayClient Protocol."""
    client = FakeGatewayClient([_stop_response()])
    assert isinstance(client, GatewayClient)


# ---------------------------------------------------------------------------
# Normal bounded run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_run_completes() -> None:
    """A run that finishes at finish_reason='stop' returns a RunResult."""
    client = FakeGatewayClient([_stop_response(input_tokens=50, output_tokens=50)])
    runner = AgentRunner(spec=_spec(), client=client)
    result = await runner.run(user_message="Summarize clause 4.2")
    assert isinstance(result, RunResult)
    assert result.iterations == 1
    assert result.tokens_used == 100
    assert result.elapsed_s >= 0


@pytest.mark.asyncio
async def test_multi_turn_run_completes() -> None:
    """A run with several turns completes when the last response is 'stop'."""
    responses = [
        _continue_response(input_tokens=10, output_tokens=10),
        _continue_response(input_tokens=10, output_tokens=10),
        _stop_response(input_tokens=10, output_tokens=10),
    ]
    client = FakeGatewayClient(responses)
    runner = AgentRunner(spec=_spec(max_iterations=5), client=client)
    result = await runner.run(user_message="Hello")
    assert result.iterations == 3
    assert result.tokens_used == 60
    assert len(result.responses) == 3


@pytest.mark.asyncio
async def test_single_iteration_spec() -> None:
    """max_iterations=1 with a single stop response completes cleanly."""
    client = FakeGatewayClient([_stop_response()])
    runner = AgentRunner(spec=_spec(max_iterations=1), client=client)
    result = await runner.run(user_message="Q")
    assert result.iterations == 1


# ---------------------------------------------------------------------------
# Iteration cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iteration_cap_raises_cap_breach_error() -> None:
    """Loop must raise CapBreachError when max_iterations is reached."""
    # Provide infinite non-stop responses; the cap fires first.
    responses = [_continue_response()] * 20
    client = FakeGatewayClient(responses)
    spec = _spec(max_iterations=3)
    runner = AgentRunner(spec=spec, client=client)
    with pytest.raises(CapBreachError) as exc_info:
        await runner.run(user_message="Loop me")
    err = exc_info.value
    assert err.cap == "max_iterations"
    assert err.limit == 3


@pytest.mark.asyncio
async def test_iteration_cap_error_message_names_cap() -> None:
    """CapBreachError message must contain 'max_iterations' and the limit."""
    responses = [_continue_response()] * 10
    client = FakeGatewayClient(responses)
    runner = AgentRunner(spec=_spec(max_iterations=2), client=client)
    with pytest.raises(CapBreachError) as exc_info:
        await runner.run(user_message="X")
    msg = str(exc_info.value)
    assert "max_iterations" in msg
    assert "2" in msg


# ---------------------------------------------------------------------------
# Token cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_cap_raises_cap_breach_error() -> None:
    """Loop must raise CapBreachError when cumulative tokens exceed budget."""
    # 200 tokens per call, budget=300 — third call would push it to 600 > 300.
    # But the cap fires after the second call (tokens_used=400 > 300).
    responses = [_continue_response(input_tokens=100, output_tokens=100)] * 10
    client = FakeGatewayClient(responses)
    spec = _spec(token_budget=300, max_iterations=20)
    runner = AgentRunner(spec=spec, client=client)
    with pytest.raises(CapBreachError) as exc_info:
        await runner.run(user_message="expensive")
    err = exc_info.value
    assert err.cap == "token_budget"
    assert err.limit == 300


@pytest.mark.asyncio
async def test_token_cap_error_message_names_cap() -> None:
    """CapBreachError message must contain 'token_budget' and the limit."""
    responses = [_continue_response(input_tokens=500, output_tokens=500)] * 5
    client = FakeGatewayClient(responses)
    runner = AgentRunner(spec=_spec(token_budget=100, max_iterations=10), client=client)
    with pytest.raises(CapBreachError) as exc_info:
        await runner.run(user_message="X")
    msg = str(exc_info.value)
    assert "token_budget" in msg
    assert "100" in msg


# ---------------------------------------------------------------------------
# Wall-time cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_walltime_cap_raises_cap_breach_error() -> None:
    """Loop must raise CapBreachError when timeout_s wall-time elapses."""
    # Each fake response takes 0.15s; timeout is 0.1s → first call will
    # exhaust the budget.
    slow_client = FakeGatewayClient(
        [_continue_response()] * 10,
        delay_s=0.15,
    )
    spec = _spec(timeout_s=1, max_iterations=50, token_budget=1_000_000)
    runner = AgentRunner(spec=spec, client=slow_client)
    with pytest.raises(CapBreachError) as exc_info:
        await runner.run(user_message="slow")
    err = exc_info.value
    assert err.cap == "timeout_s"
    assert err.limit == 1


@pytest.mark.asyncio
async def test_walltime_cap_error_message_names_cap() -> None:
    """CapBreachError message must contain 'timeout_s' and the limit."""
    slow_client = FakeGatewayClient([_continue_response()] * 10, delay_s=0.2)
    spec = _spec(timeout_s=1, max_iterations=50, token_budget=1_000_000)
    runner = AgentRunner(spec=spec, client=slow_client)
    with pytest.raises(CapBreachError) as exc_info:
        await runner.run(user_message="slow")
    msg = str(exc_info.value)
    assert "timeout_s" in msg


# ---------------------------------------------------------------------------
# CapBreachError unit tests (independent of AgentRunner)
# ---------------------------------------------------------------------------


def test_cap_breach_error_stores_cap_name() -> None:
    err = CapBreachError("max_iterations", 10)
    assert err.cap == "max_iterations"


def test_cap_breach_error_stores_limit() -> None:
    err = CapBreachError("token_budget", 32000)
    assert err.limit == 32000


def test_cap_breach_error_message_format_iterations() -> None:
    err = CapBreachError("max_iterations", 10)
    assert str(err) == "CapBreachError: max_iterations (10) exceeded"


def test_cap_breach_error_message_format_tokens() -> None:
    err = CapBreachError("token_budget", 32000)
    assert str(err) == "CapBreachError: token_budget (32000) exceeded"


def test_cap_breach_error_message_format_walltime() -> None:
    err = CapBreachError("timeout_s", 120, "s")
    assert str(err) == "CapBreachError: timeout_s (120 s) exceeded"


def test_cap_breach_error_is_exception() -> None:
    err = CapBreachError("timeout_s", 60, "s")
    assert isinstance(err, Exception)
