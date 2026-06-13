#!/usr/bin/env python3
"""AGT-15 — Demo: RegDoc agent end-to-end.

Run from the repo root:
    python scripts/demo_regdoc_agent.py


Demonstrates four scenarios with the RegDoc Q&A agent using a
``FakeGatewayClient`` — no live gateway or corpus needed.

Scenarios
---------
1. **Cited answer**   — agent answers a regulation question with citations.
2. **Refusal**        — agent refuses an unanswerable question (no supporting docs).
3. **Runaway cap**    — agent exceeds ``max_iterations``; ``CapBreachError`` raised.
4. **Trace emission** — OTel span tags logged to stdout via a simple exporter.

Usage
-----
    python scripts/demo_regdoc_agent.py

Expected output (abbreviated)
------------------------------
    [SCENARIO 1] Cited answer ...
    Agent: Article 6(1)(a) of GDPR requires explicit consent ...  [src:reg-042]
    Run completed: 1 iteration(s), 320 tokens

    [SCENARIO 2] Refusal ...
    Agent: I cannot answer this question because no supporting documents were found.
    Run completed: 1 iteration(s), 85 tokens

    [SCENARIO 3] Runaway cap ...
    CapBreachError caught: CapBreachError: max_iterations (3) exceeded

    [SCENARIO 4] Trace emission ...
    Span: agent.llm_call  gen_ai.agent.name=regdoc-qa  tokens=320
    Span: agent.llm_call  gen_ai.agent.name=regdoc-qa  tokens=85
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

# Allow running from scripts/ or from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agentspec.model import AgentSpec
from app.loop.errors import CapBreachError
from app.loop.gateway_client import LLMResponse
from app.loop.runner import AgentRunner, RunResult

# ---------------------------------------------------------------------------
# Fake gateway responses
# ---------------------------------------------------------------------------


def _cited_answer() -> LLMResponse:
    return LLMResponse(
        content=(
            "Article 6(1)(a) of GDPR requires explicit consent from data subjects "
            "before processing personal data for marketing purposes [src:reg-042]. "
            "Recital 32 further clarifies that consent must be granular [src:reg-043]."
        ),
        finish_reason="stop",
        input_tokens=180,
        output_tokens=140,
    )


def _refusal() -> LLMResponse:
    return LLMResponse(
        content=(
            "I cannot answer this question because no supporting documents were found "
            "in the regulatory corpus. Please consult a compliance professional."
        ),
        finish_reason="stop",
        input_tokens=60,
        output_tokens=25,
    )


def _endless_loop_response() -> LLMResponse:
    return LLMResponse(
        content="Let me search for more information.",
        finish_reason="tool_use",
        input_tokens=40,
        output_tokens=20,
    )


# ---------------------------------------------------------------------------
# Minimal AgentSpec for the demo
# ---------------------------------------------------------------------------

_SPEC = AgentSpec(
    agent_name="regdoc-qa",
    agent_version="1.0.0",
    system_prompt_ref="prompts/regdoc-qa/1.0.0/system.txt",
    model_alias="claude-sonnet-4-6",
    tool_whitelist=["doc_search", "verify_citation"],
    max_iterations=8,
    token_budget=4096,
    timeout_s=60,
)

_SPEC_TIGHT = AgentSpec(
    agent_name="regdoc-qa",
    agent_version="1.0.0",
    system_prompt_ref="prompts/regdoc-qa/1.0.0/system.txt",
    model_alias="claude-sonnet-4-6",
    tool_whitelist=["doc_search", "verify_citation"],
    max_iterations=3,  # tight cap to trigger breach
    token_budget=4096,
    timeout_s=60,
)


# ---------------------------------------------------------------------------
# Simple span collector for trace demo
# ---------------------------------------------------------------------------

_SPANS: list[dict[str, Any]] = []


class _CapturingTracer:
    """Stub OTel tracer that records span start events."""

    class _Span:
        def __init__(self, name: str) -> None:
            self.name = name
            self._attrs: dict[str, Any] = {}

        def set_attribute(self, key: str, value: Any) -> None:
            self._attrs[key] = value

        def set_status(self, *_: Any) -> None:
            pass

        def end(self) -> None:
            _SPANS.append({"name": self.name, **self._attrs})

        def __enter__(self) -> _CapturingTracer._Span:
            return self

        def __exit__(self, *_: Any) -> None:
            self.end()

    def start_span(self, name: str, **_kwargs: Any) -> _CapturingTracer._Span:
        return self._Span(name)

    def start_as_current_span(self, name: str, **_kwargs: Any) -> _CapturingTracer._Span:
        return self._Span(name)


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------


async def _scenario_cited_answer() -> None:
    print("[SCENARIO 1] Cited answer ...")
    client = AsyncMock()
    client.chat.return_value = _cited_answer()

    runner = AgentRunner(spec=_SPEC, client=client)
    result: RunResult = await runner.run(
        user_message="What GDPR article governs consent for marketing emails?"
    )
    final = result.responses[-1].content if result.responses else "(no response)"
    print(f"Agent: {final}")
    print(f"Run completed: {result.iterations} iteration(s), {result.tokens_used} tokens")
    print()


async def _scenario_refusal() -> None:
    print("[SCENARIO 2] Refusal (unanswerable question) ...")
    client = AsyncMock()
    client.chat.return_value = _refusal()

    runner = AgentRunner(spec=_SPEC, client=client)
    result: RunResult = await runner.run(user_message="What is the capital of France?")
    final = result.responses[-1].content if result.responses else "(no response)"
    print(f"Agent: {final}")
    print(f"Run completed: {result.iterations} iteration(s), {result.tokens_used} tokens")
    print()


async def _scenario_runaway_cap() -> None:
    print("[SCENARIO 3] Runaway cap (max_iterations=3) ...")
    client = AsyncMock()
    client.chat.return_value = _endless_loop_response()

    runner = AgentRunner(spec=_SPEC_TIGHT, client=client)
    try:
        await runner.run(user_message="Keep calling tools forever.")
        print("ERROR: expected CapBreachError but run completed")
    except CapBreachError as exc:
        print(f"CapBreachError caught: {exc}")
    print()


async def _scenario_trace_emission() -> None:
    print("[SCENARIO 4] Trace emission ...")
    _SPANS.clear()
    tracer = _CapturingTracer()

    async def _run_with_tracer(response: LLMResponse) -> None:
        client = AsyncMock()
        client.chat.return_value = response
        runner = AgentRunner(spec=_SPEC, client=client, tracer=tracer)  # type: ignore[arg-type]
        await runner.run(user_message="What are the data retention rules?")

    await _run_with_tracer(_cited_answer())
    await _run_with_tracer(_refusal())

    for span in _SPANS:
        attrs = "  ".join(f"{k}={v}" for k, v in span.items() if k != "name")
        print(f"Span: {span['name']:<30} {attrs}")
    print()


async def _main() -> int:
    print("=" * 65)
    print("Atlas RegDoc agent demo (AGT-15)")
    print("=" * 65)
    print()

    await _scenario_cited_answer()
    await _scenario_refusal()
    await _scenario_runaway_cap()
    await _scenario_trace_emission()

    print("Demo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
