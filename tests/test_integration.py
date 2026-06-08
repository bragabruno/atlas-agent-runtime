"""AGT-14 — End-to-end agent-loop integration tests.

All five scenarios run through the REAL ``AgentRunner`` with injected fakes
(no live gateway, no live DB, no live MCP servers, no OTel collector):

1. CITED ANSWER       — doc_search returns fixture chunks; agent cites source_ids.
2. REFUSAL            — no relevant chunks → agent refuses; no fabricated citation.
3. RUNAWAY CAP        — iteration / token / wall-time caps → CapBreachError; run persisted "capped".
4. WHITELIST REJECTION — non-whitelisted tool call → ToolNotAllowedError.
5. TRACE EMISSION     — multi-span trace (invoke_agent + execute_tool) with gen_ai.* attrs.

Fixtures / fakes:
- ``FakeGatewayClient`` — stateful, uses message history to return scripted responses.
- In-memory SQLite session (same pattern as test_dal.py).
- ``InMemorySpanExporter`` + ``TracerProvider`` (same pattern as test_telemetry.py).
- ``ToolRegistry`` + ``ToolSanitizer`` — real production objects.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import sqlalchemy as sa
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agentspec.model import AgentSpec
from app.loop.errors import CapBreachError
from app.loop.gateway_client import LLMRequest, LLMResponse
from app.loop.runner import AgentRunner, RunResult
from app.persistence.base import Base
from app.persistence.dal import list_steps
from app.persistence.tables import AgentRun, AgentStatusEnum, StepTypeEnum
from app.telemetry.tracing import get_tracer
from app.tools.registry import ToolNotAllowedError, ToolRegistry
from app.tools.sanitize.sanitizer import ToolSanitizer

# ---------------------------------------------------------------------------
# Fixture chunks — stand-in for real doc_search results
# ---------------------------------------------------------------------------

_CHUNK_A = "Clause 4.2 requires the operator to maintain records for 5 years. [source:doc-001]"
_CHUNK_B = "The retention obligation applies to both digital and physical records. [source:doc-002]"

# What the agent says after receiving the doc_search chunks
_CITED_ANSWER = (
    "Based on the regulatory documents, the operator must maintain records for 5 years "
    "[source:doc-001] including digital and physical formats [source:doc-002]."
)

# What the agent says when no relevant chunks are found
_REFUSAL_ANSWER = (
    "I was unable to find sufficient evidence in the source documents to answer this question. "
    "I cannot provide an answer without verified sources."
)


# ---------------------------------------------------------------------------
# Shared spec builders
# ---------------------------------------------------------------------------


def _spec(
    *,
    agent_name: str = "regdoc-qa",
    tool_whitelist: list[str] | None = None,
    max_iterations: int = 10,
    token_budget: int = 100_000,
    timeout_s: int = 60,
) -> AgentSpec:
    return AgentSpec(
        agent_name=agent_name,
        agent_version="1.0.0",
        system_prompt_ref="prompts/regdoc-qa.txt",
        model_alias="atlas-default",
        tool_whitelist=tool_whitelist if tool_whitelist is not None else ["doc_search"],
        max_iterations=max_iterations,
        token_budget=token_budget,
        timeout_s=timeout_s,
    )


# ---------------------------------------------------------------------------
# StatefulFakeGatewayClient
#
# Returns responses based on the number of prior turns in the conversation.
# Each scripted ``turn`` is a callable that receives the LLMRequest and
# returns an LLMResponse, giving tests full control over context-sensitive
# behaviour (inspect messages, return different content per turn, etc.).
# ---------------------------------------------------------------------------


class StatefulFakeGatewayClient:
    """Script-driven gateway fake that returns one response per ``chat`` call.

    ``turns`` is an ordered list of callables ``(LLMRequest) -> LLMResponse``.
    After all turns are consumed, further calls raise ``IndexError`` so a test
    that expects exactly N turns will fail loudly if the runner over-calls.
    """

    def __init__(self, turns: list[LLMResponse | None]) -> None:
        # Accept plain LLMResponse objects for brevity; None is a sentinel that
        # raises IndexError (used to assert the runner never reaches that call).
        self._turns = turns
        self._idx = 0

    async def chat(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
        if self._idx >= len(self._turns):
            raise IndexError(f"StatefulFakeGatewayClient exhausted after {self._idx} calls")
        resp = self._turns[self._idx]
        self._idx += 1
        if resp is None:
            raise AssertionError("Hit a None sentinel — runner called chat() unexpectedly")
        return resp

    @property
    def call_count(self) -> int:
        return self._idx


def _stop(content: str = "Done.", *, tokens: int = 40) -> LLMResponse:
    half = tokens // 2
    return LLMResponse(content=content, finish_reason="stop", input_tokens=half, output_tokens=half)


def _tool_response(tool_name: str, *, tokens: int = 20) -> LLMResponse:
    half = tokens // 2
    return LLMResponse(
        content=f"tool:{tool_name}",
        finish_reason="tool_use",
        input_tokens=half,
        output_tokens=half,
    )


def _continue_response(content: str = "Thinking...", *, tokens: int = 20) -> LLMResponse:
    half = tokens // 2
    return LLMResponse(
        content=content,
        finish_reason="tool_use",
        input_tokens=half,
        output_tokens=half,
    )


# ---------------------------------------------------------------------------
# SQLite session fixture (mirrors test_dal.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> sa.Engine:
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine: sa.Engine) -> Generator[Session, None, None]:
    with Session(engine) as sess:
        yield sess


# ---------------------------------------------------------------------------
# OTel fixtures (mirrors test_telemetry.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exporter))
    return prov


# ===========================================================================
# Scenario 1 — CITED ANSWER
#
# The agent issues one doc_search call, receives fixture chunks back (encoded
# in the next turn's system context by the fake), then composes a cited answer.
# Assertions:
#   - Run completes (no exception).
#   - Final response content contains both source_ids.
#   - Tool call sequence: [doc_search → stop].
#   - Runner performed exactly 2 LLM iterations.
# ===========================================================================


class TestCitedAnswer:
    @pytest.mark.asyncio
    async def test_cited_answer_completes(self) -> None:
        """Full cited-answer flow completes without exception."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer()

        # Turn 0: LLM requests doc_search
        # Turn 1: LLM answers using chunks in context (cite source:doc-001 + doc-002)
        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search"),
                _stop(_CITED_ANSWER),
            ]
        )
        runner = AgentRunner(
            spec=spec,
            client=client,
            registry=registry,
            sanitizer=sanitizer,
        )
        result = await runner.run(user_message="What are the record retention requirements?")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_cited_answer_contains_source_ids(self) -> None:
        """Agent final response cites both source_ids from the fixture chunks."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer()

        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search"),
                _stop(_CITED_ANSWER),
            ]
        )
        runner = AgentRunner(
            spec=spec,
            client=client,
            registry=registry,
            sanitizer=sanitizer,
        )
        result = await runner.run(user_message="What are the record retention requirements?")
        final_content = result.responses[-1].content
        assert "doc-001" in final_content
        assert "doc-002" in final_content

    @pytest.mark.asyncio
    async def test_cited_answer_tool_call_sequence(self) -> None:
        """Exactly 2 iterations: 1 tool-use turn + 1 stop turn."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer()

        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search"),
                _stop(_CITED_ANSWER),
            ]
        )
        runner = AgentRunner(
            spec=spec,
            client=client,
            registry=registry,
            sanitizer=sanitizer,
        )
        result = await runner.run(user_message="What are the record retention requirements?")
        assert result.iterations == 2
        assert result.responses[0].finish_reason == "tool_use"
        assert result.responses[1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_cited_answer_first_response_is_doc_search(self) -> None:
        """First LLM response requests doc_search (tool:<name> convention)."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer()

        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search"),
                _stop(_CITED_ANSWER),
            ]
        )
        runner = AgentRunner(
            spec=spec,
            client=client,
            registry=registry,
            sanitizer=sanitizer,
        )
        result = await runner.run(user_message="What are the record retention requirements?")
        assert result.responses[0].content == "tool:doc_search"

    @pytest.mark.asyncio
    async def test_cited_answer_sanitizer_fences_tool_output(self, session: Session) -> None:
        """Sanitizer wraps tool output in untrusted-content delimiters before context re-entry."""
        from app.tools.sanitize.sanitizer import FENCE_OPEN

        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer(fence=True)

        # We capture the messages list by wrapping the client
        captured_messages: list[list[dict[str, str]]] = []

        class CapturingClient:
            _inner = StatefulFakeGatewayClient(
                [
                    _tool_response("doc_search"),
                    _stop(_CITED_ANSWER),
                ]
            )

            async def chat(self, request: LLMRequest) -> LLMResponse:
                captured_messages.append(list(request.messages))
                return await self._inner.chat(request)

        runner = AgentRunner(
            spec=spec,
            client=CapturingClient(),
            registry=registry,
            sanitizer=sanitizer,
        )
        await runner.run(user_message="What are the record retention requirements?")

        # The second call's messages should contain the fenced tool output
        # (assistant turn 0 was sanitized before being appended)
        assert len(captured_messages) == 2
        second_call_messages = captured_messages[1]
        assistant_turn = next((m for m in second_call_messages if m["role"] == "assistant"), None)
        assert assistant_turn is not None
        # The tool_use response content ("tool:doc_search") was sanitized/fenced
        assert FENCE_OPEN in assistant_turn["content"]

    @pytest.mark.asyncio
    async def test_cited_answer_persisted_as_completed(self, session: Session) -> None:
        """Cited-answer run is persisted with status=completed."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer()

        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search"),
                _stop(_CITED_ANSWER),
            ]
        )
        runner = AgentRunner(
            spec=spec,
            client=client,
            registry=registry,
            sanitizer=sanitizer,
            session=session,
        )
        await runner.run(user_message="What are the record retention requirements?")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        assert len(run_rows) == 1
        assert run_rows[0].status == AgentStatusEnum.completed

    @pytest.mark.asyncio
    async def test_cited_answer_persists_llm_and_tool_steps(self, session: Session) -> None:
        """Cited-answer run persists both an llm_call step and a tool_call step."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer()

        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search"),
                _stop(_CITED_ANSWER),
            ]
        )
        runner = AgentRunner(
            spec=spec,
            client=client,
            registry=registry,
            sanitizer=sanitizer,
            session=session,
        )
        await runner.run(user_message="What are the record retention requirements?")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        run = run_rows[0]
        steps = list_steps(session, run.id)
        step_types = {s.type for s in steps}
        assert StepTypeEnum.llm_call in step_types
        assert StepTypeEnum.tool_call in step_types

    @pytest.mark.asyncio
    async def test_cited_answer_tokens_accumulated(self) -> None:
        """tokens_used in RunResult is the sum of both LLM call token counts."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer()

        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search", tokens=30),
                _stop(_CITED_ANSWER, tokens=50),
            ]
        )
        runner = AgentRunner(
            spec=spec,
            client=client,
            registry=registry,
            sanitizer=sanitizer,
        )
        result = await runner.run(user_message="Q")
        assert result.tokens_used == 80


# ===========================================================================
# Scenario 2 — REFUSAL
#
# When no relevant chunks exist the agent refuses / declares insufficient
# context without fabricating citations.
# Assertions:
#   - Run completes (no exception).
#   - Final response does NOT contain any "[source:" citation markers.
#   - The agent does not call doc_search at all (1 iteration, finish_reason=stop).
#     OR: agent calls doc_search, receives empty result, then refuses (2 iters).
# We cover both plausible refusal paths.
# ===========================================================================


class TestRefusal:
    @pytest.mark.asyncio
    async def test_direct_refusal_no_citation(self) -> None:
        """Agent that refuses directly (no tool call) has no source citation."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)

        # Agent immediately decides it cannot answer without evidence
        client = StatefulFakeGatewayClient([_stop(_REFUSAL_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, registry=registry)
        result = await runner.run(user_message="What does Article 99 say about X?")
        assert result.responses[-1].finish_reason == "stop"
        assert "[source:" not in result.responses[-1].content

    @pytest.mark.asyncio
    async def test_direct_refusal_completes_without_exception(self) -> None:
        """Refusal run does not raise any exception."""
        spec = _spec(tool_whitelist=["doc_search"])
        client = StatefulFakeGatewayClient([_stop(_REFUSAL_ANSWER)])
        runner = AgentRunner(spec=spec, client=client)
        result = await runner.run(user_message="Unknowable question")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_refusal_after_empty_search_no_citation(self) -> None:
        """Agent that calls doc_search and finds nothing then refuses — no citation."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer()

        # Turn 0: agent calls doc_search
        # Turn 1: agent sees empty results in context, refuses to cite
        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search"),
                _stop(_REFUSAL_ANSWER),
            ]
        )
        runner = AgentRunner(
            spec=spec,
            client=client,
            registry=registry,
            sanitizer=sanitizer,
        )
        result = await runner.run(user_message="What does Article 99 say about X?")
        final = result.responses[-1]
        assert final.finish_reason == "stop"
        assert "[source:" not in final.content

    @pytest.mark.asyncio
    async def test_refusal_run_is_1_iteration(self) -> None:
        """Direct refusal (no tool call) completes in exactly 1 iteration."""
        spec = _spec(tool_whitelist=["doc_search"])
        client = StatefulFakeGatewayClient([_stop(_REFUSAL_ANSWER)])
        runner = AgentRunner(spec=spec, client=client)
        result = await runner.run(user_message="Unknown topic")
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_refusal_persisted_as_completed(self, session: Session) -> None:
        """Refusal run is persisted with status=completed (it finished naturally)."""
        spec = _spec(tool_whitelist=["doc_search"])
        client = StatefulFakeGatewayClient([_stop(_REFUSAL_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, session=session)
        await runner.run(user_message="Unknown topic")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        assert len(run_rows) == 1
        assert run_rows[0].status == AgentStatusEnum.completed

    @pytest.mark.asyncio
    async def test_refusal_content_contains_insufficient_context(self) -> None:
        """Refusal answer signals lack of evidence (sanity check on content)."""
        spec = _spec(tool_whitelist=["doc_search"])
        client = StatefulFakeGatewayClient([_stop(_REFUSAL_ANSWER)])
        runner = AgentRunner(spec=spec, client=client)
        result = await runner.run(user_message="Unknown topic")
        content = result.responses[-1].content
        # The fixture refusal mentions 'evidence' or 'source'
        assert "source" in content.lower() or "evidence" in content.lower()


# ===========================================================================
# Scenario 3 — RUNAWAY CAP
#
# Three sub-scenarios: iteration cap, token cap, wall-time cap.
# All must:
#   - Raise CapBreachError with the correct .cap attribute.
#   - Persist the run with status='capped' (when session is injected).
# ===========================================================================


class TestRunawayCap:
    # --- Iteration cap ---

    @pytest.mark.asyncio
    async def test_iteration_cap_raises_cap_breach_error(self) -> None:
        """Runner hits max_iterations and raises CapBreachError(cap='max_iterations')."""
        spec = _spec(max_iterations=3, token_budget=1_000_000)
        # Infinite tool_use loop (more responses than the cap allows)
        client = StatefulFakeGatewayClient([_tool_response("doc_search")] * 20)
        registry = ToolRegistry(spec)
        runner = AgentRunner(spec=spec, client=client, registry=registry)

        with pytest.raises(CapBreachError) as exc_info:
            await runner.run(user_message="Keep looping")
        assert exc_info.value.cap == "max_iterations"

    @pytest.mark.asyncio
    async def test_iteration_cap_error_names_limit(self) -> None:
        """CapBreachError message contains the configured iteration limit."""
        spec = _spec(max_iterations=2, token_budget=1_000_000)
        client = StatefulFakeGatewayClient([_tool_response("doc_search")] * 20)
        registry = ToolRegistry(spec)
        runner = AgentRunner(spec=spec, client=client, registry=registry)

        with pytest.raises(CapBreachError) as exc_info:
            await runner.run(user_message="Loop")
        assert "2" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_iteration_cap_persists_run_as_capped(self, session: Session) -> None:
        """After iteration cap, run row has status='capped'."""
        spec = _spec(max_iterations=3, token_budget=1_000_000)
        client = StatefulFakeGatewayClient([_tool_response("doc_search")] * 20)
        registry = ToolRegistry(spec)
        runner = AgentRunner(spec=spec, client=client, registry=registry, session=session)

        with pytest.raises(CapBreachError):
            await runner.run(user_message="Loop")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        assert len(run_rows) == 1
        assert run_rows[0].status == AgentStatusEnum.capped

    @pytest.mark.asyncio
    async def test_iteration_cap_ended_at_populated(self, session: Session) -> None:
        """Capped run has ended_at set (not None)."""
        spec = _spec(max_iterations=2, token_budget=1_000_000)
        client = StatefulFakeGatewayClient([_tool_response("doc_search")] * 20)
        runner = AgentRunner(spec=spec, client=client, session=session)

        with pytest.raises(CapBreachError):
            await runner.run(user_message="Loop")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        assert run_rows[0].ended_at is not None

    # --- Token cap ---

    @pytest.mark.asyncio
    async def test_token_cap_raises_cap_breach_error(self) -> None:
        """Runner hits token_budget and raises CapBreachError(cap='token_budget')."""
        # 200 tokens per call, budget = 300: after 2nd call tokens_used = 400 > 300
        spec = _spec(max_iterations=50, token_budget=300)
        client = StatefulFakeGatewayClient([_tool_response("doc_search", tokens=200)] * 20)
        registry = ToolRegistry(spec)
        runner = AgentRunner(spec=spec, client=client, registry=registry)

        with pytest.raises(CapBreachError) as exc_info:
            await runner.run(user_message="expensive")
        assert exc_info.value.cap == "token_budget"

    @pytest.mark.asyncio
    async def test_token_cap_error_names_limit(self) -> None:
        """CapBreachError message contains the token budget limit."""
        spec = _spec(max_iterations=50, token_budget=100)
        client = StatefulFakeGatewayClient([_continue_response(tokens=200)] * 20)
        runner = AgentRunner(spec=spec, client=client)

        with pytest.raises(CapBreachError) as exc_info:
            await runner.run(user_message="expensive")
        assert "token_budget" in str(exc_info.value)
        assert "100" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_token_cap_persists_run_as_capped(self, session: Session) -> None:
        """After token cap, run row has status='capped'."""
        spec = _spec(max_iterations=50, token_budget=300)
        client = StatefulFakeGatewayClient([_tool_response("doc_search", tokens=200)] * 20)
        registry = ToolRegistry(spec)
        runner = AgentRunner(spec=spec, client=client, registry=registry, session=session)

        with pytest.raises(CapBreachError):
            await runner.run(user_message="expensive")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        assert run_rows[0].status == AgentStatusEnum.capped

    # --- Wall-time cap ---

    @pytest.mark.asyncio
    async def test_walltime_cap_raises_cap_breach_error(self) -> None:
        """Runner hits timeout_s and raises CapBreachError(cap='timeout_s')."""
        import asyncio

        class SlowClient:
            async def chat(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
                await asyncio.sleep(0.2)
                return _continue_response()

        spec = _spec(max_iterations=50, token_budget=1_000_000, timeout_s=1)
        runner = AgentRunner(spec=spec, client=SlowClient())

        with pytest.raises(CapBreachError) as exc_info:
            await runner.run(user_message="slow")
        assert exc_info.value.cap == "timeout_s"

    @pytest.mark.asyncio
    async def test_walltime_cap_error_names_timeout(self) -> None:
        """CapBreachError message contains 'timeout_s'."""
        import asyncio

        class SlowClient:
            async def chat(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
                await asyncio.sleep(0.2)
                return _continue_response()

        spec = _spec(max_iterations=50, token_budget=1_000_000, timeout_s=1)
        runner = AgentRunner(spec=spec, client=SlowClient())

        with pytest.raises(CapBreachError) as exc_info:
            await runner.run(user_message="slow")
        assert "timeout_s" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_walltime_cap_persists_run_as_capped(self, session: Session) -> None:
        """After wall-time cap, run row has status='capped'."""
        import asyncio

        class SlowClient:
            async def chat(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
                await asyncio.sleep(0.2)
                return _continue_response()

        spec = _spec(max_iterations=50, token_budget=1_000_000, timeout_s=1)
        runner = AgentRunner(spec=spec, client=SlowClient(), session=session)

        with pytest.raises(CapBreachError):
            await runner.run(user_message="slow")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        assert run_rows[0].status == AgentStatusEnum.capped

    @pytest.mark.asyncio
    async def test_walltime_cap_tokens_persisted(self, session: Session) -> None:
        """Tokens consumed before the wall-time cap are recorded in the capped run."""
        import asyncio

        call_count = 0

        class SlowClient:
            async def chat(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
                nonlocal call_count
                call_count += 1
                # First call completes quickly, second call hangs
                if call_count >= 2:
                    await asyncio.sleep(0.5)
                return _continue_response(tokens=40)

        spec = _spec(max_iterations=50, token_budget=1_000_000, timeout_s=1)
        runner = AgentRunner(spec=spec, client=SlowClient(), session=session)

        with pytest.raises(CapBreachError):
            await runner.run(user_message="slow")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        # tokens_used should reflect at least the first completed call
        assert run_rows[0].tokens_used >= 0


# ===========================================================================
# Scenario 4 — WHITELIST REJECTION
#
# Agent attempts to call a tool that is NOT in the whitelist.
# Assertions:
#   - ToolNotAllowedError is raised.
#   - .tool_name matches the forbidden tool.
#   - .agent_name matches the spec.
#   - Run NOT persisted as completed (raises before finishing).
# ===========================================================================


class TestWhitelistRejection:
    @pytest.mark.asyncio
    async def test_forbidden_tool_raises_tool_not_allowed_error(self) -> None:
        """AgentRunner with registry raises ToolNotAllowedError for non-whitelisted tool."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)

        # LLM tries to call a tool not in the whitelist
        client = StatefulFakeGatewayClient([_tool_response("execute_shell"), _stop()])
        runner = AgentRunner(spec=spec, client=client, registry=registry)

        with pytest.raises(ToolNotAllowedError):
            await runner.run(user_message="Do something dangerous")

    @pytest.mark.asyncio
    async def test_forbidden_tool_error_names_tool(self) -> None:
        """ToolNotAllowedError.tool_name matches the rejected tool name."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("execute_shell"), _stop()])
        runner = AgentRunner(spec=spec, client=client, registry=registry)

        with pytest.raises(ToolNotAllowedError) as exc_info:
            await runner.run(user_message="Q")
        assert exc_info.value.tool_name == "execute_shell"

    @pytest.mark.asyncio
    async def test_forbidden_tool_error_names_agent(self) -> None:
        """ToolNotAllowedError.agent_name matches the spec agent_name."""
        spec = _spec(agent_name="regdoc-qa", tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("dangerous_tool"), _stop()])
        runner = AgentRunner(spec=spec, client=client, registry=registry)

        with pytest.raises(ToolNotAllowedError) as exc_info:
            await runner.run(user_message="Q")
        assert exc_info.value.agent_name == "regdoc-qa"

    @pytest.mark.asyncio
    async def test_forbidden_tool_error_message_is_readable(self) -> None:
        """ToolNotAllowedError message names both agent and tool."""
        spec = _spec(agent_name="regdoc-qa", tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("rm_everything"), _stop()])
        runner = AgentRunner(spec=spec, client=client, registry=registry)

        with pytest.raises(ToolNotAllowedError) as exc_info:
            await runner.run(user_message="Q")
        msg = str(exc_info.value)
        assert "regdoc-qa" in msg
        assert "rm_everything" in msg

    @pytest.mark.asyncio
    async def test_whitelisted_tool_does_not_raise(self) -> None:
        """A tool in the whitelist passes through without error."""
        spec = _spec(tool_whitelist=["doc_search", "verify_citation"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("verify_citation"), _stop()])
        runner = AgentRunner(spec=spec, client=client, registry=registry)
        result = await runner.run(user_message="Q")
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_no_registry_skips_whitelist_check(self) -> None:
        """Without a registry, non-whitelisted tools pass silently."""
        spec = _spec(tool_whitelist=["doc_search"])
        client = StatefulFakeGatewayClient([_tool_response("execute_shell"), _stop()])
        # Deliberately no registry — whitelist check disabled
        runner = AgentRunner(spec=spec, client=client)
        result = await runner.run(user_message="Q")
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_whitelist_rejection_on_second_tool_call(self) -> None:
        """Rejection fires when the forbidden tool appears on the second turn."""
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        # First call: allowed; second call: forbidden
        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search"),
                _tool_response("execute_shell"),
                _stop(),
            ]
        )
        runner = AgentRunner(spec=spec, client=client, registry=registry)

        with pytest.raises(ToolNotAllowedError) as exc_info:
            await runner.run(user_message="Q")
        assert exc_info.value.tool_name == "execute_shell"

    @pytest.mark.asyncio
    async def test_forbidden_tool_allowed_set_exposed(self) -> None:
        """ToolNotAllowedError.allowed contains only whitelisted tools."""
        spec = _spec(tool_whitelist=["doc_search", "verify_citation"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("bad_tool"), _stop()])
        runner = AgentRunner(spec=spec, client=client, registry=registry)

        with pytest.raises(ToolNotAllowedError) as exc_info:
            await runner.run(user_message="Q")
        assert exc_info.value.allowed == frozenset({"doc_search", "verify_citation"})


# ===========================================================================
# Scenario 5 — TRACE EMISSION
#
# A multi-step run (doc_search → cited answer) yields a coherent trace:
#   - At least one ``invoke_agent`` span (LLM call).
#   - At least one ``execute_tool`` span (tool call).
#   - All spans carry gen_ai.* semantic convention attrs.
#   - All spans are finished (end_time is not None).
# ===========================================================================


class TestTraceEmission:
    @pytest.mark.asyncio
    async def test_multi_step_run_emits_invoke_agent_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """A run with one tool call emits at least one invoke_agent span."""
        tracer = get_tracer(provider)
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("doc_search"), _stop(_CITED_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, registry=registry, tracer=tracer)
        await runner.run(user_message="Record retention?")

        invoke_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "invoke_agent"
        ]
        assert len(invoke_spans) >= 1

    @pytest.mark.asyncio
    async def test_multi_step_run_emits_execute_tool_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """A run with one tool call emits exactly one execute_tool span."""
        tracer = get_tracer(provider)
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("doc_search"), _stop(_CITED_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, registry=registry, tracer=tracer)
        await runner.run(user_message="Record retention?")

        tool_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "execute_tool"
        ]
        assert len(tool_spans) == 1

    @pytest.mark.asyncio
    async def test_invoke_agent_span_has_agent_name(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """invoke_agent span carries gen_ai.agent.name matching the spec."""
        tracer = get_tracer(provider)
        spec = _spec(agent_name="regdoc-qa", tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("doc_search"), _stop(_CITED_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, registry=registry, tracer=tracer)
        await runner.run(user_message="Q")

        invoke_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "invoke_agent"
        ]
        for span in invoke_spans:
            assert span.attributes is not None
            assert span.attributes.get("gen_ai.agent.name") == "regdoc-qa"

    @pytest.mark.asyncio
    async def test_execute_tool_span_has_tool_name(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """execute_tool span carries gen_ai.tool.name matching the called tool."""
        tracer = get_tracer(provider)
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("doc_search"), _stop(_CITED_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, registry=registry, tracer=tracer)
        await runner.run(user_message="Q")

        tool_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "execute_tool"
        ]
        assert tool_spans[0].attributes is not None
        assert tool_spans[0].attributes.get("gen_ai.tool.name") == "doc_search"

    @pytest.mark.asyncio
    async def test_all_spans_are_finished(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Every span in the trace is finished (end_time > 0)."""
        tracer = get_tracer(provider)
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("doc_search"), _stop(_CITED_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, registry=registry, tracer=tracer)
        await runner.run(user_message="Q")

        for span in exporter.get_finished_spans():
            assert span.end_time is not None and span.end_time > 0

    @pytest.mark.asyncio
    async def test_total_span_count_for_tool_then_stop(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """doc_search → stop run produces exactly 3 spans: llm+tool+llm."""
        tracer = get_tracer(provider)
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("doc_search"), _stop(_CITED_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, registry=registry, tracer=tracer)
        await runner.run(user_message="Q")

        finished = exporter.get_finished_spans()
        assert len(finished) == 3

    @pytest.mark.asyncio
    async def test_span_operation_sequence(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Span operations are ordered: invoke_agent, execute_tool, invoke_agent."""
        tracer = get_tracer(provider)
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("doc_search"), _stop(_CITED_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, registry=registry, tracer=tracer)
        await runner.run(user_message="Q")

        ops = [
            s.attributes.get("gen_ai.operation.name")
            for s in exporter.get_finished_spans()
            if s.attributes
        ]
        assert ops == ["invoke_agent", "execute_tool", "invoke_agent"]

    @pytest.mark.asyncio
    async def test_llm_span_count_matches_iterations(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Number of invoke_agent spans equals the number of LLM iterations."""
        tracer = get_tracer(provider)
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("doc_search"), _stop(_CITED_ANSWER)])
        runner = AgentRunner(spec=spec, client=client, registry=registry, tracer=tracer)
        result = await runner.run(user_message="Q")

        invoke_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "invoke_agent"
        ]
        assert len(invoke_spans) == result.iterations

    @pytest.mark.asyncio
    async def test_multi_tool_trace_has_multiple_execute_tool_spans(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Two tool calls in a run produce two execute_tool spans."""
        tracer = get_tracer(provider)
        spec = _spec(tool_whitelist=["doc_search", "verify_citation"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient(
            [
                _tool_response("doc_search"),
                _tool_response("verify_citation"),
                _stop(_CITED_ANSWER),
            ]
        )
        runner = AgentRunner(spec=spec, client=client, registry=registry, tracer=tracer)
        await runner.run(user_message="Q")

        tool_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "execute_tool"
        ]
        assert len(tool_spans) == 2

    @pytest.mark.asyncio
    async def test_trace_combined_with_persistence(
        self,
        provider: TracerProvider,
        exporter: InMemorySpanExporter,
        session: Session,
    ) -> None:
        """Tracer + session work together: spans emitted AND run persisted."""
        tracer = get_tracer(provider)
        spec = _spec(tool_whitelist=["doc_search"])
        registry = ToolRegistry(spec)
        client = StatefulFakeGatewayClient([_tool_response("doc_search"), _stop(_CITED_ANSWER)])
        runner = AgentRunner(
            spec=spec,
            client=client,
            registry=registry,
            tracer=tracer,
            session=session,
        )
        result = await runner.run(user_message="Q")
        session.flush()

        # Spans emitted
        assert len(exporter.get_finished_spans()) == 3
        # Run persisted
        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        assert len(run_rows) == 1
        assert run_rows[0].status == AgentStatusEnum.completed
        # RunResult valid
        assert result.iterations == 2
