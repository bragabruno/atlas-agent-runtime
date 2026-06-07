"""AGT-7 OTel telemetry tests — fully offline using in-memory span exporter.

Covers:
- get_tracer returns a Tracer from an injected TracerProvider.
- span_llm_call produces a span with gen_ai.operation.name='invoke_agent'.
- span_llm_call span has gen_ai.agent.name set correctly.
- span_tool_call produces a span with gen_ai.operation.name='execute_tool'.
- span_tool_call span has gen_ai.agent.name and gen_ai.tool.name set.
- A full AgentRunner run with tracer produces a coherent multi-span trace.
- LLM span count matches iteration count.
- Tool spans are produced for tool_use responses.
- Spans are properly ended (not live) after the run.
- span_llm_call caller can set additional attributes on the span.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.agentspec.model import AgentSpec
from app.loop.gateway_client import LLMRequest, LLMResponse
from app.loop.runner import AgentRunner
from app.telemetry.tracing import get_tracer, span_llm_call, span_tool_call
from app.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Fixture: in-memory OTel provider
# ---------------------------------------------------------------------------


@pytest.fixture()
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exporter))
    return prov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(whitelist: list[str] | None = None) -> AgentSpec:
    return AgentSpec(
        agent_name="telemetry-agent",
        agent_version="1.0.0",
        system_prompt_ref="prompts/test.txt",
        model_alias="atlas-default",
        tool_whitelist=whitelist if whitelist is not None else ["doc_search"],
        max_iterations=10,
        token_budget=100_000,
        timeout_s=60,
    )


class FakeGatewayClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def chat(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
        if self._idx >= len(self._responses):
            raise IndexError("FakeGatewayClient exhausted")
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


def _stop() -> LLMResponse:
    return LLMResponse(content="Done.", finish_reason="stop", input_tokens=10, output_tokens=10)


def _tool_use(tool_name: str = "doc_search") -> LLMResponse:
    return LLMResponse(
        content=f"tool:{tool_name}",
        finish_reason="tool_use",
        input_tokens=10,
        output_tokens=10,
    )


# ---------------------------------------------------------------------------
# get_tracer tests
# ---------------------------------------------------------------------------


class TestGetTracer:
    def test_returns_tracer_from_provider(self, provider: TracerProvider) -> None:
        tracer = get_tracer(provider)
        assert tracer is not None

    def test_no_provider_uses_global(self) -> None:
        tracer = get_tracer(None)
        assert tracer is not None


# ---------------------------------------------------------------------------
# span_llm_call tests
# ---------------------------------------------------------------------------


class TestSpanLlmCall:
    @pytest.mark.asyncio
    async def test_produces_one_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        async with span_llm_call(tracer, agent_name="telemetry-agent"):
            pass
        spans = exporter.get_finished_spans()
        assert len(spans) == 1

    @pytest.mark.asyncio
    async def test_operation_name_invoke_agent(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        async with span_llm_call(tracer, agent_name="telemetry-agent"):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes is not None
        assert span.attributes.get("gen_ai.operation.name") == "invoke_agent"

    @pytest.mark.asyncio
    async def test_agent_name_attribute(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        async with span_llm_call(tracer, agent_name="my-agent"):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes is not None
        assert span.attributes.get("gen_ai.agent.name") == "my-agent"

    @pytest.mark.asyncio
    async def test_span_is_ended(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        async with span_llm_call(tracer, agent_name="a"):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.end_time is not None and span.end_time > 0

    @pytest.mark.asyncio
    async def test_caller_can_set_extra_attributes(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        async with span_llm_call(tracer, agent_name="a") as span:
            span.set_attribute("gen_ai.tokens", 42)
        finished = exporter.get_finished_spans()[0]
        assert finished.attributes is not None
        assert finished.attributes.get("gen_ai.tokens") == 42


# ---------------------------------------------------------------------------
# span_tool_call tests
# ---------------------------------------------------------------------------


class TestSpanToolCall:
    @pytest.mark.asyncio
    async def test_produces_one_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        async with span_tool_call(tracer, agent_name="a", tool_name="doc_search"):
            pass
        assert len(exporter.get_finished_spans()) == 1

    @pytest.mark.asyncio
    async def test_operation_name_execute_tool(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        async with span_tool_call(tracer, agent_name="a", tool_name="doc_search"):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes is not None
        assert span.attributes.get("gen_ai.operation.name") == "execute_tool"

    @pytest.mark.asyncio
    async def test_agent_name_attribute(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        async with span_tool_call(tracer, agent_name="my-agent", tool_name="x"):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes is not None
        assert span.attributes.get("gen_ai.agent.name") == "my-agent"

    @pytest.mark.asyncio
    async def test_tool_name_attribute(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        async with span_tool_call(tracer, agent_name="a", tool_name="verify_citation"):
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes is not None
        assert span.attributes.get("gen_ai.tool.name") == "verify_citation"


# ---------------------------------------------------------------------------
# AgentRunner + tracer integration (AGT-7)
# ---------------------------------------------------------------------------


class TestAgentRunnerTelemetry:
    @pytest.mark.asyncio
    async def test_single_llm_call_produces_one_llm_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        spec = _spec()
        client = FakeGatewayClient([_stop()])
        runner = AgentRunner(spec=spec, client=client, tracer=tracer)
        await runner.run(user_message="Q")
        llm_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "invoke_agent"
        ]
        assert len(llm_spans) == 1

    @pytest.mark.asyncio
    async def test_multi_turn_produces_multiple_llm_spans(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        spec = _spec()
        client = FakeGatewayClient(
            [
                LLMResponse(content="t", finish_reason="tool_use", input_tokens=5, output_tokens=5),
                LLMResponse(content="t", finish_reason="tool_use", input_tokens=5, output_tokens=5),
                _stop(),
            ]
        )
        runner = AgentRunner(spec=spec, client=client, tracer=tracer)
        result = await runner.run(user_message="Q")
        llm_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "invoke_agent"
        ]
        assert len(llm_spans) == result.iterations

    @pytest.mark.asyncio
    async def test_tool_use_produces_tool_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        spec = _spec(["doc_search"])
        registry = ToolRegistry(spec)
        client = FakeGatewayClient([_tool_use("doc_search"), _stop()])
        runner = AgentRunner(spec=spec, client=client, tracer=tracer, registry=registry)
        await runner.run(user_message="Q")
        tool_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "execute_tool"
        ]
        assert len(tool_spans) == 1
        assert tool_spans[0].attributes is not None
        assert tool_spans[0].attributes.get("gen_ai.tool.name") == "doc_search"

    @pytest.mark.asyncio
    async def test_all_spans_are_finished(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        spec = _spec()
        client = FakeGatewayClient([_stop()])
        runner = AgentRunner(spec=spec, client=client, tracer=tracer)
        await runner.run(user_message="Q")
        for span in exporter.get_finished_spans():
            assert span.end_time is not None and span.end_time > 0

    @pytest.mark.asyncio
    async def test_llm_span_has_agent_name(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = get_tracer(provider)
        spec = _spec()
        client = FakeGatewayClient([_stop()])
        runner = AgentRunner(spec=spec, client=client, tracer=tracer)
        await runner.run(user_message="Q")
        llm_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes and s.attributes.get("gen_ai.operation.name") == "invoke_agent"
        ]
        assert llm_spans[0].attributes is not None
        assert llm_spans[0].attributes.get("gen_ai.agent.name") == "telemetry-agent"

    @pytest.mark.asyncio
    async def test_coherent_trace_multi_step(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """A two-step run (tool then stop) produces LLM + tool + LLM spans in order."""
        tracer = get_tracer(provider)
        spec = _spec(["doc_search"])
        registry = ToolRegistry(spec)
        client = FakeGatewayClient([_tool_use("doc_search"), _stop()])
        runner = AgentRunner(spec=spec, client=client, tracer=tracer, registry=registry)
        await runner.run(user_message="Q")
        finished = exporter.get_finished_spans()
        ops = [s.attributes.get("gen_ai.operation.name") for s in finished if s.attributes]
        assert "invoke_agent" in ops
        assert "execute_tool" in ops
        # Total 3 spans: llm_call(tool_use) + tool_call + llm_call(stop)
        assert len(finished) == 3

    @pytest.mark.asyncio
    async def test_no_tracer_produces_no_spans(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Without an injected tracer the exporter receives nothing."""
        spec = _spec()
        client = FakeGatewayClient([_stop()])
        runner = AgentRunner(spec=spec, client=client)  # no tracer
        await runner.run(user_message="Q")
        assert len(exporter.get_finished_spans()) == 0
