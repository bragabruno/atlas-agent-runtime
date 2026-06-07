"""OTel span instrumentation (AGT-7).

Provides two async context-manager helpers that wrap one LLM call or one tool
call in an OTel span with the GenAI semantic conventions defined in README.md:

    gen_ai.operation.name ∈ {invoke_agent, execute_tool}
    gen_ai.agent.name      = agent name from AgentSpec

Callers inject a `Tracer` (from `get_tracer`) so the telemetry module itself
has no global state and tests can pass an in-memory `TracerProvider`.

Usage::

    from opentelemetry.sdk.trace import TracerProvider
    provider = TracerProvider()
    tracer = get_tracer(provider)

    async with span_llm_call(tracer, agent_name="regdoc-qa") as span:
        response = await client.chat(request)
        span.set_attribute("gen_ai.tokens", response.total_tokens)

    async with span_tool_call(tracer, agent_name="regdoc-qa", tool_name="doc_search"):
        result = await tool.run(args)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Tracer

_INSTRUMENTATION_SCOPE = "atlas.agent.runtime"

# GenAI semantic convention attribute names (README.md § OTel Instrumentation)
_ATTR_OPERATION = "gen_ai.operation.name"
_ATTR_AGENT_NAME = "gen_ai.agent.name"
_ATTR_TOOL_NAME = "gen_ai.tool.name"


def get_tracer(provider: TracerProvider | None = None) -> Tracer:
    """Return a `Tracer` scoped to the atlas agent runtime.

    When *provider* is ``None`` the global OTel `TracerProvider` is used
    (production path).  Tests inject a `TracerProvider` with an in-memory
    exporter to capture spans without any network I/O.
    """
    if provider is not None:
        return provider.get_tracer(_INSTRUMENTATION_SCOPE)
    return trace.get_tracer(_INSTRUMENTATION_SCOPE)


@asynccontextmanager
async def span_llm_call(
    tracer: Tracer,
    *,
    agent_name: str,
) -> AsyncGenerator[Span, None]:
    """Async context manager that wraps one LLM call in an OTel span.

    Sets::

        gen_ai.operation.name = "invoke_agent"
        gen_ai.agent.name     = <agent_name>

    The caller may set additional attributes on the yielded ``Span``.
    """
    with tracer.start_as_current_span(
        "llm_call",
        attributes={
            _ATTR_OPERATION: "invoke_agent",
            _ATTR_AGENT_NAME: agent_name,
        },
    ) as span:
        yield span


@asynccontextmanager
async def span_tool_call(
    tracer: Tracer,
    *,
    agent_name: str,
    tool_name: str,
) -> AsyncGenerator[Span, None]:
    """Async context manager that wraps one tool call in an OTel span.

    Sets::

        gen_ai.operation.name = "execute_tool"
        gen_ai.agent.name     = <agent_name>
        gen_ai.tool.name      = <tool_name>

    The caller may set additional attributes on the yielded ``Span``.
    """
    with tracer.start_as_current_span(
        "tool_call",
        attributes={
            _ATTR_OPERATION: "execute_tool",
            _ATTR_AGENT_NAME: agent_name,
            _ATTR_TOOL_NAME: tool_name,
        },
    ) as span:
        yield span
