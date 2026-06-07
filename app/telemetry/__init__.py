"""OTel span instrumentation for the agent runtime (AGT-7).

Public surface: `get_tracer`, `span_llm_call`, `span_tool_call`.
"""

from __future__ import annotations

from app.telemetry.tracing import get_tracer, span_llm_call, span_tool_call

__all__ = ["get_tracer", "span_llm_call", "span_tool_call"]
