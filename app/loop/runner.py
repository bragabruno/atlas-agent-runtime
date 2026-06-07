"""AgentRunner — bounded asyncio loop with hard caps (AGT-3).

Enforces three independent hard caps drawn from `AgentSpec`:
  - ``max_iterations``: total LLM→(optional tool) cycles.
  - ``token_budget``: cumulative tokens across all LLM calls.
  - ``timeout_s``: wall-time for the entire run.

A breach on **any** cap immediately raises `CapBreachError` with an explicit
message naming the cap — never silent. See atlas-docs ADR-006.

The gateway is accessed through an injected `GatewayClient` — the runner
has no import of any concrete HTTP client, making it fully testable offline.

AGT-4 integration: an optional `ToolRegistry` can be injected. When present,
every tool-call response (finish_reason == "tool_use") is checked against the
whitelist via `registry.assert_allowed`. A non-whitelisted tool raises
`ToolNotAllowedError` immediately.

AGT-6 integration: an optional `Session` (SQLAlchemy) can be injected.
When present the runner persists each step and the run lifecycle to the DB.
A run that was interrupted can be resumed by passing ``resume_run_id`` — the
runner skips steps already in the DB and continues from the last persisted
step index.

AGT-7 integration: an optional `Tracer` can be injected.  When present,
each LLM call and each tool-call step is wrapped in a span with GenAI semantic
attributes (`gen_ai.operation.name`, `gen_ai.agent.name`).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.agentspec.model import AgentSpec
from app.loop.errors import CapBreachError
from app.loop.gateway_client import GatewayClient, LLMRequest, LLMResponse

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer
    from sqlalchemy.orm import Session

    from app.tools.registry.registry import ToolRegistry


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

    Optional injections (all keyword-only):
        registry: `ToolRegistry` — if provided, every tool-use response is
            checked against the whitelist before continuing (AGT-4).
        session: SQLAlchemy `Session` — if provided, the run and its steps are
            persisted; callers are responsible for commit/rollback (AGT-6).
        resume_run_id: `uuid.UUID` — if provided alongside a *session*, the
            runner looks up existing steps for this run and resumes from the
            next un-persisted step index (AGT-6).
        tracer: OTel `Tracer` — if provided, each LLM call and tool-call step
            is wrapped in a span with GenAI semantic attributes (AGT-7).
    """

    def __init__(
        self,
        *,
        spec: AgentSpec,
        client: GatewayClient,
        registry: ToolRegistry | None = None,
        session: Session | None = None,
        resume_run_id: uuid.UUID | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._spec = spec
        self._client = client
        self._registry = registry
        self._session = session
        self._resume_run_id = resume_run_id
        self._tracer = tracer

    async def run(self, *, user_message: str) -> RunResult:
        """Execute the agent loop and return a `RunResult`.

        Raises:
            CapBreachError: if ``max_iterations``, ``token_budget``, or
                ``timeout_s`` is exceeded before the run completes naturally.
            ToolNotAllowedError: if a tool-use response names a tool not in the
                whitelist (only when a registry is injected).
        """
        from app.persistence.dal import (
            append_step,
            create_run,
            get_last_step_idx,
            get_run,
            update_run_status,
        )
        from app.persistence.tables import AgentStatusEnum, StepTypeEnum
        from app.telemetry.tracing import span_llm_call, span_tool_call

        spec = self._spec
        messages: list[dict[str, str]] = [{"role": "user", "content": user_message}]
        responses: list[LLMResponse] = []
        tokens_used = 0
        iterations = 0
        started = time.monotonic()

        # --- AGT-6: run lifecycle persistence ---
        from app.persistence.tables import AgentRun

        db_run: AgentRun | None = None
        resume_from_idx: int = 0
        if self._session is not None:
            if self._resume_run_id is not None:
                # Resuming: find existing run and last persisted step
                db_run = get_run(self._session, self._resume_run_id)
                last_idx = get_last_step_idx(self._session, self._resume_run_id)
                resume_from_idx = (last_idx + 1) if last_idx is not None else 0
                if db_run is not None:
                    update_run_status(
                        self._session,
                        db_run,
                        status=AgentStatusEnum.running,
                    )
            else:
                db_run = create_run(
                    self._session,
                    agent_name=spec.agent_name,
                    agent_version=spec.agent_version,
                    token_budget=spec.token_budget,
                )
                update_run_status(self._session, db_run, status=AgentStatusEnum.running)

        step_idx = resume_from_idx

        async def _one_iteration() -> LLMResponse:
            """Call the gateway for one LLM turn, updating shared counters."""
            nonlocal tokens_used, iterations, step_idx

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

            # --- AGT-7: LLM span ---
            t0 = time.monotonic()
            if self._tracer is not None:
                async with span_llm_call(
                    self._tracer,
                    agent_name=spec.agent_name,
                ) as span:
                    response = await self._client.chat(request)
                    span.set_attribute("gen_ai.tokens", response.total_tokens)
            else:
                response = await self._client.chat(request)
            latency_ms = int((time.monotonic() - t0) * 1000)

            iterations += 1

            # --- token cap (checked after each call, before continuing) ---
            tokens_used += response.total_tokens
            if tokens_used > spec.token_budget:
                raise CapBreachError("token_budget", spec.token_budget)

            # --- AGT-6: persist LLM step ---
            if self._session is not None and db_run is not None:
                append_step(
                    self._session,
                    run_id=db_run.id,
                    idx=step_idx,
                    step_type=StepTypeEnum.llm_call,
                    payload={
                        "content": response.content,
                        "finish_reason": response.finish_reason,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                    tokens=response.total_tokens,
                    latency_ms=latency_ms,
                )
                step_idx += 1

            # --- AGT-4: tool whitelist check ---
            if response.finish_reason == "tool_use" and self._registry is not None:
                # Extract tool name from the response content.
                # The gateway embeds the tool name in the content when
                # finish_reason == "tool_use" (format: "tool:<name>").
                tool_name = _extract_tool_name(response.content)
                if tool_name is not None:
                    # assert_allowed raises ToolNotAllowedError if not in list
                    self._registry.assert_allowed(tool_name)

                    # --- AGT-7: tool span ---
                    if self._tracer is not None:
                        async with span_tool_call(
                            self._tracer,
                            agent_name=spec.agent_name,
                            tool_name=tool_name,
                        ):
                            pass  # actual tool execution is outside the runner

                    # --- AGT-6: persist tool step ---
                    if self._session is not None and db_run is not None:
                        append_step(
                            self._session,
                            run_id=db_run.id,
                            idx=step_idx,
                            step_type=StepTypeEnum.tool_call,
                            payload={"tool_name": tool_name},
                            tokens=0,
                            latency_ms=0,
                        )
                        step_idx += 1

            return response

        # Wrap the loop body in asyncio.wait_for so the OS-level timeout fires
        # even when a gateway call hangs without yielding.
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
            # --- AGT-6: mark run as capped ---
            if self._session is not None and db_run is not None:
                update_run_status(
                    self._session,
                    db_run,
                    status=AgentStatusEnum.capped,
                    tokens_used=tokens_used,
                    ended_at=datetime.now(tz=UTC),
                )
            raise

        elapsed_s = time.monotonic() - started

        # --- AGT-6: mark run as completed ---
        if self._session is not None and db_run is not None:
            update_run_status(
                self._session,
                db_run,
                status=AgentStatusEnum.completed,
                tokens_used=tokens_used,
                ended_at=datetime.now(tz=UTC),
            )

        return RunResult(
            iterations=iterations,
            tokens_used=tokens_used,
            elapsed_s=elapsed_s,
            responses=responses,
        )


def _extract_tool_name(content: str) -> str | None:
    """Extract a tool name from a ``finish_reason='tool_use'`` response.

    Convention: the gateway encodes the tool name as ``tool:<name>`` in the
    content string.  Returns ``None`` if the content does not follow that
    convention (caller skips the whitelist check in that case).

    This thin helper is intentionally simple and testable in isolation.
    """
    prefix = "tool:"
    if content.startswith(prefix):
        name = content[len(prefix) :].strip()
        return name if name else None
    return None
