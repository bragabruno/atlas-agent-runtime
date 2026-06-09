"""RunService — orchestration for the agent-run trigger surface (AGT-16).

Bridges the thin HTTP layer (`app.api.v1.runs`) to the existing agent-runtime
internals — `AgentRunner` (AGT-3/4/5), the persistence DAL (AGT-6), and
`AgentSpec` (AGT-1) — without inventing a second store or reimplementing the
loop. The contract (ADR-020):

- ``start_run`` resolves the agent's `AgentSpec`, creates the *one* canonical
  ``agent_runs`` row via the DAL (capturing its ``run_id``), commits it so a
  concurrent poll can see it, schedules the `AgentRunner` to execute in the
  background, and returns the ``run_id`` immediately — the POST never blocks on
  the LLM loop.
- ``get_run`` reads that same row back and projects it to the API status
  vocabulary; the final answer is the content of the run's last ``llm_call``
  step (already persisted by the runner — no parallel record).

run_id surfacing: the service creates the row first, then hands the runner its
``run_id`` via ``resume_run_id`` so the runner *reuses* the existing row (the
AGT-6 resume path) instead of creating a second one. This is the only seam
touched to surface the id; the runner is otherwise unchanged.

Background execution: the run is launched through an injected ``schedule``
callable (FastAPI ``BackgroundTasks.add_task`` in production). Each background
run gets its **own** SQLAlchemy session from the injected ``session_factory``
and commits on its own, because the request-scoped session is closed once the
202 response is sent. Cap breaches and tool-whitelist violations propagate
through the runner exactly as in a direct call; the background wrapper catches
them, marks the run ``failed``, and never swallows them silently — the failure
is recorded on the run row and surfaced on the next poll.

Offline default: the injected gateway ``client`` is a Mock/Fake and the
``session_factory`` is an in-memory SQLite engine (see `app.api.deps`), so the
default path needs no Postgres, no network, and no API keys.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from app.agentspec.model import AgentSpec, load_spec
from app.api.schemas import AgentRunResponse, RunStatus
from app.loop.gateway_client import GatewayClient
from app.loop.runner import AgentRunner
from app.persistence.dal import create_run, get_run, list_steps, update_run_status
from app.persistence.tables import AgentRun, AgentStatusEnum, StepTypeEnum
from app.tools.registry import ToolRegistry
from app.tools.sanitize.sanitizer import ToolSanitizer

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

#: Callable that runs *fn* (no args) at some later point — FastAPI's
#: ``BackgroundTasks.add_task`` in production, or a synchronous shim in tests.
ScheduleFn = Callable[[Callable[[], None]], None]

#: Callable that yields a fresh, independent SQLAlchemy ``Session``. Each
#: background run uses its own session because the request session is gone by
#: the time the run executes.
SessionFactory = Callable[[], "Session"]

#: Callable that resolves an agent name to a validated ``AgentSpec``.
SpecLoader = Callable[[str], AgentSpec]


class AgentNotFoundError(Exception):
    """Raised when no `AgentSpec` can be resolved for the requested agent name.

    The controller maps this to HTTP 422 — a client asked to run an agent that
    does not exist in the configured agents directory.
    """

    def __init__(self, agent: str) -> None:
        self.agent = agent
        super().__init__(f"AgentNotFoundError: no agent spec found for {agent!r}")


class RunNotFoundError(Exception):
    """Raised when a polled ``run_id`` has no persisted run row.

    The controller maps this to HTTP 404.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"RunNotFoundError: no run found for id {run_id!r}")


def make_dir_spec_loader(agents_dir: Path) -> SpecLoader:
    """Build a `SpecLoader` that resolves ``<agents_dir>/<agent>.yaml`` specs.

    Fails fast with `AgentNotFoundError` when the file is absent so a bad agent
    name surfaces as an explicit 422 rather than a stack trace.
    """

    def _load(agent: str) -> AgentSpec:
        path = agents_dir / f"{agent}.yaml"
        if not path.is_file():
            raise AgentNotFoundError(agent)
        return load_spec(path)

    return _load


#: Persisted status → API status projection (single source of the mapping).
_STATUS_PROJECTION = {
    AgentStatusEnum.created: RunStatus.running,
    AgentStatusEnum.running: RunStatus.running,
    AgentStatusEnum.completed: RunStatus.succeeded,
    AgentStatusEnum.failed: RunStatus.failed,
    AgentStatusEnum.capped: RunStatus.failed,
}


class RunService:
    """Start and poll agent runs, reusing `AgentRunner` and the AGT-6 DAL."""

    def __init__(
        self,
        *,
        client: GatewayClient,
        session_factory: SessionFactory,
        spec_loader: SpecLoader,
        schedule: ScheduleFn,
    ) -> None:
        self._client = client
        self._session_factory = session_factory
        self._spec_loader = spec_loader
        self._schedule = schedule

    def start_run(self, *, agent: str, user_message: str) -> AgentRunResponse:
        """Create the run row, schedule execution, and return its id immediately.

        Raises:
            AgentNotFoundError: if *agent* resolves to no spec (controller → 422).
        """
        spec = self._spec_loader(agent)  # raises AgentNotFoundError on miss

        with self._unit_of_work() as session:
            run = create_run(
                session,
                agent_name=spec.agent_name,
                agent_version=spec.agent_version,
                token_budget=spec.token_budget,
            )
            run_id = run.id  # captured before commit closes the session

        self._schedule(lambda: self._execute(run_id=run_id, spec=spec, user_message=user_message))

        return AgentRunResponse(run_id=str(run_id), status=RunStatus.running)

    def get_run(self, run_id: str) -> AgentRunResponse:
        """Poll a run by id and project it to the API response.

        Raises:
            RunNotFoundError: if no row exists for *run_id* (controller → 404).
        """
        parsed = _parse_run_id(run_id)
        if parsed is None:
            raise RunNotFoundError(run_id)

        with self._unit_of_work() as session:
            run = get_run(session, parsed)
            if run is None:
                raise RunNotFoundError(run_id)
            status = _STATUS_PROJECTION[run.status]
            result = _final_answer(session, run) if status is RunStatus.succeeded else None
            error = _failure_reason(run) if status is RunStatus.failed else None

        return AgentRunResponse(run_id=run_id, status=status, result=result, error=error)

    # -- internals ----------------------------------------------------------

    def _execute(self, *, run_id: uuid.UUID, spec: AgentSpec, user_message: str) -> None:
        """Run `AgentRunner` against the pre-created row in its own session.

        Reuses the existing run row via ``resume_run_id`` (no second record) and
        re-applies the AGT-4 whitelist + AGT-5 sanitizer so caps and tool rules
        are enforced end-to-end. A cap breach / tool rejection / any error is
        recorded on the run as ``failed`` — never swallowed — then surfaced on
        the next poll.
        """
        registry = ToolRegistry(spec)
        sanitizer = ToolSanitizer()
        with self._unit_of_work() as session:
            runner = AgentRunner(
                spec=spec,
                client=self._client,
                registry=registry,
                sanitizer=sanitizer,
                session=session,
                resume_run_id=run_id,
            )
            try:
                _run_sync(runner, user_message=user_message)
            except Exception:
                # The runner persists 'capped' for cap breaches itself; for any
                # other failure (e.g. tool whitelist rejection) mark it failed so
                # the poll surfaces an explicit terminal state rather than a run
                # stuck in 'running'. Log loudly (never suppress silently); the
                # failure is also recorded on the run row for the poll.
                logger.exception("agent run %s failed in background execution", run_id)
                run = get_run(session, run_id)
                if run is not None and run.status not in (
                    AgentStatusEnum.failed,
                    AgentStatusEnum.capped,
                ):
                    update_run_status(
                        session,
                        run,
                        status=AgentStatusEnum.failed,
                        ended_at=datetime.now(tz=UTC),
                    )

    @contextmanager
    def _unit_of_work(self) -> Iterator[Session]:
        """Yield a fresh session and commit on success / rollback on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()


def _run_sync(runner: AgentRunner, *, user_message: str) -> None:
    """Execute the async runner to completion from a synchronous context.

    The background task runs outside the event loop, so drive the coroutine on
    a dedicated loop. Kept tiny and explicit so the failure path stays loud.
    """
    import asyncio

    asyncio.run(runner.run(user_message=user_message))


def _parse_run_id(run_id: str) -> uuid.UUID | None:
    """Parse *run_id* as a UUID, or ``None`` if malformed (→ 404, not 500)."""
    try:
        return uuid.UUID(run_id)
    except ValueError:
        return None


def _final_answer(session: Session, run: AgentRun) -> str | None:
    """Return the content of the run's last ``llm_call`` step (the final answer).

    Reads the persisted steps the runner already wrote (AGT-6) — no second
    store. Returns ``None`` if no llm_call step or no string content is present.
    """
    llm_steps = [s for s in list_steps(session, run.id) if s.type is StepTypeEnum.llm_call]
    if not llm_steps:
        return None
    content = llm_steps[-1].payload.get("content")
    return content if isinstance(content, str) else None


def _failure_reason(run: AgentRun) -> str:
    """Explicit terminal-failure message derived from the run's status."""
    if run.status is AgentStatusEnum.capped:
        return "run exceeded a hard cap (iterations, tokens, or wall-time)"
    return "run failed"
