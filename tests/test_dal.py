"""AGT-6 Persistence DAL tests — fully offline using SQLite in-memory.

Covers:
- create_run inserts a row with status='created' then 'running'.
- append_step inserts steps with correct fields.
- get_run retrieves the row by id.
- get_last_step_idx returns None when no steps exist.
- get_last_step_idx returns the highest idx when steps exist.
- list_steps returns steps ordered by idx.
- Steps persisted correctly as LLM-type and tool-type.
- AgentRunner persists steps during a run (integration).
- Interrupted run resumes from the last persisted step index.
- update_run_status changes status and tokens_used / ended_at.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agentspec.model import AgentSpec
from app.loop.gateway_client import LLMRequest, LLMResponse
from app.loop.runner import AgentRunner
from app.persistence.base import Base
from app.persistence.dal import (
    append_step,
    create_run,
    get_last_step_idx,
    get_run,
    list_steps,
    update_run_status,
)
from app.persistence.tables import AgentRun, AgentStatusEnum, StepTypeEnum

# ---------------------------------------------------------------------------
# Fixtures
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
# Helpers
# ---------------------------------------------------------------------------


def _spec() -> AgentSpec:
    return AgentSpec(
        agent_name="test-agent",
        agent_version="1.0.0",
        system_prompt_ref="prompts/test.txt",
        model_alias="atlas-default",
        tool_whitelist=["doc_search"],
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


def _stop(tokens: int = 20) -> LLMResponse:
    half = tokens // 2
    return LLMResponse(
        content="Answer.", finish_reason="stop", input_tokens=half, output_tokens=half
    )


def _tool_use(tool_name: str = "doc_search") -> LLMResponse:
    return LLMResponse(
        content=f"tool:{tool_name}",
        finish_reason="tool_use",
        input_tokens=10,
        output_tokens=10,
    )


# ---------------------------------------------------------------------------
# DAL unit tests
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_create_run_returns_agent_run(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        assert isinstance(run, AgentRun)

    def test_create_run_has_uuid_id(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        assert isinstance(run.id, uuid.UUID)

    def test_create_run_status_created(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        assert run.status == AgentStatusEnum.created

    def test_create_run_token_budget(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=5000)
        assert run.token_budget == 5000

    def test_create_run_tokens_used_zero(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        assert run.tokens_used == 0


class TestUpdateRunStatus:
    def test_update_status(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        update_run_status(session, run, status=AgentStatusEnum.running)
        assert run.status == AgentStatusEnum.running

    def test_update_tokens_used(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        update_run_status(session, run, status=AgentStatusEnum.completed, tokens_used=42)
        assert run.tokens_used == 42

    def test_update_ended_at(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        now = datetime.now(tz=UTC)
        update_run_status(session, run, status=AgentStatusEnum.completed, ended_at=now)
        assert run.ended_at == now


class TestAppendStep:
    def test_append_step_returns_agent_step(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        session.flush()
        step = append_step(
            session,
            run_id=run.id,
            idx=0,
            step_type=StepTypeEnum.llm_call,
            payload={"content": "hi"},
            tokens=10,
            latency_ms=50,
        )
        assert step.idx == 0
        assert step.type == StepTypeEnum.llm_call
        assert step.tokens == 10
        assert step.latency_ms == 50

    def test_append_multiple_steps(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        session.flush()
        for i in range(3):
            append_step(
                session,
                run_id=run.id,
                idx=i,
                step_type=StepTypeEnum.llm_call,
                payload={},
                tokens=5,
                latency_ms=10,
            )
        steps = list_steps(session, run.id)
        assert len(steps) == 3
        assert [s.idx for s in steps] == [0, 1, 2]

    def test_append_tool_step(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        session.flush()
        step = append_step(
            session,
            run_id=run.id,
            idx=0,
            step_type=StepTypeEnum.tool_call,
            payload={"tool_name": "doc_search"},
            tokens=0,
            latency_ms=0,
        )
        assert step.type == StepTypeEnum.tool_call


class TestGetRun:
    def test_get_run_found(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        session.flush()
        fetched = get_run(session, run.id)
        assert fetched is not None
        assert fetched.id == run.id

    def test_get_run_not_found(self, session: Session) -> None:
        result = get_run(session, uuid.uuid4())
        assert result is None


class TestGetLastStepIdx:
    def test_no_steps_returns_none(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        session.flush()
        assert get_last_step_idx(session, run.id) is None

    def test_single_step_returns_zero(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        session.flush()
        append_step(
            session,
            run_id=run.id,
            idx=0,
            step_type=StepTypeEnum.llm_call,
            payload={},
            tokens=0,
            latency_ms=0,
        )
        assert get_last_step_idx(session, run.id) == 0

    def test_multiple_steps_returns_highest_idx(self, session: Session) -> None:
        run = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        session.flush()
        for i in range(5):
            append_step(
                session,
                run_id=run.id,
                idx=i,
                step_type=StepTypeEnum.llm_call,
                payload={},
                tokens=0,
                latency_ms=0,
            )
        assert get_last_step_idx(session, run.id) == 4

    def test_other_run_not_included(self, session: Session) -> None:
        run1 = create_run(session, agent_name="a", agent_version="1.0.0", token_budget=1000)
        run2 = create_run(session, agent_name="b", agent_version="1.0.0", token_budget=1000)
        session.flush()
        for i in range(3):
            append_step(
                session,
                run_id=run1.id,
                idx=i,
                step_type=StepTypeEnum.llm_call,
                payload={},
                tokens=0,
                latency_ms=0,
            )
        # run2 has no steps
        assert get_last_step_idx(session, run2.id) is None


# ---------------------------------------------------------------------------
# AgentRunner + persistence integration (AGT-6)
# ---------------------------------------------------------------------------


class TestAgentRunnerPersistence:
    @pytest.mark.asyncio
    async def test_run_persists_steps(self, session: Session) -> None:
        """A normal run should persist one LLM step."""
        spec = _spec()
        client = FakeGatewayClient([_stop()])
        runner = AgentRunner(spec=spec, client=client, session=session)
        await runner.run(user_message="Q")
        session.flush()
        # There should be 1 run persisted
        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        assert len(run_rows) == 1
        run = run_rows[0]
        assert run.status == AgentStatusEnum.completed
        assert run.tokens_used == 20

        # One LLM step
        steps = list_steps(session, run.id)
        assert len(steps) == 1
        assert steps[0].type == StepTypeEnum.llm_call
        assert steps[0].idx == 0

    @pytest.mark.asyncio
    async def test_multi_turn_persists_all_steps(self, session: Session) -> None:
        """Multiple LLM turns all get persisted as separate steps."""
        spec = _spec()
        # Need finish_reason != stop for first two
        from app.loop.gateway_client import LLMResponse as R

        client2 = FakeGatewayClient(
            [
                R(content="t1", finish_reason="tool_use", input_tokens=5, output_tokens=5),
                R(content="t2", finish_reason="tool_use", input_tokens=5, output_tokens=5),
                R(content="done", finish_reason="stop", input_tokens=5, output_tokens=5),
            ]
        )
        runner = AgentRunner(spec=spec, client=client2, session=session)
        result = await runner.run(user_message="Q")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        run = run_rows[0]
        steps = list_steps(session, run.id)
        # 3 LLM steps (tool_use content without "tool:" prefix → no tool steps)
        assert len(steps) == 3
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_tool_use_persists_tool_step(self, session: Session) -> None:
        """When finish_reason=tool_use with tool:<name> content, a tool step is persisted."""
        from app.tools.registry import ToolRegistry

        spec = _spec()
        registry = ToolRegistry(spec)
        client = FakeGatewayClient([_tool_use("doc_search"), _stop()])
        runner = AgentRunner(spec=spec, client=client, session=session, registry=registry)
        await runner.run(user_message="Q")
        session.flush()

        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        run = run_rows[0]
        steps = list_steps(session, run.id)
        # step 0: llm_call, step 1: tool_call, step 2: llm_call (stop)
        types = [s.type for s in steps]
        assert StepTypeEnum.tool_call in types
        assert StepTypeEnum.llm_call in types

    @pytest.mark.asyncio
    async def test_resume_from_last_step(self, session: Session) -> None:
        """Resuming a run picks up from the step after the last persisted one."""
        spec = _spec()
        # First run: persist 2 steps (tool_use -> stop)
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry(spec)
        client1 = FakeGatewayClient([_tool_use("doc_search"), _stop()])
        runner1 = AgentRunner(spec=spec, client=client1, session=session, registry=registry)
        await runner1.run(user_message="Q")
        session.flush()

        # Grab the run id
        run_rows = list(session.execute(sa.select(AgentRun)).scalars())
        assert len(run_rows) == 1
        run = run_rows[0]
        first_steps = list_steps(session, run.id)
        assert len(first_steps) >= 1

        # Simulate resume: new runner with the same session + resume_run_id
        client2 = FakeGatewayClient([_stop()])
        runner2 = AgentRunner(
            spec=spec,
            client=client2,
            session=session,
            resume_run_id=run.id,
            registry=registry,
        )
        result2 = await runner2.run(user_message="Q2")
        session.flush()

        # After resuming, new steps appended beyond previous last idx
        all_steps = list_steps(session, run.id)
        assert len(all_steps) > len(first_steps)
        assert result2.iterations == 1  # resumed run only did 1 new LLM call
