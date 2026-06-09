"""AGT-16 — FastAPI trigger-surface tests (fully offline: Mock + SQLite).

Exercises the documented contract end-to-end through `TestClient`, with no real
network, API keys, or Postgres:

(a) POST /v1/agent/runs returns 202 + run_id + status.
(b) GET /v1/agent/runs/{id} polls to a terminal status and returns the answer.
(c) Unknown run_id → 404.
(d) Caps + tool-whitelist are enforced end-to-end: a non-whitelisted tool call
    drives the run to a terminal ``failed`` status (the AGT-4 guard fires
    through the real `AgentRunner`), and an over-cap loop is reported ``failed``.

The default wiring (`MockGatewayClient` + shared in-memory SQLite engine) is
used as-is for the happy paths. Scenarios that need a scripted gateway override
`get_gateway_client` via FastAPI dependency overrides; the in-memory engine is
reset per test via `get_engine.cache_clear()` so runs don't leak across tests.

TestClient note: FastAPI runs `BackgroundTasks` synchronously *after* the
response is returned, so by the time POST returns 202 the background run has
already completed against the shared engine — a subsequent GET observes the
terminal state deterministically (no polling loop / sleeps needed).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_engine, get_gateway_client, get_spec_loader
from app.loop.gateway_client import LLMRequest, LLMResponse
from app.main import create_app

# ---------------------------------------------------------------------------
# Scripted gateway fakes (mirror tests/test_integration.py conventions)
# ---------------------------------------------------------------------------


class _ScriptedGatewayClient:
    """Returns one scripted ``LLMResponse`` per ``chat`` call, in order."""

    def __init__(self, turns: list[LLMResponse]) -> None:
        self._turns = turns
        self._idx = 0

    async def chat(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
        if self._idx >= len(self._turns):
            raise IndexError(f"_ScriptedGatewayClient exhausted after {self._idx} calls")
        resp = self._turns[self._idx]
        self._idx += 1
        return resp


def _stop(content: str = "Done.", *, tokens: int = 40) -> LLMResponse:
    half = tokens // 2
    return LLMResponse(content=content, finish_reason="stop", input_tokens=half, output_tokens=half)


def _tool_use(tool_name: str, *, tokens: int = 20) -> LLMResponse:
    half = tokens // 2
    return LLMResponse(
        content=f"tool:{tool_name}",
        finish_reason="tool_use",
        input_tokens=half,
        output_tokens=half,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """A `TestClient` over a fresh app with an isolated in-memory engine.

    `get_engine` / `get_spec_loader` are `lru_cache`d on the module, so the
    caches are cleared before and after each test to guarantee a clean DB and no
    cross-test leakage. The default `regdoc-qa.yaml` fixture spec is used.
    """
    get_engine.cache_clear()
    get_spec_loader.cache_clear()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    get_engine.cache_clear()
    get_spec_loader.cache_clear()


# ===========================================================================
# (a) POST returns 202 + run_id + status
# ===========================================================================


def test_post_run_returns_202(client: TestClient) -> None:
    resp = client.post(
        "/v1/agent/runs",
        json={"agent": "regdoc-qa", "user_message": "What is clause 4.2?"},
    )
    assert resp.status_code == 202


def test_post_run_returns_run_id_and_status(client: TestClient) -> None:
    resp = client.post(
        "/v1/agent/runs",
        json={"agent": "regdoc-qa", "user_message": "What is clause 4.2?"},
    )
    body = resp.json()
    assert "run_id" in body and body["run_id"]
    assert body["status"] == "running"
    # run_id is a UUID string
    import uuid

    uuid.UUID(body["run_id"])  # raises if malformed


def test_post_unknown_agent_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/v1/agent/runs",
        json={"agent": "does-not-exist", "user_message": "hi"},
    )
    assert resp.status_code == 422


def test_post_missing_field_returns_422(client: TestClient) -> None:
    # Pydantic request validation: missing user_message.
    resp = client.post("/v1/agent/runs", json={"agent": "regdoc-qa"})
    assert resp.status_code == 422


# ===========================================================================
# (b) GET polls to a terminal status and returns the result
# ===========================================================================


def test_get_run_polls_to_succeeded_with_result(client: TestClient) -> None:
    """Default Mock client → run completes; poll returns succeeded + the answer."""
    start = client.post(
        "/v1/agent/runs",
        json={"agent": "regdoc-qa", "user_message": "Record retention rules?"},
    )
    run_id = start.json()["run_id"]

    poll = client.get(f"/v1/agent/runs/{run_id}")
    assert poll.status_code == 200
    body = poll.json()
    assert body["run_id"] == run_id
    assert body["status"] == "succeeded"
    # The MockGatewayClient echoes the user message into the final answer.
    assert body["result"] is not None
    assert "Record retention rules?" in body["result"]
    assert body["error"] is None


def test_get_run_result_is_final_llm_content(client: TestClient) -> None:
    """A scripted multi-turn run surfaces the *last* llm_call content as result."""
    app = create_app()
    app.dependency_overrides[get_gateway_client] = lambda: _ScriptedGatewayClient(
        [_tool_use("doc_search"), _stop("FINAL ANSWER with [source:doc-001].")]
    )
    get_engine.cache_clear()
    get_spec_loader.cache_clear()
    with TestClient(app) as c:
        run_id = c.post(
            "/v1/agent/runs",
            json={"agent": "regdoc-qa", "user_message": "Q"},
        ).json()["run_id"]
        body = c.get(f"/v1/agent/runs/{run_id}").json()
    get_engine.cache_clear()
    get_spec_loader.cache_clear()

    assert body["status"] == "succeeded"
    assert body["result"] == "FINAL ANSWER with [source:doc-001]."
    assert body["error"] is None


# ===========================================================================
# (c) Unknown run_id → 404
# ===========================================================================


def test_get_unknown_run_id_returns_404(client: TestClient) -> None:
    import uuid

    resp = client.get(f"/v1/agent/runs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_malformed_run_id_returns_404(client: TestClient) -> None:
    resp = client.get("/v1/agent/runs/not-a-uuid")
    assert resp.status_code == 404


# ===========================================================================
# (d) Caps + tool whitelist enforced end-to-end
# ===========================================================================


def _run_with_scripted_client(turns: list[LLMResponse], *, user_message: str) -> dict[str, object]:
    """Start + poll a run with a scripted gateway client; return the poll body."""
    app = create_app()
    app.dependency_overrides[get_gateway_client] = lambda: _ScriptedGatewayClient(turns)
    get_engine.cache_clear()
    get_spec_loader.cache_clear()
    try:
        with TestClient(app) as c:
            run_id = c.post(
                "/v1/agent/runs",
                json={"agent": "regdoc-qa", "user_message": user_message},
            ).json()["run_id"]
            return c.get(f"/v1/agent/runs/{run_id}").json()
    finally:
        get_engine.cache_clear()
        get_spec_loader.cache_clear()


def test_whitelist_violation_surfaces_failed() -> None:
    """A non-whitelisted tool call (AGT-4) drives the run to terminal 'failed'.

    regdoc-qa.yaml whitelists only doc_search + verify_citation, so a
    ``execute_shell`` tool call raises ToolNotAllowedError inside the runner; the
    service records the run as failed and the poll surfaces it.
    """
    body = _run_with_scripted_client(
        [_tool_use("execute_shell"), _stop()],
        user_message="do something forbidden",
    )
    assert body["status"] == "failed"
    assert body["result"] is None
    assert body["error"] is not None


def test_whitelisted_tool_then_stop_succeeds() -> None:
    """A whitelisted tool call passes the guard and the run completes."""
    body = _run_with_scripted_client(
        [_tool_use("doc_search"), _stop("cited [source:doc-001]")],
        user_message="ok",
    )
    assert body["status"] == "succeeded"
    assert body["result"] == "cited [source:doc-001]"


def test_iteration_cap_surfaces_failed() -> None:
    """An infinite tool loop breaches max_iterations (8) → terminal 'failed'.

    The CapBreachError is raised inside the runner, which persists status
    'capped'; the API projects 'capped' → 'failed' and reports the cap error.
    """
    body = _run_with_scripted_client(
        [_tool_use("doc_search")] * 50,
        user_message="loop forever",
    )
    assert body["status"] == "failed"
    error = body["error"]
    assert isinstance(error, str)
    assert "cap" in error.lower()
