"""AGT-16 — FastAPI trigger surface tests.

The gateway client is faked (no network) and persistence uses an in-memory
SQLite engine (no Postgres). Exercises both the persistence-on and
persistence-off paths plus error cases.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import gateway_client_dep, session_factory_dep
from app.loop.gateway_client import LLMRequest, LLMResponse
from app.main import app
from app.persistence.session import create_all

# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class _FakeGateway:
    """GatewayClient stub returning a fixed response (no network)."""

    def __init__(self, response: LLMResponse) -> None:
        self._response = response

    async def chat(self, request: LLMRequest) -> LLMResponse:
        return self._response


_CITED = LLMResponse(
    content="Article 6(1)(a) requires explicit consent [src:reg-042].",
    finish_reason="stop",
    input_tokens=180,
    output_tokens=140,
)

_TOOL_LOOP = LLMResponse(
    content="Let me search for more information.",
    finish_reason="tool_use",
    input_tokens=40,
    output_tokens=20,
)


def _sqlite_factory() -> sessionmaker:  # type: ignore[type-arg]
    """In-memory SQLite session factory (single shared connection)."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_healthz() -> None:
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_run_without_persistence() -> None:
    app.dependency_overrides[gateway_client_dep] = lambda: _FakeGateway(_CITED)
    app.dependency_overrides[session_factory_dep] = lambda: None

    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/runs",
            json={"agent_name": "regdoc-qa", "user_message": "What governs consent?"},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"] is None
    assert data["status"] == "completed"
    assert data["iterations"] == 1
    assert data["tokens_used"] == 320
    assert "[src:reg-042]" in data["content"]
    assert data["agent_version"] == "1.0.0"


def test_create_run_with_persistence_then_get() -> None:
    factory = _sqlite_factory()
    app.dependency_overrides[gateway_client_dep] = lambda: _FakeGateway(_CITED)
    app.dependency_overrides[session_factory_dep] = lambda: factory

    with TestClient(app) as client:
        post = client.post(
            "/v1/agent/runs",
            json={"agent_name": "regdoc-qa", "user_message": "What governs consent?"},
        )
        assert post.status_code == 201
        run_id = post.json()["run_id"]
        assert run_id is not None

        get = client.get(f"/v1/agent/runs/{run_id}")

    assert get.status_code == 200
    data = get.json()
    assert data["run_id"] == run_id
    assert data["status"] == "completed"
    assert data["tokens_used"] == 320
    assert data["agent_name"] == "regdoc-qa"
    assert data["ended_at"] is not None
    # One llm_call step persisted (stop on first turn).
    assert len(data["steps"]) == 1
    assert data["steps"][0]["type"] == "llm_call"


def test_unknown_agent_returns_404() -> None:
    app.dependency_overrides[gateway_client_dep] = lambda: _FakeGateway(_CITED)
    app.dependency_overrides[session_factory_dep] = lambda: None

    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/runs",
            json={"agent_name": "does-not-exist", "user_message": "hi"},
        )
    assert resp.status_code == 404
    assert "unknown agent" in resp.json()["detail"]


def test_get_run_not_found() -> None:
    factory = _sqlite_factory()
    app.dependency_overrides[session_factory_dep] = lambda: factory

    with TestClient(app) as client:
        resp = client.get("/v1/agent/runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_get_run_invalid_uuid() -> None:
    factory = _sqlite_factory()
    app.dependency_overrides[session_factory_dep] = lambda: factory

    with TestClient(app) as client:
        resp = client.get("/v1/agent/runs/not-a-uuid")
    assert resp.status_code == 400


def test_get_run_persistence_disabled_returns_503() -> None:
    app.dependency_overrides[session_factory_dep] = lambda: None

    with TestClient(app) as client:
        resp = client.get("/v1/agent/runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 503


def test_cap_breach_persists_capped_status() -> None:
    factory = _sqlite_factory()
    # Always returns tool_use → never stops → trips max_iterations (8).
    app.dependency_overrides[gateway_client_dep] = lambda: _FakeGateway(_TOOL_LOOP)
    app.dependency_overrides[session_factory_dep] = lambda: factory

    with TestClient(app) as client:
        post = client.post(
            "/v1/agent/runs",
            json={"agent_name": "regdoc-qa", "user_message": "loop forever"},
        )
        assert post.status_code == 201
        assert post.json()["status"] == "capped"
        run_id = post.json()["run_id"]

        get = client.get(f"/v1/agent/runs/{run_id}")
    assert get.json()["status"] == "capped"


def test_upstream_429_maps_to_429() -> None:
    class _Throttled:
        async def chat(self, request: LLMRequest) -> LLMResponse:
            req = httpx.Request("POST", "http://gw:8000/v1/chat/completions")
            resp = httpx.Response(429, request=req)
            raise httpx.HTTPStatusError("429", request=req, response=resp)

    app.dependency_overrides[gateway_client_dep] = lambda: _Throttled()
    app.dependency_overrides[session_factory_dep] = lambda: None

    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/runs",
            json={"agent_name": "regdoc-qa", "user_message": "hi"},
        )
    assert resp.status_code == 429
    assert "rate-limited" in resp.json()["detail"]


def test_upstream_500_maps_to_502() -> None:
    class _Broken:
        async def chat(self, request: LLMRequest) -> LLMResponse:
            req = httpx.Request("POST", "http://gw:8000/v1/chat/completions")
            resp = httpx.Response(500, request=req)
            raise httpx.HTTPStatusError("500", request=req, response=resp)

    app.dependency_overrides[gateway_client_dep] = lambda: _Broken()
    app.dependency_overrides[session_factory_dep] = lambda: None

    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/runs",
            json={"agent_name": "regdoc-qa", "user_message": "hi"},
        )
    assert resp.status_code == 502
