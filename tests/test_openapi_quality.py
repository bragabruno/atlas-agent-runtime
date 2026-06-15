"""AGT-16 + BRA-878 — OpenAPI quality invariants.

The drift guard in `test_openapi_contract.py` keeps the committed spec in sync
with the live `create_app().openapi()`. This file asserts the structural
invariants that make `/docs` documentation-grade — rich metadata, per-operation
summary/description/tag, the bearer security scheme, and the traversal pattern
on the `agent` field. Mirrors the gateway pattern (GW-21). Fully offline.
See ADR-016 / ADR-020.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config import Settings
from app.main import app, create_app


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return app.openapi()


def test_docs_disabled_in_prod() -> None:
    """`/docs`, `/redoc`, `/openapi.json` must be off when environment=prod.
    `app.openapi()` (programmatic) still works — only the HTTP endpoints go away.
    """
    prod_app = create_app(Settings(environment="prod"))
    assert prod_app.docs_url is None
    assert prod_app.redoc_url is None
    assert prod_app.openapi_url is None
    # Programmatic spec is still available — drift + quality tests keep passing.
    assert prod_app.openapi().get("info", {}).get("title") == "Atlas Agent Runtime"


def test_docs_enabled_in_dev_and_stage() -> None:
    """Dev keeps docs for local use; stage keeps them for QA. Only prod gates them."""
    for env in ("dev", "stage"):
        a = create_app(Settings(environment=env))  # type: ignore[arg-type]
        assert a.docs_url == "/docs", f"docs_url disabled for environment={env}"
        assert a.openapi_url == "/openapi.json", f"openapi_url disabled for environment={env}"


def test_app_metadata_is_populated(spec: dict[str, Any]) -> None:
    info = spec["info"]
    assert info["title"] == "Atlas Agent Runtime"
    assert info["version"], "version must be non-empty"
    assert len(info.get("description") or "") > 100, "description should be markdown-rich"
    assert info.get("contact"), "contact block missing"
    assert info.get("license"), "license_info missing"
    assert len(spec.get("servers", [])) >= 1, "at least one server URL must be advertised"


def test_every_operation_has_summary_description_and_tag(spec: dict[str, Any]) -> None:
    missing: list[str] = []
    for path, methods in spec["paths"].items():
        for verb, op in methods.items():
            if verb in ("parameters", "summary", "description"):
                continue
            op_id = f"{verb.upper()} {path}"
            if not op.get("summary"):
                missing.append(f"{op_id}: summary")
            if not op.get("description"):
                missing.append(f"{op_id}: description")
            if not op.get("tags"):
                missing.append(f"{op_id}: tags")
    assert not missing, "operations missing metadata:\n  " + "\n  ".join(missing)


def test_bearer_security_scheme_published(spec: dict[str, Any]) -> None:
    """BRA-878: the bearer scheme is declared once and the trigger routes require
    it, so Swagger UI renders the Authorize button and codegen knows the routes
    are protected. ``/healthz`` stays open (no security requirement).
    """
    schemes = spec.get("components", {}).get("securitySchemes", {})
    assert "BearerAuth" in schemes, "BearerAuth security scheme missing"
    assert schemes["BearerAuth"].get("type") == "http"
    assert schemes["BearerAuth"].get("scheme") == "bearer"

    paths = spec["paths"]
    for path, verb in (("/v1/agent/runs", "post"), ("/v1/agent/runs/{run_id}", "get")):
        op = paths[path][verb]
        security = op.get("security", [])
        assert any("BearerAuth" in req for req in security), (
            f"{verb.upper()} {path} must require BearerAuth"
        )
    # /healthz stays open.
    assert not paths["/healthz"]["get"].get("security")


def test_agent_field_has_traversal_pattern(spec: dict[str, Any]) -> None:
    """`StartAgentRunRequest.agent` must publish a non-empty anchored `pattern`
    so the schema layer rejects path-separator / `..` payloads before the route
    runs. Catches accidental constraint drops in future edits to the field.
    """
    schemas = spec.get("components", {}).get("schemas", {})
    field = schemas.get("StartAgentRunRequest", {}).get("properties", {}).get("agent", {})
    pattern = field.get("pattern")
    assert pattern, "StartAgentRunRequest.agent is missing a `pattern` (traversal defense)"
    assert pattern.startswith("^") and pattern.endswith("$"), (
        f"agent pattern must be anchored (`^…$`), got: {pattern!r}"
    )
    assert field.get("maxLength"), "agent must declare maxLength"
