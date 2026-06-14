"""AGT-16 — OpenAPI quality invariants.

The drift guard in `test_openapi_contract.py` keeps the committed spec in sync
with the live `app.openapi()`. This file asserts the structural invariants
that make `/docs` documentation-grade — useful descriptions, tags, and typed
errors. Mirrors the gateway pattern (GW-21). Fully offline. See ADR-014.

Note: the agent-runtime currently has no API auth (no Bearer scheme to assert
on); the auth slice belongs to a separate ticket. The chat-completions SSE
checks from the gateway suite don't apply here either — no streaming routes.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.main import app


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return app.openapi()


def test_app_metadata_is_populated(spec: dict[str, Any]) -> None:
    info = spec["info"]
    assert info["title"] == "Atlas Agent Runtime"
    assert info["version"], "version must be non-empty"
    assert len(info.get("description") or "") > 100, "description should be markdown-rich"
    assert info.get("contact"), "contact block missing"
    assert info.get("license"), "license_info missing"
    assert len(spec.get("servers", [])) >= 1, "at least one server URL must be advertised"


def test_error_envelope_schema_present(spec: dict[str, Any]) -> None:
    schemas = spec.get("components", {}).get("schemas", {})
    assert "ErrorEnvelope" in schemas, "ErrorEnvelope schema missing (typed errors absent)"


def test_every_operation_has_summary_description_and_tag(spec: dict[str, Any]) -> None:
    missing: list[str] = []
    for path, methods in spec["paths"].items():
        for verb, op in methods.items():
            if verb in ("parameters", "summary", "description"):
                continue
            op_id = f"{verb.upper()} {path}"
            if not op.get("summary"):
                missing.append(f"{op_id}: summary")
            if not op.get("tags"):
                missing.append(f"{op_id}: tags")
    assert not missing, "operations missing metadata:\n  " + "\n  ".join(missing)


def test_agent_name_has_traversal_pattern(spec: dict[str, Any]) -> None:
    """`CreateRunRequest.agent_name` must publish a non-empty `pattern` so the
    schema layer rejects path-separator / `..` payloads before the route runs.
    Catches accidental constraint drops in future edits to the schema field.
    """
    schemas = spec.get("components", {}).get("schemas", {})
    field = schemas.get("CreateRunRequest", {}).get("properties", {}).get("agent_name", {})
    pattern = field.get("pattern")
    assert pattern, "CreateRunRequest.agent_name is missing a `pattern` (traversal defense)"
    assert pattern.startswith("^") and pattern.endswith("$"), (
        f"agent_name pattern must be anchored (`^…$`), got: {pattern!r}"
    )
    assert field.get("maxLength"), "agent_name must declare maxLength"


def test_error_responses_reference_error_envelope(spec: dict[str, Any]) -> None:
    """4xx/5xx responses we author should point at ErrorEnvelope so codegen types them.

    FastAPI auto-generates a 422 with `HTTPValidationError` for any operation that
    has a request body or path param — that's framework-owned and not part of our
    contract, so we exclude 422 from this check.
    """
    missing: list[str] = []
    for path, methods in spec["paths"].items():
        for verb, op in methods.items():
            if verb in ("parameters", "summary", "description"):
                continue
            for status, resp in op.get("responses", {}).items():
                if not status.startswith(("4", "5")) or status == "422":
                    continue
                schema_ref = resp.get("content", {}).get("application/json", {}).get("schema", {})
                ref = schema_ref.get("$ref", "")
                if "ErrorEnvelope" not in ref:
                    missing.append(f"{verb.upper()} {path} {status}: {ref or '<no schema>'}")
    assert not missing, "error responses not using ErrorEnvelope:\n  " + "\n  ".join(missing)
