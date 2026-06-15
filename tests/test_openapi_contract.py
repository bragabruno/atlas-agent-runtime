"""AGT-16 — OpenAPI drift guard.

Asserts the committed ``openapi.json`` (the source of truth consumers codegen
against) matches the live ``create_app().openapi()``. If a route, schema, or
version changes without regenerating the spec, this test fails and points the
author at ``scripts/export_openapi.py``. Fully offline. Mirrors atlas-gateway's
``tests/test_openapi_contract.py``. See ADR-016 / ADR-020.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.main import create_app

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OPENAPI_PATH = _REPO_ROOT / "openapi.json"


def test_committed_openapi_matches_live_spec() -> None:
    assert _OPENAPI_PATH.is_file(), (
        f"missing {_OPENAPI_PATH}; run `.venv/bin/python scripts/export_openapi.py`"
    )
    committed = json.loads(_OPENAPI_PATH.read_text(encoding="utf-8"))
    live = create_app().openapi()
    assert committed == live, (
        "openapi.json is out of date; regenerate with `.venv/bin/python scripts/export_openapi.py`"
    )


def test_committed_openapi_has_stable_ordering_and_trailing_newline() -> None:
    raw = _OPENAPI_PATH.read_text(encoding="utf-8")
    assert raw.endswith("\n"), "openapi.json must end with a trailing newline"
    expected = (
        json.dumps(create_app().openapi(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    )
    assert raw == expected, (
        "openapi.json is not in stable (sort_keys) form; regenerate with "
        "`.venv/bin/python scripts/export_openapi.py`"
    )


def test_run_trigger_routes_present() -> None:
    """The two documented routes are in the published contract."""
    spec = create_app().openapi()
    paths = spec["paths"]
    assert "/v1/agent/runs" in paths
    assert "post" in paths["/v1/agent/runs"]
    assert "/v1/agent/runs/{run_id}" in paths
    assert "get" in paths["/v1/agent/runs/{run_id}"]
