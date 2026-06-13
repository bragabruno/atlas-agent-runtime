"""AGT-16 — Export the live OpenAPI spec as the committed source of truth.

The FastAPI app is the single source of truth for the agent-runtime's HTTP
trigger surface (ADR-020). This script imports `app.main.app`, renders
`app.openapi()`, and writes it to `openapi.json` at the repo root with stable
ordering (`sort_keys=True`) and a trailing newline so the committed artifact
is diff-friendly and reproducible.

The committed file is enforced against the live spec by
`tests/test_openapi_contract.py` (drift guard): regenerate by running this
script whenever a route, schema, or version changes.

Mirrors the gateway pattern (GW-21) so consumers can codegen typed clients
against either service from a committed artifact. See ADR-014.

Usage:

    .venv/bin/python scripts/export_openapi.py
"""

from __future__ import annotations

import json
from pathlib import Path

from app.main import app

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = _REPO_ROOT / "openapi.json"


def export_openapi(output_path: Path = _OUTPUT_PATH) -> Path:
    """Dump `app.openapi()` to `output_path` with stable ordering + newline."""
    spec = app.openapi()
    serialized = json.dumps(spec, sort_keys=True, indent=2, ensure_ascii=False)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    written = export_openapi()
    print(f"wrote {written}")
