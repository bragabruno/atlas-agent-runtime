#!/usr/bin/env bash
# local.sh — run the agent-runtime trigger surface locally. PORT overrides the
# listen port (default 8000).
# NOTE: the FastAPI trigger surface (ADR-020, app/api/main.py) is not yet coded,
# and fastapi/uvicorn are not yet in [project.dependencies]. Until then this is a
# documented no-op rather than a hard failure — the package is a library today.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

if ! has_cmd uvicorn || ! python -c "import app.api.main" >/dev/null 2>&1; then
  skip "local" "FastAPI trigger surface not coded yet (ADR-020: app/api/main.py + fastapi/uvicorn)"
  exit 0
fi
port="${PORT:-8000}"
log_info "atlas-agent-runtime → http://127.0.0.1:${port} (Ctrl-C to stop)"
exec uvicorn app.api.main:app --reload --port "$port"
