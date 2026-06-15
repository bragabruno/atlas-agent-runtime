#!/usr/bin/env bash
# build.sh — build verification: the package imports cleanly. Publishes nothing.
# NOTE: no OpenAPI export here — the FastAPI trigger surface (ADR-020, app/api/)
# is not yet coded, so there is no served contract to export. Smoke-test the
# package that actually exists today.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

require_cmd python "pip install -e .[dev]"
run "import smoke (app)" python -c "import app"
log_ok "build verification passed"
