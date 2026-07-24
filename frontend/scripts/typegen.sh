#!/usr/bin/env bash
# typegen.sh — regenerate frontend/openapi.json + src/api/schema.d.ts offline.
#
# Two steps, no running server required:
#   1. Import the FastAPI app object directly and dump app.openapi() to JSON
#      (importing src.main initializes singletons that need ANTHROPIC_API_KEY,
#      hence sourcing .env with `set -a` so it's actually exported).
#   2. Run openapi-typescript against that JSON to produce schema.d.ts.
#
# Run via `npm run typegen` (writes committed files) or `npm run typegen:check`
# (writes to a temp file and diffs — see check-typegen.sh).
#
# Backend contract changes require a regen + commit of both openapi.json and
# schema.d.ts; typegen:check catches drift in CI.
set -euo pipefail

FRONTEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$FRONTEND_DIR/.." && pwd)"
OPENAPI_JSON="${1:-$FRONTEND_DIR/openapi.json}"
SCHEMA_OUT="${2:-$FRONTEND_DIR/src/api/schema.d.ts}"

PYTHON_BIN="${PYTHON:-$HOME/.pyenv/shims/python}"

(
  cd "$REPO_ROOT"
  set -a
  # shellcheck disable=SC1091
  source .env 2>/dev/null || true
  set +a
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$FRONTEND_DIR/scripts/dump_openapi.py" "$OPENAPI_JSON"
)

npx --prefix "$FRONTEND_DIR" openapi-typescript "$OPENAPI_JSON" -o "$SCHEMA_OUT"
