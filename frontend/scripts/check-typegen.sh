#!/usr/bin/env bash
# check-typegen.sh — fail (nonzero exit) if the committed schema has drifted
# from the live backend contract. Regenerates to a temp location and diffs
# against the committed frontend/openapi.json + src/api/schema.d.ts; does
# NOT overwrite committed files.
set -euo pipefail

FRONTEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

TMP_OPENAPI="$TMP_DIR/openapi.json"
TMP_SCHEMA="$TMP_DIR/schema.d.ts"

"$FRONTEND_DIR/scripts/typegen.sh" "$TMP_OPENAPI" "$TMP_SCHEMA"

STATUS=0

if ! diff -u "$FRONTEND_DIR/openapi.json" "$TMP_OPENAPI"; then
  echo "error: frontend/openapi.json is out of date — run 'npm run typegen' and commit the result." >&2
  STATUS=1
fi

if ! diff -u "$FRONTEND_DIR/src/api/schema.d.ts" "$TMP_SCHEMA"; then
  echo "error: frontend/src/api/schema.d.ts is out of date — run 'npm run typegen' and commit the result." >&2
  STATUS=1
fi

exit $STATUS
