#!/usr/bin/env bash
# Test runner for humans and agents: pins the pyenv interpreter and loads .env.
# Usage: ./scripts/test.sh [pytest args...]   (defaults to the unit suite)
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
source .env
set +a
if [ "$#" -eq 0 ]; then
  exec "$HOME/.pyenv/versions/3.10.7/bin/python" -m pytest tests -m "not integration" -q
fi
exec "$HOME/.pyenv/versions/3.10.7/bin/python" -m pytest "$@"
