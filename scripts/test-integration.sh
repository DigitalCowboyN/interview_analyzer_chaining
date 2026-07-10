#!/usr/bin/env bash
# Integration test runner: pins pyenv interpreter, loads .env, then overrides
# connection strings to the LOCAL TEST infra (docker compose test services).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
source .env
set +a
export NEO4J_URI=bolt://localhost:7688
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=testpassword
export ESDB_CONNECTION_STRING="esdb://localhost:2113?tls=false"
exec "$HOME/.pyenv/versions/3.10.7/bin/python" -m pytest "$@"
