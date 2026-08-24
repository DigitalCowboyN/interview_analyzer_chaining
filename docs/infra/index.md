# Infra / deployment topology

> Generated from `docker-compose.yml` by `make infra-index`. Do not edit by hand.

## Services

- **eventstore** (backing) — image `eventstore/eventstore:23.10.1-jammy`; requires: —
- **neo4j** (backing) — image `neo4j:5.26.0`; requires: —
- **neo4j-test** (backing) — image `neo4j:5.26.0`; requires: —
- **redis** (backing) — image `redis:7-alpine`; requires: —
- **app** (code) — runs `uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload`; requires: eventstore, neo4j, neo4j-test, redis
- **projection-service** (code) — runs `python -m src.run_projection_service --log-level INFO`; requires: eventstore, neo4j
- **worker** (code) — runs `celery -A src.celery_app worker --loglevel=info`; requires: eventstore, neo4j, redis

## Code → backing service (`talks_to`)

- `celery_app` → **redis**
- `events.store` → **eventstore**
- `projections.parked_events` → **eventstore**
- `projections.subscription_manager` → **eventstore**
- `utils.neo4j_driver` → **neo4j**

## Environment variables (inline; `.env` excluded)

- `ENABLE_PROJECTION_SERVICE` — projection-service
- `ESDB_CONNECTION_STRING` — app, projection-service, worker
- `EVENTSTORE_CLUSTER_SIZE` — eventstore
- `EVENTSTORE_ENABLE_ATOM_PUB_OVER_HTTP` — eventstore
- `EVENTSTORE_HTTP_PORT` — eventstore
- `EVENTSTORE_INSECURE` — eventstore
- `EVENTSTORE_INT_TCP_PORT` — eventstore
- `EVENTSTORE_MEM_DB` — eventstore
- `EVENTSTORE_RUN_PROJECTIONS` — eventstore
- `EVENTSTORE_SKIP_INDEX_VERIFY` — eventstore
- `EVENTSTORE_START_STANDARD_PROJECTIONS` — eventstore
- `NEO4J_ACCEPT_LICENSE_AGREEMENT` — neo4j, neo4j-test
- `NEO4J_AUTH` — neo4j, neo4j-test
- `PROJECTION_LANE_COUNT` — projection-service
- `PYTHONUNBUFFERED` — app, projection-service, worker
