---
type: CodeUnit
unit: api
role: surface
key_modules: []
---
FastAPI application: command/correction routers, Neo4j read queries, the ask endpoint, and the SSE live-feed bridge. Reads Neo4j but never writes it — commands only append events to EventStoreDB.
