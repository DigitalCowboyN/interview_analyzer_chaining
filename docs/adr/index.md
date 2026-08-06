# ADR Index

| id | title | status |
| --- | --- | --- |
| 0001 | EventStoreDB is the single source of truth | accepted |
| 0002 | CQRS write/read split | accepted |
| 0003 | The projection service is the sole writer to Neo4j | accepted |
| 0004 | Frozen wire format for event types and stream names | accepted |
| 0005 | Layered Mine architecture (ingestion → enrichment → lens → segment → export) | accepted |
| 0006 | Preserve, never rewrite (overlay-not-rewrite) | accepted |
| 0007 | Focused calls, not one-shot mega-calls | accepted |
| 0008 | Borrow neo4j-graphrag-python for resolution/retrieval, not its pipeline | superseded |
| 0009 | Lens engine — one generic event, one generic handler, zero per-lens code | accepted |
| 0010 | Provider strategy — config-selected chains, chat failover but pinned embeddings | accepted |
| 0011 | Deterministic-plus-review entity and person resolution, auto-link only within project | accepted |
| 0012 | Fragment dual-label rename, wire format stays frozen | accepted |
| 0013 | Read-side OKF exporter over Neo4j | accepted |
| 0014 | Hand-rolled hybrid retrieval instead of adopting neo4j-graphrag-python | accepted |
| 0015 | Adopt an OKF-conformant, non-blocking ADR corpus for architectural decisions | accepted |
| 0016 | Adopt knowledge cascade and spec/plan honesty check | accepted |
| 0017 | Adopt a capabilities domain linked to the code map | accepted |
| 0018 | Adopt the capability category axis and operations capabilities | accepted |
| 0019 | Capabilities are durable intent; implementation is a derived, replaceable link | accepted |
| 0020 | Adopt an OKF-extension typed-edge graph model | accepted |
