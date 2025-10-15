"""
Projection handlers for updating Neo4j from events.

Handlers are idempotent, version-guarded, and include retry-to-park logic.
"""
