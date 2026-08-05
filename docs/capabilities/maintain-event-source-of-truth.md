---
type: Capability
kind: primary
tier: enabling
implemented_by: [commands, events]
---
Hold the append-only, frozen-format event log that is the system's sole source of truth — every command validates intent, then appends; nothing rewrites history.
