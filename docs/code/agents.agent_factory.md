---
type: CodeUnit
unit: agents.agent_factory
role: agent
key_modules: []
---
Factory + singleton for LLM agent instances: configuration-driven provider selection, one instance per provider for connection reuse, extensible via register_provider().
