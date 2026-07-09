# Layer 2 (M4.2): Extractor Registry, Provider Strategy & Core Enrichment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize enrichment as a registry of focused extractors running on the Layer 1 ingestion output (fragments/utterances), behind a generalized multi-provider strategy, adding entities, claims, and embeddings — then retire the legacy pipeline.

**Architecture:** A new `src/enrichment/` package consumes Layer 1 aggregates (Interview: speakers/utterances; Sentence: fragments with offsets), runs config-declared extractors (each its own focused LLM call with schema-enforced structured output and numeric confidence), and emits events through the existing repositories; projection handlers materialize Entity/Claim nodes and vector properties. Chat calls go through a failover chain (Anthropic Haiku → Claude Code harness → OpenAI); embeddings are config-pinned (never silently switched) and model-tagged. Legacy `src/pipeline.py` and its collaborators are deleted after a parity check.

**Tech Stack:** Python 3.10, Pydantic v2, EventStoreDB (existing repositories), Neo4j 5.26 (vector indexes), FastAPI, anthropic SDK (tool-forcing), openai 1.93 (`response_format: json_schema`), sentence-transformers (local embeddings), spaCy.

**Spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md` (Layer 2 section + "Provider strategy" + "M4.2 scope additions").

## Global Constraints

- **Focused calls, not one-shots**: each dimension remains its own LLM call with its own prompt, response schema, and confidence. Never merge dimensions into one call.
- All confidences numeric in `[0.0, 1.0]`, validated by response models at parse time and payload models at command time.
- **Projection-delivery checklist** for every new event type: (1) handler registered in bootstrap, (2) event type in the subscription allowlists in `src/projections/config.py`, (3) Sentence-stream payloads carry `interview_id` (lane routing), (4) handlers raise (never no-op) when cross-stream MATCH targets aren't projected. The drift-guard test (`test_all_registered_event_types_are_subscription_allowed`) must stay green.
- **Embeddings**: provider config-pinned; every vector tagged `{model, dim}`; one Neo4j vector index per model; NO per-call provider failover for embeddings.
- **Chat failover**: per-call failover down the chain on quota/availability errors only (429 / 5xx / connection / subprocess failure); every emitted event records the provider and model that actually produced it.
- Deterministic IDs (existing convention): fragment UUID = `uuid5(NAMESPACE_DNS, f"{interview_id}:{index}")`; claim UUID = `uuid5(NAMESPACE_DNS, f"{interview_id}:claim:{utterance_id}:{ordinal}")`.
- Unit tests must not require API keys, network, or infrastructure — mock at module seams. Integration tests get `@pytest.mark.integration`.
- Environment: run tests with `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest ...` (plain `python` is not on PATH). Full unit suite: `... -m "not integration" -q`.
- Style: `black`, `flake8`. Follow existing patterns: Pydantic `*Data` payloads + aggregate command methods; `BaseProjectionHandler` subclasses; prompts in YAML with `{{`/`}}`-escaped JSON braces.

## File Structure (new/major)

```
src/agents/
  base_agent.py            # call_model gains schema param (default None)
  openai_agent.py          # json_schema response_format when schema given
  anthropic_agent.py       # forced tool-use when schema given
  claude_code_agent.py     # NEW: headless `claude -p` backend
  failover_agent.py        # NEW: chain-of-providers agent + CallResult
  context_builder.py       # unchanged (legacy); v2 lives in enrichment
src/enrichment/
  __init__.py
  models.py                # ExtractorSpec, EnrichedFragment, CallResult re-export
  graph_context.py         # GraphContextBuilder (speaker/utterance-aware)
  registry.py              # load extractors from config/extractors.yaml
  executor.py              # run extractors per scope with concurrency
  syntax_check.py          # spaCy cross-check for function/structure
  embedder.py              # Embedder protocol + OpenAIEmbedder + LocalEmbedder
  orchestrator.py          # EnrichmentOrchestrator (events via repositories)
  __main__.py              # python -m src.enrichment <interview_id>
src/models/
  extractor_responses.py   # NEW response models with numeric confidence
config/extractors.yaml     # the 7 core extractors + entities + claims declared
prompts/core_extractors.yaml  # ported prompts with numeric-confidence contract
```

---

### Task 1: Agent layer — `schema` parameter with API-level structured outputs

**Files:**
- Modify: `src/agents/base_agent.py` (call_model signature)
- Modify: `src/agents/openai_agent.py`, `src/agents/anthropic_agent.py`
- Test: `tests/agents/test_structured_outputs.py`

**Interfaces:**
- Produces: `BaseLLMAgent.call_model(self, function_prompt: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`. `schema` is a JSON Schema dict (from `Model.model_json_schema()`). When given: OpenAI passes `response_format={"type": "json_schema", "json_schema": {"name": "extraction", "strict": True, "schema": schema}}`; Anthropic passes `tools=[{"name": "extraction", "input_schema": schema}]` + `tool_choice={"type": "tool", "name": "extraction"}` and returns the tool-use input dict. When `schema is None`, behavior is exactly today's (backward compatible — all existing callers pass no schema).

- [ ] **Step 1: Write the failing tests** (`tests/agents/test_structured_outputs.py`)

```python
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SCHEMA = {
    "type": "object",
    "properties": {"purpose": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["purpose", "confidence"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_openai_passes_json_schema_response_format():
    from src.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent.__new__(OpenAIAgent)  # skip __init__ (no API key needed)
    agent.model = "gpt-4o-mini-2024-07-18"
    agent.max_tokens = 256
    agent.temperature = 0.2
    agent.client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = json.dumps({"purpose": "Query", "confidence": 0.9})
    completion.usage = MagicMock(total_tokens=10, prompt_tokens=5, completion_tokens=5)
    agent.client.chat.completions.create = AsyncMock(return_value=completion)

    result = await agent.call_model("classify this", schema=SCHEMA)

    assert result == {"purpose": "Query", "confidence": 0.9}
    kwargs = agent.client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["schema"] == SCHEMA


@pytest.mark.asyncio
async def test_openai_without_schema_sends_no_response_format():
    from src.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent.model = "gpt-4o-mini-2024-07-18"
    agent.max_tokens = 256
    agent.temperature = 0.2
    agent.client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = json.dumps({"ok": True})
    completion.usage = MagicMock(total_tokens=10, prompt_tokens=5, completion_tokens=5)
    agent.client.chat.completions.create = AsyncMock(return_value=completion)

    await agent.call_model("classify this")

    kwargs = agent.client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs


@pytest.mark.asyncio
async def test_anthropic_forces_tool_use_and_returns_input():
    from src.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent.__new__(AnthropicAgent)
    agent.model = "claude-3-haiku-20240307"
    agent.max_tokens = 256
    agent.temperature = 0.2
    agent.client = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"purpose": "Query", "confidence": 0.9}
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=5, output_tokens=5)
    agent.client.messages.create = AsyncMock(return_value=response)

    result = await agent.call_model("classify this", schema=SCHEMA)

    assert result == {"purpose": "Query", "confidence": 0.9}
    kwargs = agent.client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "extraction"}
    assert kwargs["tools"][0]["input_schema"] == SCHEMA
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/agents/test_structured_outputs.py -q`
Expected: FAIL — `TypeError: call_model() got an unexpected keyword argument 'schema'`

- [ ] **Step 3: Implement**

`src/agents/base_agent.py` — change the abstract signature (docstring updated to describe `schema`):

```python
    @abstractmethod
    async def call_model(
        self, function_prompt: str, schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
```

(add `Optional` to typing imports.)

`src/agents/openai_agent.py` — in `call_model`, accept the param and thread it into the create call. Read the existing method first; the change is localized to the signature and the `create(...)` kwargs:

```python
    async def call_model(self, function_prompt: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...
        create_kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": function_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if schema is not None:
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": schema},
            }
        response = await self.client.chat.completions.create(**create_kwargs)
        ...
```

`src/agents/anthropic_agent.py` — same pattern; when `schema` is given, add `tools`/`tool_choice` and extract the result from the tool_use block instead of text:

```python
        if schema is not None:
            create_kwargs["tools"] = [
                {"name": "extraction", "description": "Return the extraction result.", "input_schema": schema}
            ]
            create_kwargs["tool_choice"] = {"type": "tool", "name": "extraction"}
        response = await self.client.messages.create(**create_kwargs)
        if schema is not None:
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    return dict(block.input)
            logger.warning("Anthropic response missing tool_use block; returning {}")
            return {}
```

Keep all existing retry/metrics/logging wrapping intact — only the request construction and (for Anthropic) response extraction change. Preserve the no-schema path byte-for-byte.

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/agents -m "not integration" -q`
Expected: new tests PASS; no regressions in existing agent tests.

- [ ] **Step 5: Commit**

```bash
git add src/agents tests/agents/test_structured_outputs.py
git commit -m "feat: API-level structured outputs (schema param) for OpenAI and Anthropic agents"
```

---

### Task 2: ClaudeCodeAgent — headless `claude -p` provider

**Files:**
- Create: `src/agents/claude_code_agent.py`
- Modify: `src/agents/agent_factory.py` (register `claude_code`)
- Modify: `config.yaml` (add `claude_code:` block)
- Test: `tests/agents/test_claude_code_agent.py`

**Interfaces:**
- Produces: `ClaudeCodeAgent(BaseLLMAgent)` with `call_model(prompt, schema=None)` shelling out to the Claude Code CLI: `claude -p <prompt> --output-format json [--model <model>]` via `asyncio.create_subprocess_exec`; parses the CLI's JSON envelope, extracts `result`, strips optional ```json fences, `json.loads` → dict; `{}` on parse failure; raises `RuntimeError` on non-zero exit. `schema` is not API-enforced here (accepted caveat: behavior replication without API-lifecycle testing) — when given, a one-line contract (`"Respond ONLY with JSON matching this schema: <schema json>"`) is appended to the prompt and Pydantic remains the validator downstream.
- Config: `claude_code: {command: "claude", model: "haiku", timeout_seconds: 120}`.

- [ ] **Step 1: Write the failing tests** (`tests/agents/test_claude_code_agent.py`)

```python
import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.claude_code_agent import ClaudeCodeAgent


def make_agent():
    return ClaudeCodeAgent(config_dict={"claude_code": {"command": "claude", "model": "haiku", "timeout_seconds": 5}})


def cli_envelope(result_str):
    return json.dumps({"type": "result", "result": result_str}).encode()


@pytest.mark.asyncio
async def test_parses_result_json():
    agent = make_agent()
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(cli_envelope('{"purpose": "Query", "confidence": 0.8}'), b""))
    proc.returncode = 0
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        result = await agent.call_model("classify this")
    assert result == {"purpose": "Query", "confidence": 0.8}
    argv = spawn.call_args.args
    assert argv[0] == "claude" and "-p" in argv and "--output-format" in argv


@pytest.mark.asyncio
async def test_strips_code_fences():
    agent = make_agent()
    fenced = "```json\n{\"purpose\": \"Query\", \"confidence\": 0.8}\n```"
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(cli_envelope(fenced), b""))
    proc.returncode = 0
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        result = await agent.call_model("classify this")
    assert result["purpose"] == "Query"


@pytest.mark.asyncio
async def test_schema_appends_contract_to_prompt():
    agent = make_agent()
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(cli_envelope("{}"), b""))
    proc.returncode = 0
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        await agent.call_model("classify this", schema=schema)
    prompt_arg = spawn.call_args.args[2]  # claude -p <prompt>
    assert "Respond ONLY with JSON matching this schema" in prompt_arg


@pytest.mark.asyncio
async def test_nonzero_exit_raises():
    agent = make_agent()
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b"boom"))
    proc.returncode = 1
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(RuntimeError, match="claude CLI failed"):
            await agent.call_model("classify this")


def test_registered_in_factory():
    from src.agents.agent_factory import AgentFactory

    AgentFactory._providers = {}  # force re-init
    assert "claude_code" in AgentFactory.get_available_providers()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/agents/test_claude_code_agent.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agents.claude_code_agent'`

- [ ] **Step 3: Implement** (`src/agents/claude_code_agent.py`)

```python
"""Claude Code harness as an LLM provider.

Shells out to the local `claude` CLI in headless print mode. Behavior
replication of an API provider; the accepted caveat is that raw API lifecycle
(HTTP errors, token metering) is not exercised through this interface.
"""

import asyncio
import json
import re
from typing import Any, Dict, Optional

from src.config import config as global_config
from src.utils.logger import get_logger

from .base_agent import BaseLLMAgent

logger = get_logger()

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ClaudeCodeAgent(BaseLLMAgent):
    """LLM provider backed by the local Claude Code CLI (`claude -p`)."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = (config_dict or global_config).get("claude_code", {})
        self.command = cfg.get("command", "claude")
        self.model = cfg.get("model", "haiku")
        self.timeout = cfg.get("timeout_seconds", 120)

    async def call_model(
        self, function_prompt: str, schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        prompt = function_prompt
        if schema is not None:
            prompt += (
                "\n\nRespond ONLY with JSON matching this schema: "
                + json.dumps(schema)
            )

        argv = [self.command, "-p", prompt, "--output-format", "json"]
        if self.model:
            argv += ["--model", self.model]

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude CLI timed out after {self.timeout}s")

        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {stderr.decode()[:500]}")

        try:
            envelope = json.loads(stdout.decode())
            result_text = envelope.get("result", "")
            cleaned = _FENCE_RE.sub("", result_text).strip()
            return json.loads(cleaned) if cleaned else {}
        except json.JSONDecodeError as e:
            logger.warning(f"ClaudeCodeAgent: unparseable response ({e}); returning {{}}")
            return {}

    def get_provider_name(self) -> str:
        return "claude_code"

    def get_model_name(self) -> str:
        return f"claude-code/{self.model}"
```

In `src/agents/agent_factory.py` `_initialize_providers`, add:

```python
            from .claude_code_agent import ClaudeCodeAgent

            cls._providers = {
                "openai": OpenAIAgent,
                "anthropic": AnthropicAgent,
                "claude_code": ClaudeCodeAgent,
            }
```

In `config.yaml`, after the `anthropic:` block:

```yaml
claude_code:
  command: "claude"
  model: "haiku"
  timeout_seconds: 120
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/agents -m "not integration" -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/agents config.yaml tests/agents/test_claude_code_agent.py
git commit -m "feat: ClaudeCodeAgent provider (headless claude -p backend)"
```

---

### Task 3: FailoverAgent — the provider chain

**Files:**
- Create: `src/agents/failover_agent.py`
- Modify: `config.yaml` (`llm.provider: "anthropic"`, `llm.chain: ["anthropic", "claude_code", "openai"]`)
- Test: `tests/agents/test_failover_agent.py`

**Interfaces:**
- Produces:
  - `CallResult(BaseModel)`: `data: Dict[str, Any]`, `provider: str`, `model: str`
  - `FailoverAgent(BaseLLMAgent)` built from config `llm.chain` (list of provider names resolved via `AgentFactory.create_agent(name)`). `call_model(prompt, schema=None)` returns the dict (interface compat). `call(prompt, schema=None) -> CallResult` returns data + which provider/model served it — **the enrichment executor uses `call()`** so events record true provenance.
  - Failover triggers ONLY on availability errors: exception class names containing `RateLimitError`, `APIStatusError`/`InternalServerError`, `APIConnectionError`/`ConnectionError`, `TimeoutError`, or `RuntimeError` from the CLI provider. Any other exception propagates (a bad prompt should fail loudly, not burn the chain).
  - `get_failover_agent(config_dict=None) -> FailoverAgent` module factory (singleton per config identity not required; simple construction).

- [ ] **Step 1: Write the failing tests** (`tests/agents/test_failover_agent.py`)

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.failover_agent import CallResult, FailoverAgent


def make_provider(name, model, result=None, error=None):
    p = MagicMock()
    p.get_provider_name.return_value = name
    p.get_model_name.return_value = model
    if error is not None:
        p.call_model = AsyncMock(side_effect=error)
    else:
        p.call_model = AsyncMock(return_value=result)
    return p


class FakeRateLimitError(Exception):
    pass


FakeRateLimitError.__name__ = "RateLimitError"


@pytest.mark.asyncio
async def test_first_healthy_provider_serves():
    a = make_provider("anthropic", "haiku", result={"x": 1})
    b = make_provider("openai", "gpt", result={"x": 2})
    agent = FailoverAgent([a, b])
    result = await agent.call("prompt")
    assert result == CallResult(data={"x": 1}, provider="anthropic", model="haiku")
    b.call_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_fails_over_on_rate_limit():
    a = make_provider("anthropic", "haiku", error=FakeRateLimitError("429"))
    b = make_provider("openai", "gpt", result={"x": 2})
    agent = FailoverAgent([a, b])
    result = await agent.call("prompt")
    assert result.provider == "openai"
    assert result.data == {"x": 2}


@pytest.mark.asyncio
async def test_non_availability_error_propagates():
    a = make_provider("anthropic", "haiku", error=ValueError("bad prompt"))
    b = make_provider("openai", "gpt", result={"x": 2})
    agent = FailoverAgent([a, b])
    with pytest.raises(ValueError):
        await agent.call("prompt")
    b.call_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_exhausted_chain_raises_last_error():
    a = make_provider("anthropic", "haiku", error=FakeRateLimitError("429"))
    b = make_provider("openai", "gpt", error=RuntimeError("claude CLI failed"))
    agent = FailoverAgent([a, b])
    with pytest.raises(RuntimeError):
        await agent.call("prompt")


@pytest.mark.asyncio
async def test_call_model_returns_plain_dict():
    a = make_provider("anthropic", "haiku", result={"x": 1})
    agent = FailoverAgent([a])
    assert await agent.call_model("prompt") == {"x": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/agents/test_failover_agent.py -q`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement** (`src/agents/failover_agent.py`)

```python
"""Provider chain with per-call failover on availability errors.

The baseline provider-strategy pattern: chat calls try the configured chain
in order; only quota/availability failures advance the chain (a malformed
request should fail loudly). Events record the provider/model that actually
served each call via CallResult.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.utils.logger import get_logger

from .base_agent import BaseLLMAgent

logger = get_logger()

_AVAILABILITY_ERROR_NAMES = (
    "RateLimitError",
    "APIStatusError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailableError",
)


def _is_availability_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if any(marker in name for marker in _AVAILABILITY_ERROR_NAMES):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError, RuntimeError))


class CallResult(BaseModel):
    """A chat-call result with provenance."""

    data: Dict[str, Any]
    provider: str
    model: str


class FailoverAgent(BaseLLMAgent):
    """Tries providers in order; advances only on availability errors."""

    def __init__(self, providers: List[BaseLLMAgent]):
        super().__init__()
        if not providers:
            raise ValueError("FailoverAgent requires at least one provider")
        self.providers = providers

    async def call(
        self, function_prompt: str, schema: Optional[Dict[str, Any]] = None
    ) -> CallResult:
        last_error: Optional[Exception] = None
        for provider in self.providers:
            try:
                data = await provider.call_model(function_prompt, schema=schema)
                return CallResult(
                    data=data,
                    provider=provider.get_provider_name(),
                    model=provider.get_model_name(),
                )
            except Exception as exc:
                if not _is_availability_error(exc):
                    raise
                logger.warning(
                    f"Provider {provider.get_provider_name()} unavailable "
                    f"({type(exc).__name__}); failing over"
                )
                last_error = exc
        raise last_error  # chain exhausted

    async def call_model(
        self, function_prompt: str, schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return (await self.call(function_prompt, schema=schema)).data

    def get_provider_name(self) -> str:
        return "failover:" + ",".join(p.get_provider_name() for p in self.providers)

    def get_model_name(self) -> str:
        return self.providers[0].get_model_name()


def get_failover_agent(config_dict: Optional[Dict[str, Any]] = None) -> FailoverAgent:
    """Build the chain from config['llm']['chain'] (falls back to single provider)."""
    from src.agents.agent_factory import AgentFactory
    from src.config import config as global_config

    cfg = config_dict if config_dict is not None else global_config
    chain = cfg.get("llm", {}).get("chain")
    if not chain:
        chain = [cfg.get("llm", {}).get("provider", "anthropic")]
    return FailoverAgent([AgentFactory.create_agent(name) for name in chain])
```

`config.yaml` llm block becomes:

```yaml
llm:
  provider: "anthropic"  # Options: "openai", "anthropic", "claude_code"
  chain: ["anthropic", "claude_code", "openai"]  # enrichment failover order
```

Note: `agent_factory.py`'s module-level `agent = AgentFactory.create_agent()` will now build the Anthropic agent at import. Verify `AnthropicAgent.__init__` raises cleanly if ANTHROPIC_API_KEY is missing — if it raises at import time in test environments without the key, keep `llm.provider` as-is functional by ensuring `.env` provides ANTHROPIC_API_KEY (it does in this repo; CI note goes in the report if not).

ALSO verify the configured Anthropic model is still live: `claude-3-haiku-20240307` (config.yaml) dates from 2024 and is likely deprecated. Check with a one-off live call (`pytest tests/integration/test_anthropic_api_messages.py -k structured -x` or a curl) and bump `anthropic.model_name` to the current cheapest Haiku (e.g., `claude-haiku-4-5-20251001`) if needed — a config-only change, but it gates every enrichment call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/agents -m "not integration" -q` and then the full unit suite (config provider flip can ripple): `... -m "not integration" -q`
Expected: PASS. If existing tests assert the OpenAI singleton specifically, fix them to be provider-agnostic (patch `AgentFactory` rather than assuming provider).

- [ ] **Step 5: Commit**

```bash
git add src/agents/failover_agent.py config.yaml tests/agents/test_failover_agent.py
git commit -m "feat: FailoverAgent provider chain; Anthropic as primary provider"
```

---

### Task 4: Extractor response models (numeric confidence)

**Files:**
- Create: `src/models/extractor_responses.py`
- Test: `tests/models/test_extractor_responses.py`

**Interfaces:**
- Produces (all Pydantic, all with `confidence: float = Field(..., ge=0.0, le=1.0)` unless noted):
  - `FunctionTypeResult(function_type: str, confidence)`
  - `StructureTypeResult(structure_type: str, confidence)`
  - `PurposeResult(purpose: str, confidence)`
  - `TopicLevel1Result(topic_level_1: str, confidence)`
  - `TopicLevel3Result(topic_level_3: str, confidence)`
  - `OverallKeywordsResult(overall_keywords: List[str])` (no confidence — list task)
  - `DomainKeywordsResult(domain_keywords: List[str])`
  - `EntityMention(text: str, entity_type: str, start: int (ge 0), end: int (gt 0), confidence)` + model_validator `end > start`
  - `EntityMentionsResult(entities: List[EntityMention])`
  - `ClaimItem(text: str, kind: Literal["assertion","commitment","request"], confidence)`
  - `ClaimsResult(claims: List[ClaimItem])`
- These are the `response_model`s the registry references by name. JSON Schemas for the agent `schema` param come from `Model.model_json_schema()`.

- [ ] **Step 1: Write the failing tests** (`tests/models/test_extractor_responses.py`)

```python
import pytest
from pydantic import ValidationError

from src.models.extractor_responses import (
    ClaimsResult,
    EntityMentionsResult,
    PurposeResult,
)


def test_purpose_result_validates_confidence_range():
    PurposeResult.model_validate({"purpose": "Query", "confidence": 0.5})
    with pytest.raises(ValidationError):
        PurposeResult.model_validate({"purpose": "Query", "confidence": 1.5})
    with pytest.raises(ValidationError):
        PurposeResult.model_validate({"purpose": "Query", "confidence": -0.1})


def test_entity_mentions_span_validated():
    ok = EntityMentionsResult.model_validate(
        {"entities": [{"text": "Neo4j", "entity_type": "product", "start": 4, "end": 9, "confidence": 0.9}]}
    )
    assert ok.entities[0].entity_type == "product"
    with pytest.raises(ValidationError):
        EntityMentionsResult.model_validate(
            {"entities": [{"text": "x", "entity_type": "product", "start": 9, "end": 4, "confidence": 0.9}]}
        )


def test_claim_kind_restricted():
    ClaimsResult.model_validate({"claims": [{"text": "We will ship Friday", "kind": "commitment", "confidence": 0.8}]})
    with pytest.raises(ValidationError):
        ClaimsResult.model_validate({"claims": [{"text": "x", "kind": "vibe", "confidence": 0.8}]})


def test_schema_export_is_json_schema():
    schema = PurposeResult.model_json_schema()
    assert schema["type"] == "object"
    assert "purpose" in schema["properties"]
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/models/test_extractor_responses.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement** (`src/models/extractor_responses.py`)

```python
"""Response models for registry extractors (numeric confidence throughout)."""

from typing import List, Literal

from pydantic import BaseModel, Field, model_validator

Confidence = Field(..., ge=0.0, le=1.0)


class FunctionTypeResult(BaseModel):
    function_type: str
    confidence: float = Confidence


class StructureTypeResult(BaseModel):
    structure_type: str
    confidence: float = Confidence


class PurposeResult(BaseModel):
    purpose: str
    confidence: float = Confidence


class TopicLevel1Result(BaseModel):
    topic_level_1: str
    confidence: float = Confidence


class TopicLevel3Result(BaseModel):
    topic_level_3: str
    confidence: float = Confidence


class OverallKeywordsResult(BaseModel):
    overall_keywords: List[str]


class DomainKeywordsResult(BaseModel):
    domain_keywords: List[str]


class EntityMention(BaseModel):
    text: str
    entity_type: str = Field(..., description="person | organization | product | tool | other")
    start: int = Field(..., ge=0, description="Offset within the fragment text")
    end: int = Field(..., gt=0)
    confidence: float = Confidence

    @model_validator(mode="after")
    def _span_valid(self) -> "EntityMention":
        if self.end <= self.start:
            raise ValueError("end must be > start")
        return self


class EntityMentionsResult(BaseModel):
    entities: List[EntityMention]


class ClaimItem(BaseModel):
    text: str
    kind: Literal["assertion", "commitment", "request"]
    confidence: float = Confidence


class ClaimsResult(BaseModel):
    claims: List[ClaimItem]
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/models -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/models/extractor_responses.py tests/models/test_extractor_responses.py && git commit -m "feat: extractor response models with numeric confidence"`

---

### Task 5: GraphContextBuilder — speaker/utterance-aware contexts

**Files:**
- Create: `src/enrichment/__init__.py` (empty), `src/enrichment/graph_context.py`
- Test: `tests/enrichment/__init__.py` (empty), `tests/enrichment/test_graph_context.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure function over Layer 1 shapes).
- Produces:
  - `FragmentView(BaseModel)`: `index: int`, `text: str`, `speaker_handle: str`, `utterance_id: Optional[str]` — the enrichment-side view of a fragment (Task 10's orchestrator constructs these from aggregates).
  - `GraphContextBuilder(context_windows: Dict[str, int])` with `build_all(fragments: List[FragmentView], utterance_texts: Dict[str, str]) -> List[Dict[str, str]]`. For each fragment returns context strings keyed exactly like today's config (`immediate_context`, `observer_context`, `broader_context`, `overall_context`) — each line rendered `[S1]: text`, target line marked `>>> [S1]: text <<<` — plus `utterance_context` (the full stitched utterance text, or the fragment text when it has no utterance).

- [ ] **Step 1: Write the failing tests** (`tests/enrichment/test_graph_context.py`)

```python
from src.enrichment.graph_context import FragmentView, GraphContextBuilder

WINDOWS = {"immediate_context": 1, "observer_context": 2, "broader_context": 3, "overall_context": 5}


def make_fragments():
    rows = [
        ("S1", "Hi, thanks for joining."),
        ("S2", "Happy to be here."),
        ("S1", "First question:"),
        ("S2", "Sure."),
        ("S1", "how do you build flashable files?"),
    ]
    return [
        FragmentView(index=i, text=t, speaker_handle=h, utterance_id=("u-1" if i in (2, 4) else None))
        for i, (h, t) in enumerate(rows)
    ]


def test_contexts_render_speaker_labels_and_mark_target():
    builder = GraphContextBuilder(WINDOWS)
    contexts = builder.build_all(make_fragments(), {"u-1": "First question: how do you build flashable files?"})
    immediate = contexts[2]["immediate_context"]
    assert "[S2]: Happy to be here." in immediate
    assert ">>> [S1]: First question: <<<" in immediate
    assert "[S1]: how do you build flashable files?" in immediate


def test_window_sizes_respected():
    builder = GraphContextBuilder(WINDOWS)
    contexts = builder.build_all(make_fragments(), {})
    immediate = contexts[0]["immediate_context"]
    assert "First question" not in immediate  # window 1: only fragments 0-1


def test_utterance_context_supplies_stitched_thought():
    builder = GraphContextBuilder(WINDOWS)
    contexts = builder.build_all(make_fragments(), {"u-1": "First question: how do you build flashable files?"})
    assert contexts[2]["utterance_context"] == "First question: how do you build flashable files?"
    assert contexts[1]["utterance_context"] == "Happy to be here."  # no utterance -> own text


def test_all_configured_windows_present():
    builder = GraphContextBuilder(WINDOWS)
    contexts = builder.build_all(make_fragments(), {})
    assert set(contexts[0].keys()) == set(WINDOWS.keys()) | {"utterance_context"}
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.

- [ ] **Step 3: Implement** (`src/enrichment/graph_context.py`)

```python
"""Speaker- and utterance-aware context building for enrichment extractors.

Replaces the legacy ContextBuilder's bare-sentence windows with lines rendered
as `[S1]: text` from the Layer 1 graph, plus the stitched utterance a fragment
belongs to.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FragmentView(BaseModel):
    """Enrichment-side view of one fragment."""

    index: int = Field(..., ge=0)
    text: str
    speaker_handle: str
    utterance_id: Optional[str] = None


class GraphContextBuilder:
    """Builds context strings with speaker labels and utterance awareness."""

    def __init__(self, context_windows: Dict[str, int]):
        self.context_windows = context_windows

    def build_all(
        self, fragments: List[FragmentView], utterance_texts: Dict[str, str]
    ) -> List[Dict[str, str]]:
        contexts: List[Dict[str, str]] = []
        for frag in fragments:
            ctx: Dict[str, str] = {}
            for name, window in self.context_windows.items():
                ctx[name] = self._window(fragments, frag.index, window)
            if frag.utterance_id and frag.utterance_id in utterance_texts:
                ctx["utterance_context"] = utterance_texts[frag.utterance_id]
            else:
                ctx["utterance_context"] = frag.text
            contexts.append(ctx)
        return contexts

    @staticmethod
    def _window(fragments: List[FragmentView], idx: int, window: int) -> str:
        start = max(0, idx - window)
        end = min(len(fragments), idx + window + 1)
        lines = []
        for frag in fragments[start:end]:
            line = f"[{frag.speaker_handle}]: {frag.text}"
            if frag.index == idx:
                line = f">>> {line} <<<"
            lines.append(line)
        return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/enrichment -q` → 4 passed.
- [ ] **Step 5: Commit** — `git add src/enrichment tests/enrichment && git commit -m "feat: GraphContextBuilder with speaker labels and utterance context"`

---

### Task 6: Core extractor prompts + registry

**Files:**
- Create: `prompts/core_extractors.yaml`, `config/extractors.yaml`, `src/enrichment/registry.py`, `src/enrichment/models.py`
- Test: `tests/enrichment/test_registry.py`

**Interfaces:**
- Produces:
  - `ExtractorSpec(BaseModel)`: `name: str`, `prompt_key: str` (key in the prompts YAML), `response_model: str` (class name in `src.models.extractor_responses`), `context_needs: List[str]` (context keys the prompt's `{context}` uses; empty = sentence-only), `scope: Literal["fragment","utterance","document"]`, `enabled: bool = True`. Method `resolve_model() -> type[BaseModel]` (getattr on the module; raises `ValueError` for unknown names).
  - `ExtractorRegistry.load(path: str = "config/extractors.yaml") -> List[ExtractorSpec]` — order preserved; disabled extractors filtered out.
- `config/extractors.yaml` declares the 7 core fragment extractors (`function_type`, `structure_type`, `purpose`, `topic_level_1`, `topic_level_3`, `overall_keywords`, `domain_keywords`), `entity_mentions` (fragment), `claims` (utterance).
- `prompts/core_extractors.yaml`: ports of today's 7 prompts (same option lists — copy them verbatim from `prompts/task_prompts.yaml`) with the format line changed to demand numeric confidence, e.g. `{{"purpose": "<Purpose>", "confidence": <number between 0 and 1>}}`; plus `entity_mentions` and `claims` prompts. Placeholders: `{sentence}` and/or `{context}` only ( `domain_keywords` also uses `{domain_keywords}` — the domain list from config, as today).

- [ ] **Step 1: Write the failing tests** (`tests/enrichment/test_registry.py`)

```python
from src.enrichment.registry import ExtractorRegistry
from src.models.extractor_responses import PurposeResult


def test_loads_core_extractors_in_order():
    specs = ExtractorRegistry.load("config/extractors.yaml")
    names = [s.name for s in specs]
    for expected in [
        "function_type", "structure_type", "purpose", "topic_level_1",
        "topic_level_3", "overall_keywords", "domain_keywords",
        "entity_mentions", "claims",
    ]:
        assert expected in names


def test_scopes_and_models_resolve():
    specs = {s.name: s for s in ExtractorRegistry.load("config/extractors.yaml")}
    assert specs["purpose"].scope == "fragment"
    assert specs["claims"].scope == "utterance"
    assert specs["purpose"].resolve_model() is PurposeResult


def test_prompts_exist_for_every_extractor():
    from src.utils.helpers import load_yaml

    prompts = load_yaml("prompts/core_extractors.yaml")
    for spec in ExtractorRegistry.load("config/extractors.yaml"):
        assert spec.prompt_key in prompts, spec.name
        assert "prompt" in prompts[spec.prompt_key]


def test_disabled_extractors_filtered(tmp_path):
    doc = tmp_path / "x.yaml"
    doc.write_text(
        "extractors:\n"
        "  - {name: a, prompt_key: purpose, response_model: PurposeResult, scope: fragment}\n"
        "  - {name: b, prompt_key: purpose, response_model: PurposeResult, scope: fragment, enabled: false}\n"
    )
    specs = ExtractorRegistry.load(str(doc))
    assert [s.name for s in specs] == ["a"]
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.

- [ ] **Step 3: Implement.**

`src/enrichment/models.py`:

```python
"""Enrichment data models."""

import importlib
from typing import List, Literal

from pydantic import BaseModel, Field


class ExtractorSpec(BaseModel):
    """One declared, focused extractor (one LLM call per unit of its scope)."""

    name: str
    prompt_key: str
    response_model: str
    context_needs: List[str] = Field(default_factory=list)
    scope: Literal["fragment", "utterance", "document"] = "fragment"
    enabled: bool = True

    def resolve_model(self) -> type:
        module = importlib.import_module("src.models.extractor_responses")
        model = getattr(module, self.response_model, None)
        if model is None:
            raise ValueError(f"Unknown response model: {self.response_model}")
        return model
```

`src/enrichment/registry.py`:

```python
"""Loads the declared extractor set from YAML config."""

from typing import List

from src.utils.helpers import load_yaml

from .models import ExtractorSpec


class ExtractorRegistry:
    """Reads config/extractors.yaml into ordered, enabled ExtractorSpecs."""

    @staticmethod
    def load(path: str = "config/extractors.yaml") -> List[ExtractorSpec]:
        doc = load_yaml(path)
        specs = [ExtractorSpec.model_validate(item) for item in doc["extractors"]]
        return [s for s in specs if s.enabled]
```

`config/extractors.yaml`:

```yaml
extractors:
  - name: function_type
    prompt_key: function_type
    response_model: FunctionTypeResult
    scope: fragment
  - name: structure_type
    prompt_key: structure_type
    response_model: StructureTypeResult
    scope: fragment
  - name: purpose
    prompt_key: purpose
    response_model: PurposeResult
    context_needs: [observer_context]
    scope: fragment
  - name: topic_level_1
    prompt_key: topic_level_1
    response_model: TopicLevel1Result
    context_needs: [immediate_context]
    scope: fragment
  - name: topic_level_3
    prompt_key: topic_level_3
    response_model: TopicLevel3Result
    context_needs: [broader_context]
    scope: fragment
  - name: overall_keywords
    prompt_key: overall_keywords
    response_model: OverallKeywordsResult
    context_needs: [overall_context]
    scope: fragment
  - name: domain_keywords
    prompt_key: domain_keywords
    response_model: DomainKeywordsResult
    scope: fragment
  - name: entity_mentions
    prompt_key: entity_mentions
    response_model: EntityMentionsResult
    scope: fragment
  - name: claims
    prompt_key: claims
    response_model: ClaimsResult
    context_needs: [utterance_context]
    scope: utterance
```

`prompts/core_extractors.yaml` — port each of the 7 prompts from `prompts/task_prompts.yaml` **keeping the option lists verbatim** and changing only the Format line to numeric confidence. Two shown in full here; replicate the pattern exactly for the rest (the implementer copies option lists from the legacy file):

```yaml
purpose:
  prompt: |
    Given the sentence and its surrounding context, identify the sentence's purpose
    from the perspective of an outside observer. Speaker labels like [S1] identify
    who is talking.

    Options:
      - Statement
      - Query
      - Exclamation
      - Answer
      - Commentary
      - Observation
      - Retraction
      - Mockery
      - Objection
      - Clarification
      - Conclusion
      - Confession
      - Speculation
      - Recitation
      - Correction
      - Explanation
      - Qualification
      - Threat
      - Warning
      - Advisory
      - Request
      - Addendum
      - Musing
      - Amendment

    Format: {{"purpose": "<Purpose>", "confidence": <number between 0 and 1>}}

    Sentence: "{sentence}"
    Context:
    """
    {context}
    """

    Provide your response explicitly formatted as JSON.

entity_mentions:
  prompt: |
    Identify entity mentions in the sentence: people, organizations, products,
    tools. For each entity report its exact character span WITHIN THE SENTENCE
    (0-based start offset, exclusive end offset) such that
    sentence[start:end] == text. Do not include pronouns.

    Format: {{"entities": [{{"text": "<exact substring>", "entity_type": "person|organization|product|tool|other", "start": <int>, "end": <int>, "confidence": <number between 0 and 1>}}]}}

    If there are no entities, return {{"entities": []}}.

    Sentence: "{sentence}"

    Provide your response explicitly formatted as JSON.

claims:
  prompt: |
    The following is one speaker's complete utterance from a conversation.
    Extract the claims it makes. A claim is one of:
      - assertion: a statement of fact or belief about the world
      - commitment: something the speaker commits to doing
      - request: something the speaker asks another party to do

    Ignore small talk and filler. Quote or closely paraphrase the utterance.

    Format: {{"claims": [{{"text": "<claim>", "kind": "assertion|commitment|request", "confidence": <number between 0 and 1>}}]}}

    If there are no claims, return {{"claims": []}}.

    Utterance:
    """
    {sentence}
    """

    Provide your response explicitly formatted as JSON.
```

(`function_type`, `structure_type`: no context, `{sentence}` only. `topic_level_1`/`topic_level_3`: `{sentence}` + `{context}`. `overall_keywords`: `{context}` only. `domain_keywords`: `{sentence}` + `{domain_keywords}`. All Format lines use `"confidence": <number between 0 and 1>` except the two keyword extractors, which keep their list-only format.)

- [ ] **Step 4: Run to verify pass** — `pytest tests/enrichment -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/enrichment config/extractors.yaml prompts/core_extractors.yaml tests/enrichment/test_registry.py && git commit -m "feat: extractor registry, declared core extractors, ported prompts"`

---

### Task 7: spaCy cross-check (syntax review signal)

**Files:**
- Create: `src/enrichment/syntax_check.py`
- Test: `tests/enrichment/test_syntax_check.py`

**Interfaces:**
- Produces: `syntax_flags(text: str, function_type: str, structure_type: str) -> Dict[str, str]` — returns `{}` when spaCy agrees (or spaCy unavailable), else e.g. `{"function_type_disagreement": "spacy=interrogative llm=declarative"}`. Deterministic rules: interrogative iff text ends with `?` (or root sentence has an interrogative structure); imperative iff root verb in base form with no subject; exclamatory iff ends with `!`; else declarative. Structure: simple = 1 verbal clause, compound = coordinated independent clauses (root conj with `cc`), complex = subordinate clause markers (`mark`, `advcl`, `ccomp`, `relcl`), compound-complex = both.

- [ ] **Step 1: Write the failing tests**

```python
from src.enrichment.syntax_check import syntax_flags


def test_agreement_returns_empty():
    assert syntax_flags("Can you hear me?", "interrogative", "simple") == {}


def test_function_disagreement_flagged():
    flags = syntax_flags("Can you hear me?", "declarative", "simple")
    assert "function_type_disagreement" in flags
    assert "interrogative" in flags["function_type_disagreement"]


def test_structure_disagreement_flagged():
    flags = syntax_flags(
        "I stayed home because it rained, and Bob left early.", "declarative", "simple"
    )
    assert "structure_type_disagreement" in flags
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.

- [ ] **Step 3: Implement** (`src/enrichment/syntax_check.py`)

```python
"""Deterministic spaCy cross-check for function/structure classifications.

Does NOT replace the LLM calls (spec: expansion, not reduction); disagreement
becomes a review flag carried on the AnalysisGenerated event.
"""

from typing import Dict

from src.utils.text_processing import nlp


def _spacy_function_type(doc, text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("?"):
        return "interrogative"
    if stripped.endswith("!"):
        return "exclamatory"
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is not None and root.pos_ == "VERB" and root.tag_ == "VB":
        has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in root.children)
        if not has_subject:
            return "imperative"
    return "declarative"


def _spacy_structure_type(doc) -> str:
    has_coord = any(t.dep_ == "conj" and t.head.dep_ == "ROOT" for t in doc)
    has_sub = any(t.dep_ in ("advcl", "ccomp", "relcl", "acl", "csubj") for t in doc)
    if has_coord and has_sub:
        return "compound-complex"
    if has_coord:
        return "compound"
    if has_sub:
        return "complex"
    return "simple"


def syntax_flags(text: str, function_type: str, structure_type: str) -> Dict[str, str]:
    """Return review flags where spaCy's parse disagrees with the LLM labels."""
    if nlp is None or not text.strip():
        return {}
    doc = nlp(text)
    flags: Dict[str, str] = {}
    spacy_fn = _spacy_function_type(doc, text)
    if function_type and spacy_fn != function_type.lower():
        flags["function_type_disagreement"] = f"spacy={spacy_fn} llm={function_type}"
    spacy_st = _spacy_structure_type(doc)
    if structure_type and spacy_st != structure_type.lower():
        flags["structure_type_disagreement"] = f"spacy={spacy_st} llm={structure_type}"
    return flags
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/enrichment/test_syntax_check.py -q` → PASS (adjust the structure test sentence if the parse labels differently — the assertion targets a clearly compound-complex sentence; verify with a quick doc inspection if it fails, and pick a sentence that parses as intended rather than weakening the assertion).
- [ ] **Step 5: Commit** — `git add src/enrichment/syntax_check.py tests/enrichment/test_syntax_check.py && git commit -m "feat: spaCy cross-check flags for function/structure"`

---

### Task 8: AnalysisGenerated payload extension (dimension confidences + flags + provenance)

**Files:**
- Modify: `src/events/sentence_events.py` (`AnalysisGeneratedData`), `src/events/aggregates.py` (`Sentence.generate_analysis` + `_apply_analysis_generated`)
- Modify: `src/projections/handlers/sentence_handlers.py` (`AnalysisGeneratedHandler` stores the new properties)
- Test: `tests/events/test_analysis_payload_v2.py`

**Interfaces:**
- Produces (all optional/backward-compatible — old events still parse):
  - `AnalysisGeneratedData` gains `dimension_confidences: Dict[str, float] = {}`, `flags: Dict[str, str] = {}`, `provider: Optional[str] = None`.
  - `Sentence.generate_analysis(..., dimension_confidences=None, flags=None, provider=None)` passes them through (payload-model-validated); apply method stores `self.dimension_confidences`, `self.flags`, `self.analysis_provider`.
  - Projection: Sentence node gains `analysis_provider`, `dimension_confidences` (JSON string property — Neo4j maps can't nest), `flags` (JSON string), each only SET when present.

- [ ] **Step 1: Write the failing tests** (`tests/events/test_analysis_payload_v2.py`)

```python
import json

from src.events.aggregates import Sentence


def make_sentence():
    s = Sentence("11111111-1111-1111-1111-111111111111")
    s.create(interview_id="22222222-2222-2222-2222-222222222222", index=0, text="Hi.")
    return s


def test_generate_analysis_carries_v2_fields():
    s = make_sentence()
    event = s.generate_analysis(
        model="claude-3-haiku-20240307",
        model_version="1.0",
        classification={"purpose": "Statement"},
        dimension_confidences={"purpose": 0.85},
        flags={"function_type_disagreement": "spacy=interrogative llm=declarative"},
        provider="anthropic",
    )
    assert event.data["dimension_confidences"] == {"purpose": 0.85}
    assert event.data["provider"] == "anthropic"
    assert s.dimension_confidences["purpose"] == 0.85
    assert s.analysis_provider == "anthropic"


def test_v1_events_still_apply():
    s = make_sentence()
    event = s.generate_analysis(model="gpt", model_version="1.0", classification={"purpose": "Q"})
    assert event.data["dimension_confidences"] == {}
    replayed = Sentence("11111111-1111-1111-1111-111111111111")
    replayed.load_from_history(s.get_uncommitted_events())
    assert replayed.classification == {"purpose": "Q"}
    assert replayed.dimension_confidences == {}
```

- [ ] **Step 2: Run to verify fail** — TypeError on unexpected kwargs.

- [ ] **Step 3: Implement.** In `AnalysisGeneratedData` add:

```python
    dimension_confidences: Dict[str, float] = Field(
        default_factory=dict, description="Per-dimension numeric confidence (0-1)"
    )
    flags: Dict[str, str] = Field(
        default_factory=dict, description="Review flags (e.g., spaCy disagreement)"
    )
    provider: Optional[str] = Field(None, description="Provider that served the calls")
```

In `Sentence.__init__` add `self.dimension_confidences: Dict[str, float] = {}`, `self.flags: Dict[str, str] = {}`, `self.analysis_provider: Optional[str] = None`. In `_apply_analysis_generated` add the three `data.get(...)` lines. In `generate_analysis`, add the three keyword params (defaults None → `or {}`) and route the payload through `AnalysisGeneratedData(...).model_dump()` (matching the Layer 1 command-time-validation pattern) instead of the raw dict.

In `AnalysisGeneratedHandler.apply` (read the existing Cypher first), extend the SET clause with:

```cypher
            s.analysis_provider = coalesce($provider, s.analysis_provider),
            s.dimension_confidences = coalesce($dimension_confidences_json, s.dimension_confidences),
            s.flags = coalesce($flags_json, s.flags),
```

passing `provider=data.get("provider")`, `dimension_confidences_json=json.dumps(data["dimension_confidences"]) if data.get("dimension_confidences") else None`, and likewise for flags.

- [ ] **Step 4: Run to verify pass** — `pytest tests/events tests/projections -m "not integration" -q` → PASS, no regressions.
- [ ] **Step 5: Commit** — `git add src/events src/projections tests/events/test_analysis_payload_v2.py && git commit -m "feat: AnalysisGenerated v2 fields (dimension confidences, flags, provider)"`

---

### Task 9: New domain events — EntitiesExtracted, ClaimExtracted (+ projections, checklist)

**Files:**
- Modify: `src/events/sentence_events.py` (`EntitiesExtractedData`), `src/events/interview_events.py` (`ClaimExtractedData`), `src/events/aggregates.py` (command+apply methods)
- Create: `src/projections/handlers/entity_handlers.py`, `src/projections/handlers/claim_handlers.py`
- Modify: `src/projections/bootstrap.py`, `src/projections/config.py` (allowlists)
- Test: `tests/events/test_entity_claim_events.py`, `tests/projections/test_entity_claim_handlers.py`

**Interfaces:**
- Produces:
  - `EntitiesExtractedData`: `interview_id: Optional[str]` (lane key), `entities: List[Dict]` (each `{text, entity_type, start, end, confidence}` — validated upstream by `EntityMentionsResult`), `model: str`, `provider: str`. `Sentence.record_entities(entities: List[Dict], model: str, provider: str, **kw)` (guard: created); state `self.entities: List[Dict]` (replaced on re-extraction).
  - `ClaimExtractedData`: `claim_id: str`, `utterance_id: str`, `speaker_id: str`, `text: str`, `kind: str`, `confidence: float (0..1)`, `model: str`, `provider: str`. `Interview.record_claim(claim_id, utterance_id, text, kind, confidence, model, provider, **kw)` — speaker_id derived from `self.utterances[utterance_id]["speaker_id"]`; guards: created, utterance known + not removed, duplicate claim_id.
  - Projections: `EntitiesExtractedHandler` (Sentence stream): deletes the fragment's existing `MENTIONS` edges then `MERGE (e:Entity {surface: <lower(text)>, entity_type})` + `(s)-[:MENTIONS {text, start, end, confidence}]->(e)`; raises via zero-write guard (reuse `_raise_if_no_writes` from `speaker_handlers`) when the Sentence node is missing. `ClaimExtractedHandler` (Interview stream): `MERGE (c:Claim {claim_id})` SET text/kind/confidence/model + `MATCH speaker/utterance` for `MADE_BY` → Speaker and `SUPPORTED_BY` → each fragment of the utterance (`(s:Sentence)-[:PART_OF_UTTERANCE]->(u)`); returns `count(*)` and raises when speaker/utterance not yet projected.
  - Allowlists: `EntitiesExtracted` (+`EmbeddingGenerated` reserved for Task 10) on the sentence subscription; `ClaimExtracted` (+`UtteranceEmbeddingGenerated`) on the interview subscription. Registered in bootstrap; the existing drift test enforces the pairing (update its count comments only if it pins counts — it pins subset relation, so registration + allowlist must land together).

- [ ] **Step 1: Write the failing tests** — `tests/events/test_entity_claim_events.py`:

```python
import pytest

from src.events.aggregates import Interview, Sentence

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"
F1 = "77777777-7777-7777-7777-777777777771"
CLAIM = "88888888-8888-8888-8888-888888888881"

ENTITY = {"text": "Neo4j", "entity_type": "product", "start": 4, "end": 9, "confidence": 0.9}


def make_sentence():
    s = Sentence(F1)
    s.create(interview_id=IID, index=0, text="Use Neo4j here.")
    return s


def make_interview():
    i = Interview(IID)
    i.create(title="t.txt", source="s")
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    i.identify_utterance(U1, SP1, [F1], 0.9)
    return i


def test_record_entities_event_and_state():
    s = make_sentence()
    event = s.record_entities([ENTITY], model="haiku", provider="anthropic")
    assert event.event_type == "EntitiesExtracted"
    assert event.data["interview_id"] == IID  # lane routing key
    assert s.entities == [ENTITY]


def test_record_claim_derives_speaker_and_guards():
    i = make_interview()
    event = i.record_claim(CLAIM, U1, "We will ship Friday", "commitment", 0.8, "haiku", "anthropic")
    assert event.event_type == "ClaimExtracted"
    assert event.data["speaker_id"] == SP1
    with pytest.raises(ValueError, match="already recorded"):
        i.record_claim(CLAIM, U1, "dup", "assertion", 0.5, "haiku", "anthropic")
    with pytest.raises(ValueError, match="Unknown utterance"):
        i.record_claim("99999999-9999-9999-9999-999999999999", "no-such", "x", "assertion", 0.5, "haiku", "anthropic")


def test_claim_replay_reconstructs():
    i = make_interview()
    i.record_claim(CLAIM, U1, "We will ship Friday", "commitment", 0.8, "haiku", "anthropic")
    replayed = Interview(IID)
    replayed.load_from_history(i.get_uncommitted_events())
    assert CLAIM in replayed.claims
```

`tests/projections/test_entity_claim_handlers.py` (mock pattern as in `test_speaker_handlers.py`; the shared `mock_matched`-style helpers are redefined locally):

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.claim_handlers import ClaimExtractedHandler
from src.projections.handlers.entity_handlers import EntitiesExtractedHandler

IID = "22222222-2222-2222-2222-222222222222"
F1 = "77777777-7777-7777-7777-777777777771"


def make_event(event_type, aggregate_type, aggregate_id, data):
    return EventEnvelope(
        event_type=event_type, aggregate_type=aggregate_type,
        aggregate_id=aggregate_id, version=2, data=data,
    )


@pytest.mark.asyncio
async def test_entities_extracted_merges_entity_and_mentions():
    handler = EntitiesExtractedHandler()
    tx = AsyncMock()
    counters = MagicMock(nodes_created=1, properties_set=4, relationships_created=1)
    tx.run.return_value.consume = AsyncMock(return_value=MagicMock(counters=counters))
    event = make_event(
        "EntitiesExtracted", AggregateType.SENTENCE, F1,
        {"interview_id": IID, "model": "haiku", "provider": "anthropic",
         "entities": [{"text": "Neo4j", "entity_type": "product", "start": 4, "end": 9, "confidence": 0.9}]},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    assert "MENTIONS" in query and "MERGE (e:Entity" in query


@pytest.mark.asyncio
async def test_claim_extracted_raises_when_targets_missing():
    handler = ClaimExtractedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value=None)
    event = make_event(
        "ClaimExtracted", AggregateType.INTERVIEW, IID,
        {"claim_id": "c", "utterance_id": "u", "speaker_id": "sp",
         "text": "x", "kind": "assertion", "confidence": 0.8, "model": "m", "provider": "p"},
    )
    with pytest.raises(ValueError, match="not yet projected"):
        await handler.apply(tx, event)


def test_new_event_types_in_bootstrap_and_allowlists():
    from src.projections.bootstrap import create_handler_registry
    from src.projections.config import get_all_allowed_event_types

    registry = create_handler_registry(parked_events_manager=MagicMock())
    allowed = set(get_all_allowed_event_types())
    for event_type in ("EntitiesExtracted", "ClaimExtracted"):
        assert registry.has_handler(event_type)
        assert event_type in allowed
```

- [ ] **Step 2: Run to verify fail** — AttributeError/ModuleNotFoundError.

- [ ] **Step 3: Implement** exactly per the Interfaces block, following the Layer 1 patterns file-for-file: payload models with Field constraints; command methods validating through the payload model (`.model_dump()`); apply methods as pure state writes (`self.entities`, `self.claims: Dict[str, Dict]`); dispatch branches in `apply_event`. `EntitiesExtractedHandler` Cypher:

```cypher
        MATCH (s:Sentence {aggregate_id: $aggregate_id})
        OPTIONAL MATCH (s)-[old:MENTIONS]->(:Entity)
        DELETE old
        WITH DISTINCT s
        UNWIND $entities AS ent
        MERGE (e:Entity {surface: toLower(ent.text), entity_type: ent.entity_type})
        MERGE (s)-[m:MENTIONS]->(e)
        SET m.text = ent.text, m.start = ent.start, m.end = ent.end,
            m.confidence = ent.confidence
```

Guard: after `consume()`, reuse the `_raise_if_no_writes` helper (import from `.speaker_handlers`) — but only when `$entities` is non-empty; an empty extraction with an existing Sentence legitimately touches nothing except the deleted old edges, so run a separate first statement `MATCH (s:Sentence {aggregate_id:$aggregate_id}) SET s.entities_extracted_at = datetime($occurred_at)` and guard on THAT statement's zero-write instead (property always set → replay-safe, missing sentence → raise).

`ClaimExtractedHandler` Cypher (single statement, count-guarded like `UtteranceIdentifiedHandler`):

```cypher
        MATCH (sp:Speaker {speaker_id: $speaker_id})
        MATCH (u:Utterance {utterance_id: $utterance_id})
        MERGE (c:Claim {claim_id: $claim_id})
        SET c.text = $text, c.kind = $kind, c.confidence = $confidence,
            c.model = $model, c.provider = $provider, c.interview_id = $interview_id
        MERGE (c)-[:MADE_BY]->(sp)
        WITH c, u
        MATCH (s:Sentence)-[:PART_OF_UTTERANCE]->(u)
        MERGE (c)-[:SUPPORTED_BY]->(s)
        RETURN count(s) AS supported
```

`record.single()` None or `supported == 0` → raise `ValueError(f"ClaimExtracted {claim_id}: targets not yet projected")`.

Allowlists in `src/projections/config.py`: add `"EntitiesExtracted"` to the sentence list and `"ClaimExtracted"` to the interview list (with a `# Layer 2 (M4.2)` comment). Register both handlers in bootstrap. Update `tests/projections/test_bootstrap_unit.py` count pins (15 → 17) and expected-types set.

- [ ] **Step 4: Run to verify pass** — `pytest tests/events tests/projections -m "not integration" -q` → PASS including drift test.
- [ ] **Step 5: Commit** — `git add src/events src/projections tests && git commit -m "feat: EntitiesExtracted and ClaimExtracted events with projections"`

---

### Task 10: Embedder protocol + EmbeddingGenerated events + projection

**Files:**
- Create: `src/enrichment/embedder.py`
- Modify: `src/events/sentence_events.py` (`EmbeddingGeneratedData`), `src/events/interview_events.py` (`UtteranceEmbeddingGeneratedData`), `src/events/aggregates.py` (`Sentence.record_embedding`, `Interview.record_utterance_embedding`)
- Create: `src/projections/handlers/embedding_handlers.py`
- Modify: `src/projections/bootstrap.py`, `src/projections/config.py`, `config.yaml` (embeddings provider pin), `requirements.txt` (`sentence-transformers`)
- Test: `tests/enrichment/test_embedder.py`, `tests/events/test_embedding_events.py`, `tests/projections/test_embedding_handlers.py`

**Interfaces:**
- Produces:
  - `Embedder` Protocol: `async embed(texts: List[str]) -> List[List[float]]`; properties `model_name: str`, `dim: int`.
  - `OpenAIEmbedder(model="text-embedding-3-small", dim=1536)` — calls `client.embeddings.create(model=..., input=texts)`; `LocalEmbedder(model_name="all-MiniLM-L6-v2", dim=384)` — lazy-imports sentence_transformers (encode via `asyncio.to_thread`).
  - `get_embedder(config_dict=None) -> Embedder` — reads `config["embeddings"]["provider"]` (`"openai" | "local"`; **config-pinned, no failover** per spec); reuses the existing `embedding:` block's model/dim for local.
  - `encode_vector(vec: List[float]) -> str` / `decode_vector(s: str) -> List[float]` — base64 of little-endian float32 (`struct.pack(f"<{len(vec)}f", *vec)`).
  - Events: `EmbeddingGenerated` (Sentence stream: `interview_id`, `model`, `dim`, `vector_b64`); `UtteranceEmbeddingGenerated` (Interview stream: `utterance_id`, `model`, `dim`, `vector_b64`). Aggregate methods with created/known-utterance guards; state stores `{model, dim}` only (not the vector — keep aggregates light).
  - Projection `EmbeddingGeneratedHandler`: decodes and `SET s.embedding = $vector, s.embedding_model = $model, s.embedding_dim = $dim` (zero-write guard); `UtteranceEmbeddingGeneratedHandler` same on the Utterance node (count-guard). Vector index creation: handler-side `CREATE VECTOR INDEX fragment_embedding_<sanitized_model> IF NOT EXISTS FOR (s:Sentence) ON s.embedding OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}` executed once per handler instance (guarded by an instance flag; index name sanitization: non-alphanumerics → `_`).
- Config addition:

```yaml
embeddings:
  provider: "openai"        # "openai" | "local" — config-pinned; switch = re-embed via replay
  openai:
    model_name: "text-embedding-3-small"
    embedding_dim: 1536
```

- [ ] **Step 1: Write the failing tests** — `tests/enrichment/test_embedder.py`:

```python
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enrichment.embedder import decode_vector, encode_vector, get_embedder


def test_vector_roundtrip():
    vec = [0.1, -2.5, 3.25]
    decoded = decode_vector(encode_vector(vec))
    assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(vec, decoded))


def test_get_embedder_openai_pinned():
    cfg = {"embeddings": {"provider": "openai", "openai": {"model_name": "text-embedding-3-small", "embedding_dim": 1536}},
           "openai": {"api_key": "sk-test"}}
    embedder = get_embedder(cfg)
    assert embedder.model_name == "text-embedding-3-small"
    assert embedder.dim == 1536


def test_get_embedder_local_uses_existing_embedding_block():
    cfg = {"embeddings": {"provider": "local"},
           "embedding": {"model_name": "all-MiniLM-L6-v2", "embedding_dim": 384}}
    embedder = get_embedder(cfg)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.dim == 384


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="embeddings.provider"):
        get_embedder({"embeddings": {"provider": "mystery"}})


@pytest.mark.asyncio
async def test_openai_embedder_calls_api():
    from src.enrichment.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.model_name = "text-embedding-3-small"
    embedder.dim = 3
    embedder.client = MagicMock()
    item = MagicMock(embedding=[0.1, 0.2, 0.3])
    embedder.client.embeddings.create = AsyncMock(return_value=MagicMock(data=[item]))
    vectors = await embedder.embed(["hello"])
    assert vectors == [[0.1, 0.2, 0.3]]
```

`tests/events/test_embedding_events.py`:

```python
import pytest

from src.events.aggregates import Interview, Sentence

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"
F1 = "77777777-7777-7777-7777-777777777771"


def test_fragment_embedding_event_carries_lane_key():
    s = Sentence(F1)
    s.create(interview_id=IID, index=0, text="Hi.")
    event = s.record_embedding(model="text-embedding-3-small", dim=3, vector_b64="AAAA")
    assert event.event_type == "EmbeddingGenerated"
    assert event.data["interview_id"] == IID
    assert s.embedding_model == "text-embedding-3-small"


def test_utterance_embedding_guards_unknown_utterance():
    i = Interview(IID)
    i.create(title="t", source="s")
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    i.identify_utterance(U1, SP1, [F1], 0.9)
    event = i.record_utterance_embedding(U1, model="m", dim=3, vector_b64="AAAA")
    assert event.event_type == "UtteranceEmbeddingGenerated"
    with pytest.raises(ValueError, match="Unknown utterance"):
        i.record_utterance_embedding("no-such", model="m", dim=3, vector_b64="AAAA")
```

`tests/projections/test_embedding_handlers.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.embedding_handlers import EmbeddingGeneratedHandler


@pytest.mark.asyncio
async def test_embedding_written_with_model_tag():
    handler = EmbeddingGeneratedHandler()
    handler._index_ensured = True  # skip index DDL in unit test
    tx = AsyncMock()
    counters = MagicMock(nodes_created=0, properties_set=3, relationships_created=0)
    tx.run.return_value.consume = AsyncMock(return_value=MagicMock(counters=counters))
    from src.enrichment.embedder import encode_vector

    event = EventEnvelope(
        event_type="EmbeddingGenerated", aggregate_type=AggregateType.SENTENCE,
        aggregate_id="77777777-7777-7777-7777-777777777771", version=3,
        data={"interview_id": "22222222-2222-2222-2222-222222222222",
              "model": "text-embedding-3-small", "dim": 3,
              "vector_b64": encode_vector([0.1, 0.2, 0.3])},
    )
    await handler.apply(tx, event)
    params = tx.run.call_args[1]
    assert params["model"] == "text-embedding-3-small"
    assert len(params["vector"]) == 3


def test_embedding_events_in_bootstrap_and_allowlists():
    from src.projections.bootstrap import create_handler_registry
    from src.projections.config import get_all_allowed_event_types

    registry = create_handler_registry(parked_events_manager=MagicMock())
    allowed = set(get_all_allowed_event_types())
    for event_type in ("EmbeddingGenerated", "UtteranceEmbeddingGenerated"):
        assert registry.has_handler(event_type)
        assert event_type in allowed
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.

- [ ] **Step 3: Implement** per the Interfaces block. `src/enrichment/embedder.py` core:

```python
"""Embedding providers (config-pinned — never per-call failover).

Vectors from different models are incomparable; the provider is pinned in
config, every vector is tagged {model, dim}, and switching means re-running
the embedding extractor via event replay.
"""

import asyncio
import base64
import struct
from typing import Any, Dict, List, Optional, Protocol

from src.utils.logger import get_logger

logger = get_logger()


def encode_vector(vec: List[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")


def decode_vector(s: str) -> List[float]:
    raw = base64.b64decode(s.encode("ascii"))
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


class Embedder(Protocol):
    model_name: str
    dim: int

    async def embed(self, texts: List[str]) -> List[List[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model_name: str, dim: int, api_key: str):
        from openai import AsyncOpenAI

        self.model_name = model_name
        self.dim = dim
        self.client = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]


class LocalEmbedder:
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model = None  # lazy: sentence-transformers import is heavy

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        model = await asyncio.to_thread(self._load)
        vectors = await asyncio.to_thread(model.encode, texts)
        return [list(map(float, v)) for v in vectors]


def get_embedder(config_dict: Optional[Dict[str, Any]] = None) -> Embedder:
    from src.config import config as global_config

    cfg = config_dict if config_dict is not None else global_config
    embeddings_cfg = cfg.get("embeddings", {})
    provider = embeddings_cfg.get("provider", "openai")
    if provider == "openai":
        ocfg = embeddings_cfg.get("openai", {})
        return OpenAIEmbedder(
            model_name=ocfg.get("model_name", "text-embedding-3-small"),
            dim=ocfg.get("embedding_dim", 1536),
            api_key=cfg.get("openai", {}).get("api_key", ""),
        )
    if provider == "local":
        lcfg = cfg.get("embedding", {})  # reuse the pre-existing block
        return LocalEmbedder(
            model_name=lcfg.get("model_name", "all-MiniLM-L6-v2"),
            dim=lcfg.get("embedding_dim", 384),
        )
    raise ValueError(f"Unknown embeddings.provider: {provider!r} (openai|local)")
```

Events/aggregates/handlers follow the exact Layer 1/Task 9 patterns (payload model validation at command time; guards; zero-write raises; allowlists sentence:`EmbeddingGenerated`, interview:`UtteranceEmbeddingGenerated`; bootstrap registration; bootstrap-unit pins 17 → 19). Handler decodes with `decode_vector` and passes `vector` (list of floats) to Cypher. Index DDL runs once per handler instance before the first apply (separate `await tx.run(...)`; name `fragment_embedding_` / `utterance_embedding_` + model sanitized `re.sub(r"[^A-Za-z0-9]", "_", model)`). Add `sentence-transformers>=3.0.0` to `requirements.txt`.

- [ ] **Step 4: Run to verify pass** — `pytest tests/enrichment tests/events tests/projections -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit** — `git add src tests requirements.txt config.yaml && git commit -m "feat: embedder providers and embedding events with vector-index projection"`

---

### Task 11: Enrichment executor

**Files:**
- Create: `src/enrichment/executor.py`
- Test: `tests/enrichment/test_executor.py`

**Interfaces:**
- Consumes: `ExtractorSpec` (Task 6), `FragmentView`/`GraphContextBuilder` (Task 5), `FailoverAgent.call` (Task 3), response models (Task 4), `syntax_flags` (Task 7).
- Produces:
  - `FragmentEnrichment(BaseModel)`: `index: int`, `classification: Dict[str, Any]` (function_type/structure_type/purpose/topic_level_1/topic_level_3 values), `keywords: List[str]`, `domain_keywords: List[str]`, `dimension_confidences: Dict[str, float]`, `flags: Dict[str, str]`, `entities: List[Dict]`, `provider: str`, `model: str`.
  - `UtteranceEnrichment(BaseModel)`: `utterance_id: str`, `claims: List[Dict]` (validated ClaimItem dumps), `provider: str`, `model: str`.
  - `EnrichmentExecutor(agent: FailoverAgent, specs: List[ExtractorSpec], prompts: Dict, domain_keywords: List[str], concurrency: int = 10)` with:
    - `async enrich_fragments(fragments: List[FragmentView], contexts: List[Dict[str, str]]) -> List[FragmentEnrichment]` — for each fragment, one `agent.call` per fragment-scoped spec (schema = spec.resolve_model().model_json_schema()); prompt formatted with `sentence=frag.text`, `context=contexts[i][spec.context_needs[0]]` when needed, `domain_keywords=", ".join(...)` for that extractor; Pydantic-validates each response (invalid → dimension omitted + flag `"<name>_invalid_response"`); spaCy flags merged after function/structure resolve. Concurrency bounded by `asyncio.Semaphore`.
    - `async enrich_utterances(utterances: Dict[str, str]) -> List[UtteranceEnrichment]` — one `claims` call per utterance text.
  - Provenance: provider/model taken from the LAST CallResult per unit (all calls for a unit go to the same chain; mixed-provider units record the final one — acceptable, logged).

- [ ] **Step 1: Write the failing tests** (`tests/enrichment/test_executor.py`)

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.failover_agent import CallResult
from src.enrichment.executor import EnrichmentExecutor
from src.enrichment.graph_context import FragmentView
from src.enrichment.registry import ExtractorRegistry
from src.utils.helpers import load_yaml

RESPONSES = {
    "function_type": {"function_type": "interrogative", "confidence": 0.9},
    "structure_type": {"structure_type": "simple", "confidence": 0.8},
    "purpose": {"purpose": "Query", "confidence": 0.85},
    "topic_level_1": {"topic_level_1": "tools", "confidence": 0.7},
    "topic_level_3": {"topic_level_3": "processes", "confidence": 0.6},
    "overall_keywords": {"overall_keywords": ["audio"]},
    "domain_keywords": {"domain_keywords": []},
    "entity_mentions": {"entities": []},
    "claims": {"claims": [{"text": "We will ship", "kind": "commitment", "confidence": 0.8}]},
}


def make_agent():
    agent = MagicMock()

    async def call(prompt, schema=None):
        for key, resp in RESPONSES.items():
            marker = MARKERS[key]
            if marker in prompt:
                return CallResult(data=resp, provider="anthropic", model="haiku")
        raise AssertionError(f"No canned response for prompt: {prompt[:80]}")

    agent.call = AsyncMock(side_effect=call)
    return agent


# Each core prompt contains a distinguishing phrase; markers let the fake agent
# route without ordering assumptions.
MARKERS = {
    "function_type": "function type",
    "structure_type": "structure type",
    "purpose": "sentence's purpose",
    "topic_level_1": "immediate surrounding context",
    "topic_level_3": "broader context",
    "overall_keywords": "main topics of conversation",
    "domain_keywords": "domain-specific list",
    "entity_mentions": "entity mentions",
    "claims": "complete utterance",
}


def make_executor(agent):
    specs = ExtractorRegistry.load("config/extractors.yaml")
    prompts = load_yaml("prompts/core_extractors.yaml")
    return EnrichmentExecutor(agent, specs, prompts, domain_keywords=["ECU"], concurrency=4)


@pytest.mark.asyncio
async def test_enrich_fragment_collects_all_dimensions():
    executor = make_executor(make_agent())
    fragments = [FragmentView(index=0, text="Can you hear me?", speaker_handle="S1")]
    contexts = [{"immediate_context": "c", "observer_context": "c", "broader_context": "c",
                 "overall_context": "c", "utterance_context": "Can you hear me?"}]
    results = await executor.enrich_fragments(fragments, contexts)
    r = results[0]
    assert r.classification["purpose"] == "Query"
    assert r.dimension_confidences["purpose"] == 0.85
    assert r.keywords == ["audio"]
    assert r.provider == "anthropic"
    # spaCy agrees with interrogative/simple here -> no syntax flags
    assert "function_type_disagreement" not in r.flags


@pytest.mark.asyncio
async def test_invalid_response_flagged_not_fatal():
    agent = make_agent()
    bad = dict(RESPONSES)
    bad["purpose"] = {"purpose": "Query", "confidence": 5.0}  # out of range

    async def call(prompt, schema=None):
        for key, marker in MARKERS.items():
            if marker in prompt:
                return CallResult(data=bad[key], provider="anthropic", model="haiku")
        raise AssertionError

    agent.call = AsyncMock(side_effect=call)
    executor = make_executor(agent)
    fragments = [FragmentView(index=0, text="Can you hear me?", speaker_handle="S1")]
    contexts = [{"immediate_context": "c", "observer_context": "c", "broader_context": "c",
                 "overall_context": "c", "utterance_context": "x"}]
    results = await executor.enrich_fragments(fragments, contexts)
    assert "purpose" not in results[0].classification
    assert "purpose_invalid_response" in results[0].flags


@pytest.mark.asyncio
async def test_enrich_utterances_extracts_claims():
    executor = make_executor(make_agent())
    results = await executor.enrich_utterances({"u-1": "We will ship Friday. That is the complete utterance."})
    assert results[0].utterance_id == "u-1"
    assert results[0].claims[0]["kind"] == "commitment"
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.

- [ ] **Step 3: Implement** (`src/enrichment/executor.py`)

```python
"""Runs registered extractors over fragments and utterances.

One focused LLM call per dimension per unit (never merged), schema-enforced,
Pydantic-validated, concurrency-bounded. Invalid responses degrade to an
omitted dimension plus a review flag; they never fail the run.
"""

import asyncio
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError

from src.agents.failover_agent import FailoverAgent
from src.enrichment.graph_context import FragmentView
from src.enrichment.models import ExtractorSpec
from src.enrichment.syntax_check import syntax_flags
from src.utils.logger import get_logger

logger = get_logger()

_CLASSIFICATION_KEYS = {
    "function_type": "function_type",
    "structure_type": "structure_type",
    "purpose": "purpose",
    "topic_level_1": "topic_level_1",
    "topic_level_3": "topic_level_3",
}


class FragmentEnrichment(BaseModel):
    index: int
    classification: Dict[str, Any] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    domain_keywords: List[str] = Field(default_factory=list)
    dimension_confidences: Dict[str, float] = Field(default_factory=dict)
    flags: Dict[str, str] = Field(default_factory=dict)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    provider: str = ""
    model: str = ""


class UtteranceEnrichment(BaseModel):
    utterance_id: str
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    provider: str = ""
    model: str = ""


class EnrichmentExecutor:
    def __init__(
        self,
        agent: FailoverAgent,
        specs: List[ExtractorSpec],
        prompts: Dict[str, Any],
        domain_keywords: List[str],
        concurrency: int = 10,
    ):
        self.agent = agent
        self.fragment_specs = [s for s in specs if s.scope == "fragment"]
        self.utterance_specs = [s for s in specs if s.scope == "utterance"]
        self.prompts = prompts
        self.domain_keywords = domain_keywords
        self.semaphore = asyncio.Semaphore(concurrency)

    def _format_prompt(self, spec: ExtractorSpec, text: str, context: Dict[str, str]) -> str:
        template = self.prompts[spec.prompt_key]["prompt"]
        kwargs: Dict[str, str] = {"sentence": text}
        if spec.context_needs:
            kwargs["context"] = context.get(spec.context_needs[0], "")
        if spec.name == "domain_keywords":
            kwargs["domain_keywords"] = ", ".join(self.domain_keywords)
        return template.format(**kwargs)

    async def _run_spec(self, spec: ExtractorSpec, text: str, context: Dict[str, str]):
        async with self.semaphore:
            prompt = self._format_prompt(spec, text, context)
            schema = spec.resolve_model().model_json_schema()
            return await self.agent.call(prompt, schema=schema)

    async def enrich_fragments(
        self, fragments: List[FragmentView], contexts: List[Dict[str, str]]
    ) -> List[FragmentEnrichment]:
        return list(
            await asyncio.gather(
                *(self._enrich_fragment(f, contexts[f.index]) for f in fragments)
            )
        )

    async def _enrich_fragment(
        self, frag: FragmentView, context: Dict[str, str]
    ) -> FragmentEnrichment:
        out = FragmentEnrichment(index=frag.index)
        results = await asyncio.gather(
            *(self._run_spec(spec, frag.text, context) for spec in self.fragment_specs)
        )
        for spec, call_result in zip(self.fragment_specs, results):
            out.provider, out.model = call_result.provider, call_result.model
            try:
                parsed = spec.resolve_model().model_validate(call_result.data)
            except ValidationError as e:
                logger.warning(f"{spec.name}: invalid response ({e.error_count()} errors)")
                out.flags[f"{spec.name}_invalid_response"] = "validation failed"
                continue
            data = parsed.model_dump()
            if spec.name in _CLASSIFICATION_KEYS:
                out.classification[spec.name] = data[_CLASSIFICATION_KEYS[spec.name]]
                out.dimension_confidences[spec.name] = data["confidence"]
            elif spec.name == "overall_keywords":
                out.keywords = data["overall_keywords"]
            elif spec.name == "domain_keywords":
                out.domain_keywords = data["domain_keywords"]
            elif spec.name == "entity_mentions":
                out.entities = data["entities"]
        out.flags.update(
            syntax_flags(
                frag.text,
                out.classification.get("function_type", ""),
                out.classification.get("structure_type", ""),
            )
        )
        return out

    async def enrich_utterances(
        self, utterance_texts: Dict[str, str]
    ) -> List[UtteranceEnrichment]:
        async def one(uid: str, text: str) -> UtteranceEnrichment:
            out = UtteranceEnrichment(utterance_id=uid)
            for spec in self.utterance_specs:
                call_result = await self._run_spec(spec, text, {})
                out.provider, out.model = call_result.provider, call_result.model
                try:
                    parsed = spec.resolve_model().model_validate(call_result.data)
                except ValidationError:
                    logger.warning(f"{spec.name}: invalid response for utterance {uid}")
                    continue
                out.claims = [c for c in parsed.model_dump()["claims"]]
            return out

        return list(
            await asyncio.gather(*(one(uid, text) for uid, text in utterance_texts.items()))
        )
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/enrichment -q` → PASS.
- [ ] **Step 5: Commit** — `git add src/enrichment/executor.py tests/enrichment/test_executor.py && git commit -m "feat: enrichment executor (focused calls, validation, syntax flags)"`

---

### Task 12: EnrichmentOrchestrator + CLI + ingestion `--enrich`

**Files:**
- Create: `src/enrichment/orchestrator.py`, `src/enrichment/__main__.py`
- Modify: `src/ingestion/__main__.py` (`--enrich` flag)
- Test: `tests/enrichment/test_orchestrator.py`

**Interfaces:**
- Consumes: repositories (`get_interview_repository`, `get_sentence_repository`), aggregates (Layer 1 + Tasks 8-10 methods), `GraphContextBuilder`, `EnrichmentExecutor`, `get_failover_agent`, `get_embedder`, `ExtractorRegistry`.
- Produces:
  - `EnrichmentResult(BaseModel)`: `interview_id: str`, `fragments_enriched: int`, `fragments_skipped: int` (already analyzed — resume), `entities_extracted: int`, `claims_extracted: int`, `embeddings_generated: int`.
  - `EnrichmentOrchestrator(config_dict=None)` with `async enrich_interview(interview_id: str, correlation_id=None, force: bool = False) -> EnrichmentResult`:
    1. Load Interview aggregate → speakers (handle by speaker_id), utterances (fragment membership), `metadata["fragment_count"]`.
    2. Load each Sentence aggregate (uuid5 `interview:index`, 0..fragment_count-1); build `FragmentView`s (`speaker_handle` from interview.speakers via sentence.speaker_id; `"S?"` when unattributed). Skip fragments whose `analysis_model` is already set unless `force` (resume-awareness; skipped count reported).
    3. Contexts via `GraphContextBuilder(config["preprocessing"]["context_windows"] + utterance texts joined from member fragments in order)`.
    4. `executor.enrich_fragments` → per fragment: `sentence.generate_analysis(model=..., model_version="m4.2", classification=..., keywords=..., domain_keywords=..., confidence=mean(dimension_confidences.values()) if any, dimension_confidences=..., flags=..., provider=...)` + `sentence.record_entities(...)` when entities present; save each sentence.
    5. `executor.enrich_utterances` → `interview.record_claim(uuid5(f"{interview_id}:claim:{utterance_id}:{ordinal}"), ...)` per claim.
    6. Embeddings: `embedder.embed([fragment texts])` batch → `sentence.record_embedding(model, dim, encode_vector(v))` per fragment (saved with the same sentence save when possible — order operations so each sentence is saved once); utterance texts → `interview.record_utterance_embedding(...)`; single interview save at the end.
    7. All events: `Actor(SYSTEM, user_id="enrichment")`, shared correlation_id.
  - CLI: `python -m src.enrichment <interview_id> [--force]` printing `EnrichmentResult` JSON; `python -m src.ingestion <file> --enrich` chains ingest → enrich and prints both results.

- [ ] **Step 1: Write the failing test** (`tests/enrichment/test_orchestrator.py`) — mocked repos + executor + embedder; asserts event flow and resume behavior:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enrichment.executor import FragmentEnrichment, UtteranceEnrichment
from src.enrichment.orchestrator import EnrichmentOrchestrator
from src.events.aggregates import Interview, Sentence

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"


def build_world():
    import uuid as uuid_mod

    interview = Interview(IID)
    interview.create(title="t.txt", source="s", metadata={"fragment_count": 2})
    interview.add_speaker(SP1, "S1", "S1", True, 0.9, "inference")
    f_ids = [str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{IID}:{i}")) for i in range(2)]
    sentences = []
    for i, fid in enumerate(f_ids):
        s = Sentence(fid)
        s.create(interview_id=IID, index=i, text=f"Fragment {i}.")
        s.attribute_speaker(SP1, 0.9, "inference")
        sentences.append(s)
    interview.identify_utterance(U1, SP1, f_ids, 0.9)
    interview.mark_events_as_committed()
    for s in sentences:
        s.mark_events_as_committed()
    return interview, {s.aggregate_id: s for s in sentences}


@pytest.mark.asyncio
async def test_enrich_interview_emits_analysis_entities_claims_embeddings(tmp_path):
    interview, sentences = build_world()

    interview_repo = MagicMock()
    interview_repo.load = AsyncMock(return_value=interview)
    interview_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    sentence_repo = MagicMock()
    sentence_repo.load = AsyncMock(side_effect=lambda sid: sentences.get(sid))
    sentence_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())

    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(return_value=[
        FragmentEnrichment(index=0, classification={"purpose": "Query"},
                           dimension_confidences={"purpose": 0.9},
                           entities=[{"text": "ECU", "entity_type": "product", "start": 0, "end": 3, "confidence": 0.9}],
                           provider="anthropic", model="haiku"),
        FragmentEnrichment(index=1, classification={"purpose": "Statement"},
                           dimension_confidences={"purpose": 0.8}, provider="anthropic", model="haiku"),
    ])
    executor.enrich_utterances = AsyncMock(return_value=[
        UtteranceEnrichment(utterance_id=U1,
                            claims=[{"text": "We ship Friday", "kind": "commitment", "confidence": 0.8}],
                            provider="anthropic", model="haiku"),
    ])
    embedder = MagicMock(model_name="text-embedding-3-small", dim=3)
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_sentence_repository", return_value=sentence_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        orchestrator = EnrichmentOrchestrator()
        result = await orchestrator.enrich_interview(IID)

    assert result.fragments_enriched == 2
    assert result.entities_extracted == 1
    assert result.claims_extracted == 1
    assert result.embeddings_generated == 3  # 2 fragments + 1 utterance
    first = sentences[list(sentences)[0]]
    assert first.classification["purpose"] == "Query"
    assert first.embedding_model == "text-embedding-3-small"
    assert len(interview.claims) == 1


@pytest.mark.asyncio
async def test_resume_skips_already_analyzed():
    interview, sentences = build_world()
    first = sentences[list(sentences)[0]]
    first.generate_analysis(model="haiku", model_version="m4.2", classification={"purpose": "Q"})
    first.mark_events_as_committed()

    interview_repo = MagicMock()
    interview_repo.load = AsyncMock(return_value=interview)
    interview_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    sentence_repo = MagicMock()
    sentence_repo.load = AsyncMock(side_effect=lambda sid: sentences.get(sid))
    sentence_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())

    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(return_value=[
        FragmentEnrichment(index=1, classification={"purpose": "Statement"},
                           dimension_confidences={"purpose": 0.8}, provider="anthropic", model="haiku"),
    ])
    executor.enrich_utterances = AsyncMock(return_value=[])
    embedder = MagicMock(model_name="m", dim=3)
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_sentence_repository", return_value=sentence_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        orchestrator = EnrichmentOrchestrator()
        result = await orchestrator.enrich_interview(IID)

    assert result.fragments_skipped == 1
    assert result.fragments_enriched == 1
    # executor only received the un-analyzed fragment
    passed_fragments = executor.enrich_fragments.call_args.args[0]
    assert [f.index for f in passed_fragments] == [1]
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.

- [ ] **Step 3: Implement** `src/enrichment/orchestrator.py` per the Interfaces block (structure mirrors `src/ingestion/orchestrator.py`: private helpers `_load_world`, `_build_executor` (constructs FailoverAgent+registry+prompts; patchable seam for tests), `_emit_fragment_results`, `_emit_claims`, `_emit_embeddings`; `EnrichmentResult` model; Actor SYSTEM "enrichment"; correlation_id via `generate_correlation_id()`). `src/enrichment/__main__.py` mirrors `src/ingestion/__main__.py` (argparse: `interview_id`, `--force`). In `src/ingestion/__main__.py` add `--enrich` flag: after ingestion, construct `EnrichmentOrchestrator` and run `enrich_interview(result.interview_id)`, print both JSON results.

Utterance text for context/claims: join member fragments' texts in `fragment_ids` order with a single space (fragment texts come from the loaded Sentence aggregates keyed by deterministic UUID).

- [ ] **Step 4: Run to verify pass** — `pytest tests/enrichment -q` → PASS; then full unit suite → no regressions.
- [ ] **Step 5: Commit** — `git add src/enrichment src/ingestion/__main__.py tests/enrichment/test_orchestrator.py && git commit -m "feat: enrichment orchestrator, CLI, and ingest --enrich chaining"`

---

### Task 13: API + Celery rewire

**Files:**
- Modify: `src/api/routers/analysis.py` (trigger ingest+enrich instead of run_pipeline), `src/tasks.py` (celery task)
- Test: `tests/api/test_analysis_rewire.py`

**Interfaces:**
- `POST /analysis/` request model unchanged externally where possible (read `src/api/routers/analysis.py` first; it takes a filename/input dir). New behavior: background task runs `IngestionOrchestrator.ingest_file(path)` then `EnrichmentOrchestrator.enrich_interview(interview_id)`. Response gains `interview_id: null` (still 202-style accepted; the background task logs the id — API contract note in docstring).
- `src/tasks.py`: `run_pipeline_for_file` renamed logic → same task name kept for queue compat, body swapped to ingest+enrich via `asyncio.run`.

- [ ] **Step 1: Write the failing test** — patch both orchestrators; POST `/analysis/` with a tmp file; assert 202-equivalent response and that background task invokes ingest then enrich (FastAPI `BackgroundTasks` runs after response in TestClient — assert via the patched mocks' await counts after request completes):

```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


def test_trigger_analysis_runs_ingest_then_enrich(tmp_path: Path):
    input_file = tmp_path / "meeting.txt"
    input_file.write_text("Alice: Hi.\nBob: Hello.\n")

    ingest_result = MagicMock(interview_id="abc")
    with patch("src.api.routers.analysis.IngestionOrchestrator") as ingest_cls, \
         patch("src.api.routers.analysis.EnrichmentOrchestrator") as enrich_cls, \
         patch("src.api.routers.analysis.config", {"paths": {"input_dir": str(tmp_path), "map_dir": str(tmp_path)}}):
        ingest_cls.return_value.ingest_file = AsyncMock(return_value=ingest_result)
        enrich_cls.return_value.enrich_interview = AsyncMock(return_value=MagicMock())
        client = TestClient(app)
        resp = client.post("/analysis/", json={"input_filename": "meeting.txt"})

    assert resp.status_code in (200, 202)
    ingest_cls.return_value.ingest_file.assert_awaited_once()
    enrich_cls.return_value.enrich_interview.assert_awaited_once_with("abc")
```

(Adjust the request body key to the router's actual schema after reading it — the brief's implementer MUST read `src/api/routers/analysis.py` first and keep the existing request model.)

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — replace the `run_pipeline` import/scheduling in the router's background function with an async runner calling the two orchestrators sequentially; same pattern in `src/tasks.py` core function (keep task name and signature; body becomes `asyncio.run(_ingest_and_enrich(...))`). Delete now-dead `run_pipeline` imports from both.
- [ ] **Step 4: Run** — `pytest tests/api tests -m "not integration" -q` → PASS (pipeline tests still pass — pipeline.py still exists until Task 14).
- [ ] **Step 5: Commit** — `git add src/api src/tasks.py tests/api/test_analysis_rewire.py && git commit -m "feat: rewire analysis API and celery task to ingest+enrich"`

---

### Task 14: Parity check + legacy retirement

**Files:**
- Delete: `src/pipeline.py`, `src/agents/sentence_analyzer.py`, `src/agents/context_builder.py`, `src/services/analysis_service.py` (+ empty `src/services/` if nothing remains), `src/pipeline_event_emitter.py`, `src/io/local_storage.py` analysis/map writer classes **only if unused elsewhere** (verify with grep; `LocalTextDataSource` may still be referenced — keep whatever ingestion uses), `src/models/llm_responses.py` (superseded by extractor_responses — verify no remaining imports)
- Delete tests: `tests/pipeline/` (entire dir), `tests/agents/test_sentence_analyzer*.py`, `tests/agents/test_context_builder*.py`, `tests/services/`, pipeline-related `tests/test_main_cli.py` cases, `tests/test_tasks.py` cases asserting run_pipeline internals (rewrite the task test to assert ingest+enrich delegation — Task 13's test already covers the router)
- Modify: `src/main.py` (drop `--run-pipeline` CLI path / `run_pipeline` import), `Makefile` (`run-pipeline` target → `ingest` target: `$(PYTHON) -m src.ingestion $(FILE) --enrich`), `config.yaml` (delete dead `classification.local.prompt_files` block only if nothing reads it — `SentenceAnalyzer` was the reader; verify)
- Test: parity is checked by script before deletion (Step 1), suite green after (Step 5)

- [ ] **Step 1: Parity check (before deleting anything).** Run the new path end-to-end against the golden fixture with mocked LLM (deterministic) and assert dimension coverage matches the legacy contract — write `tests/enrichment/test_parity.py`:

```python
"""Parity: the registry path produces every dimension the legacy pipeline did."""

LEGACY_DIMENSIONS = {
    "function_type", "structure_type", "purpose", "topic_level_1", "topic_level_3",
}


def test_registry_covers_all_legacy_dimensions():
    from src.enrichment.registry import ExtractorRegistry

    names = {s.name for s in ExtractorRegistry.load("config/extractors.yaml")}
    assert LEGACY_DIMENSIONS <= names
    assert {"overall_keywords", "domain_keywords"} <= names


def test_analysis_event_shape_superset_of_legacy():
    from src.events.sentence_events import AnalysisGeneratedData

    fields = set(AnalysisGeneratedData.model_fields)
    legacy = {"model", "version", "classification", "keywords", "topics",
              "domain_keywords", "confidence", "raw_ref"}
    assert legacy <= fields
```

Run it green, commit: `git commit -m "test: legacy parity assertions"`.

- [ ] **Step 2: Inventory references before deletion.**

Run: `command grep -rln "run_pipeline\|sentence_analyzer\|SentenceAnalyzer\|analysis_service\|AnalysisService\|pipeline_event_emitter\|context_builder\|llm_responses" src tests --include="*.py" 2>/dev/null || command grep -rln "run_pipeline" src tests`
Every hit must be in the deletion list or fixed in this task. Anything unexpected → STOP and report DONE_WITH_CONCERNS with the list.

- [ ] **Step 3: Delete the files and fix the survivors** (main.py CLI path, Makefile target, config cleanup, `tests/test_tasks.py` rewrite asserting the celery task calls ingest+enrich).

- [ ] **Step 4: Run the FULL unit suite** — `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests -m "not integration" -q`
Expected: green with a substantially smaller test count (~200 legacy tests removed). ZERO failures tolerated — a failure means a missed reference; fix, don't skip.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor!: retire legacy pipeline (registry is the only enrichment path)"
```

---

### Task 15: Integration smoke + docs

**Files:**
- Create: `tests/integration/test_layer2_enrichment_smoke.py`
- Modify: `docs/ROADMAP.md` (M4.2 milestone complete + decision log), `docs/architecture/database-schema.md` (Entity/Claim nodes, MENTIONS/MADE_BY/SUPPORTED_BY, embedding properties + vector indexes, new event types), `README.md` (What-It-Does step for enrichment; command examples)

**Interfaces:** none new.

- [ ] **Step 1: Integration smoke** (`tests/integration/test_layer2_enrichment_smoke.py`) — mirrors `test_layer1_projection_smoke.py`: ingest the labeled mini-transcript, run `EnrichmentOrchestrator` with the **executor mocked** (canned FragmentEnrichment/UtteranceEnrichment — no live LLM) and the **local embedder** (if sentence-transformers unavailable in env, mock embedder with fixed vectors), replay all events through the registry in `occurred_at` order against real Neo4j, then assert:

```cypher
MATCH (s:Sentence)-[:MENTIONS]->(e:Entity) ...
MATCH (c:Claim)-[:MADE_BY]->(sp:Speaker) ...
MATCH (s:Sentence) WHERE s.embedding IS NOT NULL ...
```

(counts per the canned enrichments; embedding_model tag asserted). Marked `@pytest.mark.integration`. Note for the runner: from a worktree, set `COMPOSE_PROJECT_NAME=interview_analyzer_chaining`; connection env per Layer 1 smoke test.

- [ ] **Step 2: Run it** against live infra (`make test-infra-up` first): expected PASS.
- [ ] **Step 3: Docs.** ROADMAP: M4.2 section (checklist mirroring Tasks 1–15), Quick Status table (M4.2 ✅, M3.1 folded into M4.2 — mark it ✅ with a pointer, M3.2 note that structured outputs landed without the SDK bump), Current Phase → M4.3 planning, decision log entries (provider chain, embeddings-as-events, legacy retired). database-schema.md: Entity/Claim node tables + relationships + embedding properties + vector index naming + 6 new event types in the events table. README: replace pipeline commands with `python -m src.ingestion <file> --enrich`, update the What-It-Does enrichment step, note the provider chain.
- [ ] **Step 4: Full unit suite green** one final time.
- [ ] **Step 5: Commit** — `git add tests/integration docs README.md && git commit -m "test+docs: Layer 2 enrichment smoke, M4.2 milestone docs"`

---

## Verification (whole-plan)

1. Full unit suite green: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests -m "not integration" -q`.
2. Integration: `COMPOSE_PROJECT_NAME=interview_analyzer_chaining make test-infra-up` then run both smoke tests + full `-m integration` (live-LLM tests require provider credit — Anthropic is primary now).
3. Manual e2e (needs ESDB+Neo4j+Anthropic key): `python -m src.ingestion data/input/GMT20231026-210203_Recording.txt --enrich`, run projection service, then in Neo4j: `MATCH (c:Claim)-[:MADE_BY]->(sp:Speaker) RETURN sp.handle, c.kind, c.text LIMIT 20`.
4. Drift guard green (`test_all_registered_event_types_are_subscription_allowed`).

## Deferred / explicitly out of scope for M4.2

- Segment/episode extractors (`scope: "segment"` is reserved in the Literal but no segmenter ships — no consumer yet).
- Entity canonicalization/resolution (Layer 4; surface-form MERGE by lowercase is the v1 key).
- Lens engine (`LensExtractionGenerated`) — M4.3; the registry's declared-extractor shape is its foundation.
- Batch/prompt-caching provider optimizations (`supports_batch_api` hooks exist; wiring them is post-M4.3 cost work).
- GraphRAG retrievers over the new vector indexes (Layer 5 era).
