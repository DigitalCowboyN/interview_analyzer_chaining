import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.claude_code_agent import ClaudeCodeAgent


def make_agent():
    return ClaudeCodeAgent(
        config_dict={"claude_code": {"command": "claude", "model": "haiku", "timeout_seconds": 5}}
    )


def cli_envelope(result_str):
    return json.dumps({"type": "result", "result": result_str}).encode()


def make_proc(stdout=b"", stderr=b"", returncode=0):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_parses_result_json():
    agent = make_agent()
    proc = make_proc(stdout=cli_envelope('{"purpose": "Query", "confidence": 0.8}'))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        result = await agent.call_model("classify this")
    assert result == {"purpose": "Query", "confidence": 0.8}
    argv = spawn.call_args.args
    assert argv[0] == "claude" and "-p" in argv and "--output-format" in argv


@pytest.mark.asyncio
async def test_strips_code_fences():
    agent = make_agent()
    fenced = "```json\n{\"purpose\": \"Query\", \"confidence\": 0.8}\n```"
    proc = make_proc(stdout=cli_envelope(fenced))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        result = await agent.call_model("classify this")
    assert result["purpose"] == "Query"


@pytest.mark.asyncio
async def test_schema_appends_contract_to_prompt():
    agent = make_agent()
    proc = make_proc(stdout=cli_envelope("{}"))
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        await agent.call_model("classify this", schema=schema)
    prompt_arg = spawn.call_args.args[2]  # claude -p <prompt>
    assert "Respond ONLY with JSON matching this schema" in prompt_arg


@pytest.mark.asyncio
async def test_nonzero_exit_raises():
    agent = make_agent()
    proc = make_proc(stderr=b"boom", returncode=1)
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(RuntimeError, match="claude CLI failed"):
            await agent.call_model("classify this")


@pytest.mark.asyncio
async def test_unparseable_result_returns_empty():
    agent = make_agent()
    proc = make_proc(stdout=cli_envelope("this is not json"))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        result = await agent.call_model("classify this")
    assert result == {}


def test_registered_in_factory():
    from src.agents.agent_factory import AgentFactory

    AgentFactory._providers = {}  # force re-init
    assert "claude_code" in AgentFactory.get_available_providers()


def test_provider_and_model_names():
    agent = make_agent()
    assert agent.get_provider_name() == "claude_code"
    assert agent.get_model_name() == "claude-code/haiku"
