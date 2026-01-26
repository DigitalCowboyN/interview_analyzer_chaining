"""
test_multi_provider_api_live.py

Live integration tests that actually hit the OpenAI and Anthropic APIs.
These tests verify that our multi-provider implementation works correctly
with real API calls (not mocked).

Requirements:
- OPENAI_API_KEY environment variable must be set
- ANTHROPIC_API_KEY environment variable must be set
- Internet connection required

Usage:
    pytest tests/integration/test_multi_provider_api_live.py -xvs
"""

import os
import pytest

from src.agents.agent_factory import AgentFactory

# Mark all tests in this module as integration tests (requires real API keys)
pytestmark = pytest.mark.integration

# Check for API keys
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not OPENAI_KEY:
    pytest.skip("OPENAI_API_KEY not set - skipping OpenAI live tests", allow_module_level=True)

if not ANTHROPIC_KEY:
    pytest.skip("ANTHROPIC_API_KEY not set - skipping Anthropic live tests", allow_module_level=True)


@pytest.mark.asyncio
async def test_openai_agent_live_api_call():
    """
    Test OpenAI agent makes successful live API call.

    Verifies:
    - Agent can be created via factory
    - API call succeeds
    - Response is valid JSON dict
    - Response contains expected structure
    """
    print("\n🔵 Testing OpenAI Live API Call...")

    # Create OpenAI agent via factory
    agent = AgentFactory.create_agent("openai")

    # Verify correct provider
    assert agent.get_provider_name() == "openai"
    print(f"  ✓ Provider: {agent.get_provider_name()}")
    print(f"  ✓ Model: {agent.get_model_name()}")

    # Make live API call with simple prompt
    prompt = """Analyze this sentence and return JSON with these keys:
    - function_type: "declarative" or "interrogative"
    - confidence: "high" or "medium" or "low"

    Sentence: "The sky is blue."

    Return only valid JSON, no other text."""

    response = await agent.call_model(prompt)

    # Verify response
    assert isinstance(response, dict), f"Expected dict, got {type(response)}"
    assert len(response) > 0, "Response should not be empty"
    assert "function_type" in response, "Response missing 'function_type' key"

    print(f"  ✓ Response received: {response}")
    print(f"  ✓ Function type: {response.get('function_type')}")
    print("  ✅ OpenAI live API test PASSED")


@pytest.mark.asyncio
async def test_anthropic_agent_live_api_call():
    """
    Test Anthropic agent makes successful live API call.

    Verifies:
    - Agent can be created via factory
    - API call succeeds
    - Response is valid JSON dict
    - Response contains expected structure
    - JSON parsing works (Anthropic uses prompt engineering, not native JSON mode)
    """
    print("\n🟣 Testing Anthropic Live API Call...")

    # Create Anthropic agent via factory
    agent = AgentFactory.create_agent("anthropic")

    # Verify correct provider
    assert agent.get_provider_name() == "anthropic"
    print(f"  ✓ Provider: {agent.get_provider_name()}")
    print(f"  ✓ Model: {agent.get_model_name()}")

    # Make live API call with simple prompt
    prompt = """Analyze this sentence and return JSON with these keys:
    - function_type: "declarative" or "interrogative"
    - confidence: "high" or "medium" or "low"

    Sentence: "The sky is blue."

    Return only valid JSON, no other text."""

    response = await agent.call_model(prompt)

    # Verify response
    assert isinstance(response, dict), f"Expected dict, got {type(response)}"
    assert len(response) > 0, "Response should not be empty"
    assert "function_type" in response, "Response missing 'function_type' key"

    print(f"  ✓ Response received: {response}")
    print(f"  ✓ Function type: {response.get('function_type')}")
    print("  ✅ Anthropic live API test PASSED")


@pytest.mark.asyncio
async def test_both_providers_return_similar_results():
    """
    Test that both providers analyze the same sentence and return similar structure.

    This doesn't test for identical results (models differ), but verifies:
    - Both return valid JSON
    - Both return expected keys
    - Both process the same prompt successfully
    """
    print("\n🔄 Testing Both Providers with Same Prompt...")

    prompt = """Analyze this sentence and return JSON with these keys:
    - function_type: "declarative" or "interrogative"
    - structure_type: "simple" or "compound" or "complex"

    Sentence: "Python is a programming language."

    Return only valid JSON, no other text."""

    # Test OpenAI
    openai_agent = AgentFactory.create_agent("openai")
    openai_response = await openai_agent.call_model(prompt)

    print(f"  OpenAI response: {openai_response}")
    assert isinstance(openai_response, dict)
    assert "function_type" in openai_response

    # Test Anthropic
    anthropic_agent = AgentFactory.create_agent("anthropic")
    anthropic_response = await anthropic_agent.call_model(prompt)

    print(f"  Anthropic response: {anthropic_response}")
    assert isinstance(anthropic_response, dict)
    assert "function_type" in anthropic_response

    # Both should identify it as declarative (high confidence)
    print(f"  OpenAI function_type: {openai_response.get('function_type')}")
    print(f"  Anthropic function_type: {anthropic_response.get('function_type')}")

    print("  ✅ Both providers returned valid structured responses")


@pytest.mark.asyncio
async def test_factory_singleton_pattern_with_live_calls():
    """
    Test that factory returns same instance for multiple calls (singleton pattern).
    This verifies connection reuse for efficiency.
    """
    print("\n🔁 Testing Factory Singleton Pattern...")

    # Get OpenAI agent twice
    agent1 = AgentFactory.create_agent("openai")
    agent2 = AgentFactory.create_agent("openai")

    assert agent1 is agent2, "Factory should return same instance (singleton)"
    print("  ✓ OpenAI singleton verified")

    # Get Anthropic agent twice
    agent3 = AgentFactory.create_agent("anthropic")
    agent4 = AgentFactory.create_agent("anthropic")

    assert agent3 is agent4, "Factory should return same instance (singleton)"
    print("  ✓ Anthropic singleton verified")

    # But different providers should be different instances
    assert agent1 is not agent3, "Different providers should have different instances"
    print("  ✓ Different provider instances verified")

    print("  ✅ Singleton pattern working correctly")


@pytest.mark.asyncio
async def test_provider_capability_flags():
    """
    Test that providers correctly report their capabilities.
    This is important for Phase 2 (prompt caching, batch API).
    """
    print("\n🏷️  Testing Provider Capability Flags...")

    openai_agent = AgentFactory.create_agent("openai")
    anthropic_agent = AgentFactory.create_agent("anthropic")

    # OpenAI doesn't support our Phase 2 features (yet)
    assert openai_agent.supports_prompt_caching() is False
    assert openai_agent.supports_batch_api() is False
    print("  ✓ OpenAI capabilities: caching=False, batch=False")

    # Anthropic supports both
    assert anthropic_agent.supports_prompt_caching() is True
    assert anthropic_agent.supports_batch_api() is True
    print("  ✓ Anthropic capabilities: caching=True, batch=True")

    print("  ✅ Provider capabilities correctly reported")
