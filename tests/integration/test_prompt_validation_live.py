"""
test_prompt_validation_live.py

Live integration tests that validate actual prompts work with both OpenAI and Anthropic.

These tests use real API calls (not mocks) to verify:
- Production prompts from task_prompts.yaml work with both providers
- Complex multi-context prompts are handled correctly
- JSON responses are parsed successfully
- Both providers return compatible response structures

Requirements:
- OPENAI_API_KEY environment variable must be set
- ANTHROPIC_API_KEY environment variable must be set
- Internet connection required

Usage:
    pytest tests/integration/test_prompt_validation_live.py -xvs
"""

import os
from pathlib import Path

import pytest
import yaml

from src.agents.agent_factory import AgentFactory

# Mark all tests in this module as integration tests (requires real API keys)
pytestmark = pytest.mark.integration

# Check for API keys
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Skip entire module if keys are missing
if not OPENAI_KEY:
    pytest.skip(
        "OPENAI_API_KEY not set - skipping live prompt validation tests",
        allow_module_level=True
    )

if not ANTHROPIC_KEY:
    pytest.skip(
        "ANTHROPIC_API_KEY not set - skipping live prompt validation tests",
        allow_module_level=True
    )


@pytest.fixture
def loaded_prompts():
    """Load actual production prompts from YAML configuration."""
    prompts_path = Path("prompts/task_prompts.yaml")
    assert prompts_path.exists(), "Prompt file not found"

    with open(prompts_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def realistic_sentence():
    """Provide a realistic interview sentence for testing."""
    return "I have been leading a team of 5 engineers to develop microservices architecture using Docker and Kubernetes."


@pytest.fixture
def realistic_contexts():
    """Provide realistic context windows for multi-context prompts."""
    return {
        "immediate_context": (
            "Interviewer: Tell me about your leadership experience. "
            "Candidate: I have been leading a team of 5 engineers to develop microservices architecture using Docker and Kubernetes. "
            "Interviewer: How long have you been in this role?"
        ),
        "observer_context": (
            "This is a technical leadership interview for a Senior Engineering Manager position. "
            "The candidate is discussing their experience managing engineering teams and technical infrastructure."
        ),
        "broader_context": (
            "Interview context: Senior Engineering Manager role at a tech company. "
            "Previous topics discussed: programming languages, development methodologies, team collaboration. "
            "Current topic: leadership experience and infrastructure management. "
            "The candidate has demonstrated strong technical knowledge and communication skills."
        ),
    }


class TestSimplePromptValidation:
    """Test simple prompts (no context) work with both providers."""

    @pytest.mark.asyncio
    async def test_sentence_function_type_openai(self, loaded_prompts, realistic_sentence):
        """Test sentence function type classification with OpenAI."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("openai")

        prompt = loaded_prompts["sentence_function_type"]["prompt"].format(
            sentence=realistic_sentence
        )

        result = await agent.call_model(prompt)

        # Verify response structure
        assert isinstance(result, dict), "Response should be a dictionary"
        assert "function_type" in result, "Response should contain function_type"

        # Verify realistic classification
        assert result["function_type"] in [
            "declarative", "interrogative", "imperative", "exclamatory"
        ], f"Invalid function_type: {result['function_type']}"

        # This sentence is declarative (making a statement)
        assert result["function_type"] == "declarative", (
            f"Expected 'declarative' for statement, got '{result['function_type']}'"
        )

        print(f"\n✓ OpenAI function classification: {result}")

    @pytest.mark.asyncio
    async def test_sentence_function_type_anthropic(self, loaded_prompts, realistic_sentence):
        """Test sentence function type classification with Anthropic."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("anthropic")

        prompt = loaded_prompts["sentence_function_type"]["prompt"].format(
            sentence=realistic_sentence
        )

        result = await agent.call_model(prompt)

        # Verify response structure (same as OpenAI)
        assert isinstance(result, dict), "Response should be a dictionary"
        assert "function_type" in result, "Response should contain function_type"

        # Verify realistic classification
        assert result["function_type"] in [
            "declarative", "interrogative", "imperative", "exclamatory"
        ], f"Invalid function_type: {result['function_type']}"

        # This sentence is declarative (making a statement)
        assert result["function_type"] == "declarative", (
            f"Expected 'declarative' for statement, got '{result['function_type']}'"
        )

        print(f"\n✓ Anthropic function classification: {result}")

    @pytest.mark.asyncio
    async def test_sentence_structure_type_openai(self, loaded_prompts, realistic_sentence):
        """Test sentence structure classification with OpenAI."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("openai")

        prompt = loaded_prompts["sentence_structure_type"]["prompt"].format(
            sentence=realistic_sentence
        )

        result = await agent.call_model(prompt)

        assert isinstance(result, dict)
        assert "structure_type" in result

        # Verify valid structure type
        valid_structures = ["simple", "compound", "complex", "compound-complex"]
        assert any(s in result["structure_type"].lower() for s in valid_structures), (
            f"Invalid structure_type: {result['structure_type']}"
        )

        print(f"\n✓ OpenAI structure classification: {result}")

    @pytest.mark.asyncio
    async def test_sentence_structure_type_anthropic(self, loaded_prompts, realistic_sentence):
        """Test sentence structure classification with Anthropic."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("anthropic")

        prompt = loaded_prompts["sentence_structure_type"]["prompt"].format(
            sentence=realistic_sentence
        )

        result = await agent.call_model(prompt)

        assert isinstance(result, dict)
        assert "structure_type" in result

        # Verify valid structure type
        valid_structures = ["simple", "compound", "complex", "compound-complex"]
        assert any(s in result["structure_type"].lower() for s in valid_structures), (
            f"Invalid structure_type: {result['structure_type']}"
        )

        print(f"\n✓ Anthropic structure classification: {result}")


class TestContextBasedPromptValidation:
    """Test complex prompts with context work with both providers."""

    @pytest.mark.asyncio
    async def test_sentence_purpose_with_context_openai(
        self, loaded_prompts, realistic_sentence, realistic_contexts
    ):
        """Test sentence purpose analysis with context using OpenAI."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("openai")

        prompt = loaded_prompts["sentence_purpose"]["prompt"].format(
            sentence=realistic_sentence,
            context=realistic_contexts["observer_context"]
        )

        result = await agent.call_model(prompt)

        # Verify response structure
        assert isinstance(result, dict)
        assert "purpose" in result

        # Purpose should be a non-empty string
        assert isinstance(result["purpose"], str)
        assert len(result["purpose"]) > 0

        print(f"\n✓ OpenAI purpose with context: {result}")

    @pytest.mark.asyncio
    async def test_sentence_purpose_with_context_anthropic(
        self, loaded_prompts, realistic_sentence, realistic_contexts
    ):
        """Test sentence purpose analysis with context using Anthropic."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("anthropic")

        prompt = loaded_prompts["sentence_purpose"]["prompt"].format(
            sentence=realistic_sentence,
            context=realistic_contexts["observer_context"]
        )

        result = await agent.call_model(prompt)

        # Verify response structure (same as OpenAI)
        assert isinstance(result, dict)
        assert "purpose" in result

        # Purpose should be a non-empty string
        assert isinstance(result["purpose"], str)
        assert len(result["purpose"]) > 0

        print(f"\n✓ Anthropic purpose with context: {result}")

    @pytest.mark.asyncio
    async def test_topic_level_1_with_immediate_context_openai(
        self, loaded_prompts, realistic_sentence, realistic_contexts
    ):
        """Test topic level 1 classification with immediate context using OpenAI."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("openai")

        prompt = loaded_prompts["topic_level_1"]["prompt"].format(
            sentence=realistic_sentence,
            context=realistic_contexts["immediate_context"]
        )

        result = await agent.call_model(prompt)

        # Verify response structure
        assert isinstance(result, dict)
        assert "topic_level_1" in result

        # Topic should be a non-empty string
        assert isinstance(result["topic_level_1"], str)
        assert len(result["topic_level_1"]) > 0

        print(f"\n✓ OpenAI topic L1 with immediate context: {result}")

    @pytest.mark.asyncio
    async def test_topic_level_1_with_immediate_context_anthropic(
        self, loaded_prompts, realistic_sentence, realistic_contexts
    ):
        """Test topic level 1 classification with immediate context using Anthropic."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("anthropic")

        prompt = loaded_prompts["topic_level_1"]["prompt"].format(
            sentence=realistic_sentence,
            context=realistic_contexts["immediate_context"]
        )

        result = await agent.call_model(prompt)

        # Verify response structure (same as OpenAI)
        assert isinstance(result, dict)
        assert "topic_level_1" in result

        # Topic should be a non-empty string
        assert isinstance(result["topic_level_1"], str)
        assert len(result["topic_level_1"]) > 0

        print(f"\n✓ Anthropic topic L1 with immediate context: {result}")

    @pytest.mark.asyncio
    async def test_topic_level_3_with_broader_context_openai(
        self, loaded_prompts, realistic_sentence, realistic_contexts
    ):
        """Test topic level 3 classification with broader context using OpenAI."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("openai")

        prompt = loaded_prompts["topic_level_3"]["prompt"].format(
            sentence=realistic_sentence,
            context=realistic_contexts["broader_context"]
        )

        result = await agent.call_model(prompt)

        # Verify response structure
        assert isinstance(result, dict)
        assert "topic_level_3" in result

        # Topic should be a non-empty string
        assert isinstance(result["topic_level_3"], str)
        assert len(result["topic_level_3"]) > 0

        print(f"\n✓ OpenAI topic L3 with broader context: {result}")

    @pytest.mark.asyncio
    async def test_topic_level_3_with_broader_context_anthropic(
        self, loaded_prompts, realistic_sentence, realistic_contexts
    ):
        """Test topic level 3 classification with broader context using Anthropic."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("anthropic")

        prompt = loaded_prompts["topic_level_3"]["prompt"].format(
            sentence=realistic_sentence,
            context=realistic_contexts["broader_context"]
        )

        result = await agent.call_model(prompt)

        # Verify response structure (same as OpenAI)
        assert isinstance(result, dict)
        assert "topic_level_3" in result

        # Topic should be a non-empty string
        assert isinstance(result["topic_level_3"], str)
        assert len(result["topic_level_3"]) > 0

        print(f"\n✓ Anthropic topic L3 with broader context: {result}")


class TestKeywordExtractionValidation:
    """Test keyword extraction prompts work with both providers."""

    @pytest.mark.asyncio
    async def test_overall_keywords_openai(
        self, loaded_prompts, realistic_contexts
    ):
        """Test overall keywords extraction with OpenAI."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("openai")

        prompt = loaded_prompts["topic_overall_keywords"]["prompt"].format(
            context=realistic_contexts["observer_context"]
        )

        result = await agent.call_model(prompt)

        # Verify response structure
        assert isinstance(result, dict)
        assert "overall_keywords" in result

        # Keywords should be a list
        assert isinstance(result["overall_keywords"], list)

        # Should extract some keywords from the context
        assert len(result["overall_keywords"]) > 0, "Should extract at least some keywords"

        print(f"\n✓ OpenAI overall keywords: {result['overall_keywords']}")

    @pytest.mark.asyncio
    async def test_overall_keywords_anthropic(
        self, loaded_prompts, realistic_contexts
    ):
        """Test overall keywords extraction with Anthropic."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("anthropic")

        prompt = loaded_prompts["topic_overall_keywords"]["prompt"].format(
            context=realistic_contexts["observer_context"]
        )

        result = await agent.call_model(prompt)

        # Verify response structure (same as OpenAI)
        assert isinstance(result, dict)
        assert "overall_keywords" in result

        # Keywords should be a list
        assert isinstance(result["overall_keywords"], list)

        # Should extract some keywords from the context
        assert len(result["overall_keywords"]) > 0, "Should extract at least some keywords"

        print(f"\n✓ Anthropic overall keywords: {result['overall_keywords']}")

    @pytest.mark.asyncio
    async def test_domain_keywords_openai(self, loaded_prompts, realistic_sentence):
        """Test domain keyword extraction with OpenAI."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("openai")

        # Use realistic domain keywords
        domain_keywords = [
            "Python", "JavaScript", "Docker", "Kubernetes", "AWS",
            "microservices", "leadership", "engineering", "architecture"
        ]
        domain_keywords_str = ", ".join(domain_keywords)

        prompt = loaded_prompts["domain_specific_keywords"]["prompt"].format(
            sentence=realistic_sentence,
            domain_keywords=domain_keywords_str
        )

        result = await agent.call_model(prompt)

        # Verify response structure
        assert isinstance(result, dict)
        assert "domain_keywords" in result

        # Should be a list
        assert isinstance(result["domain_keywords"], list)

        # Should identify Docker, Kubernetes, microservices from the sentence
        identified = [k.lower() for k in result["domain_keywords"]]
        assert any("docker" in k or "kubernetes" in k or "microservice" in k
                   for k in identified), (
            f"Should identify technical keywords from sentence: {identified}"
        )

        print(f"\n✓ OpenAI domain keywords: {result['domain_keywords']}")

    @pytest.mark.asyncio
    async def test_domain_keywords_anthropic(self, loaded_prompts, realistic_sentence):
        """Test domain keyword extraction with Anthropic."""
        AgentFactory.reset()
        agent = AgentFactory.create_agent("anthropic")

        # Use realistic domain keywords
        domain_keywords = [
            "Python", "JavaScript", "Docker", "Kubernetes", "AWS",
            "microservices", "leadership", "engineering", "architecture"
        ]
        domain_keywords_str = ", ".join(domain_keywords)

        prompt = loaded_prompts["domain_specific_keywords"]["prompt"].format(
            sentence=realistic_sentence,
            domain_keywords=domain_keywords_str
        )

        result = await agent.call_model(prompt)

        # Verify response structure (same as OpenAI)
        assert isinstance(result, dict)
        assert "domain_keywords" in result

        # Should be a list
        assert isinstance(result["domain_keywords"], list)

        # Should identify Docker, Kubernetes, microservices from the sentence
        identified = [k.lower() for k in result["domain_keywords"]]
        assert any("docker" in k or "kubernetes" in k or "microservice" in k
                   for k in identified), (
            f"Should identify technical keywords from sentence: {identified}"
        )

        print(f"\n✓ Anthropic domain keywords: {result['domain_keywords']}")


class TestCrossProviderConsistency:
    """Test that both providers return compatible response structures."""

    @pytest.mark.asyncio
    async def test_response_structure_consistency(
        self, loaded_prompts, realistic_sentence, realistic_contexts
    ):
        """Test that OpenAI and Anthropic return compatible response structures for the same prompt."""

        # Test with topic_level_1 prompt (uses context)
        prompt = loaded_prompts["topic_level_1"]["prompt"].format(
            sentence=realistic_sentence,
            context=realistic_contexts["immediate_context"]
        )

        # Get response from OpenAI
        AgentFactory.reset()
        openai_agent = AgentFactory.create_agent("openai")
        openai_result = await openai_agent.call_model(prompt)

        # Get response from Anthropic
        AgentFactory.reset()
        anthropic_agent = AgentFactory.create_agent("anthropic")
        anthropic_result = await anthropic_agent.call_model(prompt)

        # Both should be dicts
        assert isinstance(openai_result, dict)
        assert isinstance(anthropic_result, dict)

        # Both should have the same keys
        assert "topic_level_1" in openai_result
        assert "topic_level_1" in anthropic_result

        # Both should have non-empty values
        assert len(openai_result["topic_level_1"]) > 0
        assert len(anthropic_result["topic_level_1"]) > 0

        print(f"\n✓ OpenAI topic: {openai_result['topic_level_1']}")
        print(f"✓ Anthropic topic: {anthropic_result['topic_level_1']}")
        print("✓ Both providers return compatible response structures")
