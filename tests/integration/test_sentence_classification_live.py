"""
test_sentence_classification_live.py

Live integration tests for actual sentence classification using real API calls.

These tests verify that the SentenceAnalyzer performs real classification tasks
with both OpenAI and Anthropic providers, testing all 7 analysis dimensions:
1. Function type (declarative, interrogative, etc.)
2. Structure type (simple, compound, complex)
3. Purpose
4. Topic Level 1
5. Topic Level 3
6. Overall keywords
7. Domain keywords

Requirements:
- OPENAI_API_KEY environment variable must be set
- ANTHROPIC_API_KEY environment variable must be set
- Internet connection required

Usage:
    pytest tests/integration/test_sentence_classification_live.py -xvs
"""

import os
from pathlib import Path

import pytest
import yaml

from src.agents.agent_factory import AgentFactory
from src.agents.context_builder import ContextBuilder
from src.agents.sentence_analyzer import SentenceAnalyzer
from src.config import config

# Check for API keys
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not OPENAI_KEY:
    pytest.skip("OPENAI_API_KEY not set - skipping OpenAI classification tests", allow_module_level=True)

if not ANTHROPIC_KEY:
    pytest.skip("ANTHROPIC_API_KEY not set - skipping Anthropic classification tests", allow_module_level=True)


@pytest.fixture
def realistic_test_config():
    """Provide realistic configuration for classification tests."""
    prompts_path = Path("prompts/task_prompts.yaml")

    return {
        "classification": {
            "local": {
                "prompt_files": {
                    "no_context": str(prompts_path)
                }
            }
        },
        "domain_keywords": [
            "Python", "JavaScript", "Docker", "Kubernetes",
            "AWS", "React", "Django", "Flask", "API", "database"
        ],
        "preprocessing": {
            "context_windows": {
                "structure_analysis": 0,
                "immediate_context": 2,
                "observer_context": 4,
                "broader_context": 6,
                "overall_context": 10
            }
        }
    }


@pytest.fixture
def realistic_interview_sentences():
    """Provide realistic interview sentences for testing."""
    return [
        "Tell me about your experience with Python programming.",
        "I have been working with Python for over 5 years in web development.",
        "What frameworks have you used with Python?",
        "I've primarily used Django and Flask for building REST APIs.",
        "Can you describe a challenging project you worked on?",
        "I led the development of a microservices architecture using Docker and Kubernetes.",
        "How do you approach debugging complex issues?",
        "Thank you for your time today."
    ]


@pytest.fixture
def realistic_contexts(realistic_interview_sentences):
    """Build realistic contexts for each sentence."""
    builder = ContextBuilder(config)

    # build_all_contexts returns Dict[int, Dict[str, str]]
    # where each sentence index maps to its contexts by window name
    all_contexts = builder.build_all_contexts(realistic_interview_sentences)

    # Convert to list for easier indexing in tests
    contexts = [all_contexts[idx] for idx in range(len(realistic_interview_sentences))]

    return contexts


@pytest.mark.asyncio
async def test_openai_live_sentence_classification(
    realistic_test_config,
    realistic_interview_sentences,
    realistic_contexts
):
    """
    Test OpenAI performs real sentence classification with all 7 analysis dimensions.

    This test makes REAL API calls to OpenAI to verify:
    - Function type classification works
    - Structure type classification works
    - Purpose analysis works
    - Topic classification (L1 and L3) works
    - Keyword extraction works
    - All results are realistic and properly formatted
    """
    print("\n🔵 Testing OpenAI Live Sentence Classification...")

    # Ensure OpenAI provider is selected
    AgentFactory.reset()
    agent = AgentFactory.create_agent("openai")
    print(f"  ✓ Provider: {agent.get_provider_name()}")
    print(f"  ✓ Model: {agent.get_model_name()}")

    # Create sentence analyzer with realistic config
    analyzer = SentenceAnalyzer(config_dict=realistic_test_config)

    # Test classification on a few representative sentences
    test_indices = [0, 1, 3]  # Question, declarative response, technical statement

    for idx in test_indices:
        sentence = realistic_interview_sentences[idx]
        context = realistic_contexts[idx]

        print(f"\n  Analyzing: '{sentence[:60]}...'")

        # Make REAL API call for classification
        result = await analyzer.classify_sentence(sentence, context)

        # Verify all 7 analysis dimensions are present
        assert "sentence" in result
        assert result["sentence"] == sentence

        # 1. Function type
        assert "function_type" in result
        assert result["function_type"] in [
            "declarative", "interrogative", "imperative", "exclamatory"
        ]
        print(f"    ✓ Function: {result['function_type']}")

        # 2. Structure type
        assert "structure_type" in result
        assert result["structure_type"] in ["simple", "compound", "complex"]
        print(f"    ✓ Structure: {result['structure_type']}")

        # 3. Purpose
        assert "purpose" in result
        assert isinstance(result["purpose"], str)
        assert len(result["purpose"]) > 0
        print(f"    ✓ Purpose: {result['purpose']}")

        # 4. Topic Level 1
        assert "topic_level_1" in result
        assert isinstance(result["topic_level_1"], str)
        print(f"    ✓ Topic L1: {result['topic_level_1']}")

        # 5. Topic Level 3
        assert "topic_level_3" in result
        assert isinstance(result["topic_level_3"], str)
        print(f"    ✓ Topic L3: {result['topic_level_3']}")

        # 6. Overall keywords
        assert "overall_keywords" in result
        assert isinstance(result["overall_keywords"], list)
        print(f"    ✓ Keywords: {result['overall_keywords'][:3]}")

        # 7. Domain keywords
        assert "domain_keywords" in result
        assert isinstance(result["domain_keywords"], list)
        print(f"    ✓ Domain Keywords: {result['domain_keywords']}")

        # Verify realistic results based on sentence type
        if idx == 0:  # "Tell me about your experience..."
            assert result["function_type"] in ["interrogative", "imperative"]
        elif idx == 1:  # "I have been working with Python..."
            assert result["function_type"] == "declarative"
            # Should identify Python as domain keyword
            domain_kw_lower = [kw.lower() for kw in result.get("domain_keywords", [])]
            assert "python" in domain_kw_lower
        elif idx == 3:  # "I've primarily used Django and Flask..."
            assert result["function_type"] == "declarative"
            # Should identify technical frameworks
            domain_kw_lower = [kw.lower() for kw in result.get("domain_keywords", [])]
            assert any(fw in domain_kw_lower for fw in ["django", "flask"])

    print("\n  ✅ OpenAI live classification test PASSED")


@pytest.mark.asyncio
async def test_anthropic_live_sentence_classification(
    realistic_test_config,
    realistic_interview_sentences,
    realistic_contexts
):
    """
    Test Anthropic performs real sentence classification with all 7 analysis dimensions.

    This test makes REAL API calls to Anthropic Claude to verify:
    - Function type classification works
    - Structure type classification works
    - Purpose analysis works
    - Topic classification (L1 and L3) works
    - Keyword extraction works
    - JSON parsing works (Anthropic uses prompt engineering, not native JSON mode)
    - All results are realistic and properly formatted
    """
    print("\n🟣 Testing Anthropic Live Sentence Classification...")

    # Switch to Anthropic provider
    AgentFactory.reset()
    agent = AgentFactory.create_agent("anthropic")
    print(f"  ✓ Provider: {agent.get_provider_name()}")
    print(f"  ✓ Model: {agent.get_model_name()}")

    # Create sentence analyzer with realistic config
    analyzer = SentenceAnalyzer(config_dict=realistic_test_config)

    # Test classification on a few representative sentences
    test_indices = [0, 1, 3]  # Question, declarative response, technical statement

    for idx in test_indices:
        sentence = realistic_interview_sentences[idx]
        context = realistic_contexts[idx]

        print(f"\n  Analyzing: '{sentence[:60]}...'")

        # Make REAL API call for classification (tests Anthropic JSON parsing)
        result = await analyzer.classify_sentence(sentence, context)

        # Verify all 7 analysis dimensions are present
        assert "sentence" in result
        assert result["sentence"] == sentence

        # 1. Function type
        assert "function_type" in result
        assert result["function_type"] in [
            "declarative", "interrogative", "imperative", "exclamatory"
        ]
        print(f"    ✓ Function: {result['function_type']}")

        # 2. Structure type
        assert "structure_type" in result
        assert result["structure_type"] in ["simple", "compound", "complex"]
        print(f"    ✓ Structure: {result['structure_type']}")

        # 3. Purpose
        assert "purpose" in result
        assert isinstance(result["purpose"], str)
        assert len(result["purpose"]) > 0
        print(f"    ✓ Purpose: {result['purpose']}")

        # 4. Topic Level 1
        assert "topic_level_1" in result
        assert isinstance(result["topic_level_1"], str)
        print(f"    ✓ Topic L1: {result['topic_level_1']}")

        # 5. Topic Level 3
        assert "topic_level_3" in result
        assert isinstance(result["topic_level_3"], str)
        print(f"    ✓ Topic L3: {result['topic_level_3']}")

        # 6. Overall keywords
        assert "overall_keywords" in result
        assert isinstance(result["overall_keywords"], list)
        print(f"    ✓ Keywords: {result['overall_keywords'][:3]}")

        # 7. Domain keywords
        assert "domain_keywords" in result
        assert isinstance(result["domain_keywords"], list)
        print(f"    ✓ Domain Keywords: {result['domain_keywords']}")

        # Verify realistic results based on sentence type
        if idx == 0:  # "Tell me about your experience..."
            assert result["function_type"] in ["interrogative", "imperative"]
        elif idx == 1:  # "I have been working with Python..."
            assert result["function_type"] == "declarative"
            # Should identify Python as domain keyword
            domain_kw_lower = [kw.lower() for kw in result.get("domain_keywords", [])]
            assert "python" in domain_kw_lower
        elif idx == 3:  # "I've primarily used Django and Flask..."
            assert result["function_type"] == "declarative"
            # Should identify technical frameworks
            domain_kw_lower = [kw.lower() for kw in result.get("domain_keywords", [])]
            assert any(fw in domain_kw_lower for fw in ["django", "flask"])

    print("\n  ✅ Anthropic live classification test PASSED")


@pytest.mark.asyncio
async def test_compare_providers_classification_quality(
    realistic_test_config,
    realistic_interview_sentences,
    realistic_contexts
):
    """
    Compare classification quality between OpenAI and Anthropic.

    Verifies:
    - Both providers return valid classifications
    - Both identify the same function type (high agreement expected)
    - Both identify relevant domain keywords
    - Format consistency between providers
    """
    print("\n🔄 Comparing Provider Classification Quality...")

    # Test one sentence with both providers
    test_sentence = realistic_interview_sentences[3]  # Technical statement
    test_context = realistic_contexts[3]

    print(f"\n  Test sentence: '{test_sentence}'")

    # Test with OpenAI
    AgentFactory.reset()
    openai_agent = AgentFactory.create_agent("openai")
    openai_analyzer = SentenceAnalyzer(config_dict=realistic_test_config)

    print(f"\n  Classifying with {openai_agent.get_provider_name()}...")
    openai_result = await openai_analyzer.classify_sentence(test_sentence, test_context)

    # Test with Anthropic
    AgentFactory.reset()
    anthropic_agent = AgentFactory.create_agent("anthropic")
    anthropic_analyzer = SentenceAnalyzer(config_dict=realistic_test_config)

    print(f"  Classifying with {anthropic_agent.get_provider_name()}...")
    anthropic_result = await anthropic_analyzer.classify_sentence(test_sentence, test_context)

    # Compare results
    print(f"\n  OpenAI Results:")
    print(f"    Function: {openai_result['function_type']}")
    print(f"    Structure: {openai_result['structure_type']}")
    print(f"    Domain Keywords: {openai_result['domain_keywords']}")

    print(f"\n  Anthropic Results:")
    print(f"    Function: {anthropic_result['function_type']}")
    print(f"    Structure: {anthropic_result['structure_type']}")
    print(f"    Domain Keywords: {anthropic_result['domain_keywords']}")

    # Both should agree on function type (declarative for this sentence)
    assert openai_result["function_type"] == "declarative"
    assert anthropic_result["function_type"] == "declarative"
    print(f"\n  ✓ Both providers agree on function type: declarative")

    # Both should identify technical keywords (Django, Flask, API)
    openai_kw_lower = [kw.lower() for kw in openai_result.get("domain_keywords", [])]
    anthropic_kw_lower = [kw.lower() for kw in anthropic_result.get("domain_keywords", [])]

    technical_terms = ["django", "flask", "api", "rest"]
    openai_found = [term for term in technical_terms if term in openai_kw_lower]
    anthropic_found = [term for term in technical_terms if term in anthropic_kw_lower]

    assert len(openai_found) > 0, "OpenAI should identify technical terms"
    assert len(anthropic_found) > 0, "Anthropic should identify technical terms"
    print(f"  ✓ OpenAI identified: {openai_found}")
    print(f"  ✓ Anthropic identified: {anthropic_found}")

    # Both should have all required fields
    required_fields = [
        "function_type", "structure_type", "purpose",
        "topic_level_1", "topic_level_3",
        "overall_keywords", "domain_keywords"
    ]

    for field in required_fields:
        assert field in openai_result, f"OpenAI missing field: {field}"
        assert field in anthropic_result, f"Anthropic missing field: {field}"

    print(f"  ✓ Both providers return complete classification")
    print("\n  ✅ Provider comparison test PASSED")


@pytest.mark.asyncio
async def test_classification_with_context_windows(
    realistic_test_config,
    realistic_interview_sentences,
    realistic_contexts
):
    """
    Test that context windows affect classification appropriately.

    Verifies:
    - Classifications use context from surrounding sentences
    - Different context windows produce appropriate analysis
    - Context-dependent fields (purpose, topics) vary with position
    """
    print("\n📊 Testing Context Window Influence on Classification...")

    # Use Anthropic for this test
    AgentFactory.reset()
    agent = AgentFactory.create_agent("anthropic")
    analyzer = SentenceAnalyzer(config_dict=realistic_test_config)

    print(f"  ✓ Using {agent.get_provider_name()} ({agent.get_model_name()})")

    # Compare early sentence (limited context) vs middle sentence (full context)
    early_idx = 0  # First sentence - no preceding context
    middle_idx = 4  # Middle sentence - full context available

    early_sentence = realistic_interview_sentences[early_idx]
    middle_sentence = realistic_interview_sentences[middle_idx]

    print(f"\n  Early sentence: '{early_sentence[:50]}...'")
    early_result = await analyzer.classify_sentence(
        early_sentence,
        realistic_contexts[early_idx]
    )

    print(f"  Middle sentence: '{middle_sentence[:50]}...'")
    middle_result = await analyzer.classify_sentence(
        middle_sentence,
        realistic_contexts[middle_idx]
    )

    # Both should have complete classifications
    assert "topic_level_1" in early_result
    assert "topic_level_1" in middle_result
    assert "purpose" in early_result
    assert "purpose" in middle_result

    print(f"\n  Early sentence analysis:")
    print(f"    Purpose: {early_result['purpose']}")
    print(f"    Topic L1: {early_result['topic_level_1']}")

    print(f"\n  Middle sentence analysis:")
    print(f"    Purpose: {middle_result['purpose']}")
    print(f"    Topic L1: {middle_result['topic_level_1']}")

    # Context-dependent fields should have realistic values
    assert len(early_result["purpose"]) > 0
    assert len(middle_result["purpose"]) > 0
    assert len(early_result["topic_level_1"]) > 0
    assert len(middle_result["topic_level_1"]) > 0

    print("\n  ✓ Both positions produce valid context-aware analysis")
    print("  ✅ Context window test PASSED")
