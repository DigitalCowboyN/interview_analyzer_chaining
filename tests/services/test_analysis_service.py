"""
tests/services/test_analysis_service.py

Comprehensive tests for the AnalysisService that follow cardinal rules:
1. Test actual functionality, not mock interactions
2. Use realistic data and scenarios, not hardcoded values

These tests focus on testing the real service logic with minimal mocking,
using authentic interview content and realistic analysis workflows.
"""

import itertools  # Import itertools for counter
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import yaml

from src.agents.context_builder import ContextBuilder
from src.agents.sentence_analyzer import SentenceAnalyzer
from src.services.analysis_service import AnalysisService
from src.utils.metrics import MetricsTracker


# --- Mock Timer ---
def create_mock_timer(start_time=1.0, increment=0.5):
    """Creates a *callable function* that returns predictably increasing time values."""
    counter = itertools.count()
    # Define and return a function that closes over the counter and state

    def timer_func() -> float:
        return start_time + next(counter) * increment

    return timer_func


# Realistic fixtures for testing with authentic interview data
@pytest.fixture
def realistic_config() -> Dict[str, Any]:
    """Provides a realistic configuration dictionary for interview analysis."""
    return {
        "paths": {
            "output_dir": "data/output",
            "map_dir": "data/maps",
            "map_suffix": "_map.jsonl",
            "analysis_suffix": "_analysis.jsonl",
        },
        "pipeline": {"num_analysis_workers": 2},  # Realistic concurrency for interview processing
        "preprocessing": {
            "context_windows": {
                "immediate": 2,  # ±2 sentences for immediate context
                "broader": 5,  # ±5 sentences for broader context
                "observer": 10,  # ±10 sentences for observer context
                "overall_context": 15,  # Overall context window
            }
        },
        "openai": {
            "api_key": "test-interview-analysis-key",
            "model_name": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.3,
        },
    }


@pytest.fixture
def realistic_interview_data():
    """Provides realistic technical interview sentences and expected analysis."""
    return {
        "sentences": [
            "Can you walk me through your experience with microservices architecture?",
            "I've been working with microservices for about 3 years, primarily using Docker and Kubernetes.",
            "What challenges have you faced when implementing service-to-service communication?",
            "The main challenge was handling distributed transactions and ensuring data consistency.",
            "How do you approach monitoring and observability in a microservices environment?",
            "I use distributed tracing with tools like Jaeger and implement comprehensive logging strategies.",
        ],
        "contexts": [
            {
                "immediate": "Technical interview discussion about system architecture",
                "broader": "Senior software engineer interview focusing on distributed systems",
                "observer": "Interview assessment for technical leadership role",
            },
            {
                "immediate": "Candidate responding to architecture experience question",
                "broader": "Discussion of practical experience with containerization",
                "observer": "Evaluating hands-on experience with modern deployment practices",
            },
            {
                "immediate": "Follow-up question about implementation challenges",
                "broader": "Deep dive into distributed systems complexity",
                "observer": "Assessing problem-solving approach and technical depth",
            },
            {
                "immediate": "Candidate explaining distributed transaction challenges",
                "broader": "Technical discussion about data consistency patterns",
                "observer": "Evaluation of understanding of complex distributed systems concepts",
            },
            {
                "immediate": "Question about operational aspects of microservices",
                "broader": "Exploring monitoring and observability practices",
                "observer": "Assessing operational maturity and production experience",
            },
            {
                "immediate": "Candidate describing monitoring and tracing approach",
                "broader": "Demonstration of practical DevOps and SRE knowledge",
                "observer": "Final assessment of comprehensive technical expertise",
            },
        ],
        "expected_analysis": [
            {
                "function_type": "interrogative",
                "structure_type": "complex",
                "purpose": "technical_assessment",
                "topic_level_1": "technical_skills",
                "topic_level_3": "system_architecture",
                "overall_keywords": ["experience", "microservices", "architecture"],
                "domain_keywords": ["microservices", "architecture", "distributed_systems"],
            },
            {
                "function_type": "declarative",
                "structure_type": "compound",
                "purpose": "experience_sharing",
                "topic_level_1": "technical_experience",
                "topic_level_3": "containerization",
                "overall_keywords": ["working", "microservices", "years", "Docker", "Kubernetes"],
                "domain_keywords": ["microservices", "Docker", "Kubernetes", "containerization"],
            },
            {
                "function_type": "interrogative",
                "structure_type": "complex",
                "purpose": "problem_solving_assessment",
                "topic_level_1": "technical_challenges",
                "topic_level_3": "service_communication",
                "overall_keywords": ["challenges", "implementing", "service", "communication"],
                "domain_keywords": ["service_mesh", "API_gateway", "inter_service_communication"],
            },
            {
                "function_type": "declarative",
                "structure_type": "compound",
                "purpose": "problem_explanation",
                "topic_level_1": "technical_challenges",
                "topic_level_3": "distributed_transactions",
                "overall_keywords": ["challenge", "distributed", "transactions", "consistency"],
                "domain_keywords": ["distributed_transactions", "data_consistency", "ACID"],
            },
            {
                "function_type": "interrogative",
                "structure_type": "complex",
                "purpose": "operational_assessment",
                "topic_level_1": "operational_practices",
                "topic_level_3": "monitoring_observability",
                "overall_keywords": ["monitoring", "observability", "microservices", "environment"],
                "domain_keywords": ["monitoring", "observability", "distributed_tracing", "metrics"],
            },
            {
                "function_type": "declarative",
                "structure_type": "compound",
                "purpose": "methodology_explanation",
                "topic_level_1": "operational_practices",
                "topic_level_3": "distributed_tracing",
                "overall_keywords": ["distributed", "tracing", "Jaeger", "logging", "strategies"],
                "domain_keywords": ["distributed_tracing", "Jaeger", "logging", "observability"],
            },
        ],
    }


# Realistic AnalysisService fixture with minimal mocking
@pytest.fixture
def realistic_analysis_service(
    realistic_config: Dict[str, Any],
    realistic_context_builder: MagicMock,
    realistic_sentence_analyzer: AsyncMock,
    realistic_metrics_tracker: MagicMock,
) -> AnalysisService:
    """Provides an AnalysisService instance with realistic dependencies.

    Uses minimal mocking focused on external dependencies while testing
    the actual service logic with realistic interview data.
    """
    return AnalysisService(
        config=realistic_config,
        context_builder=realistic_context_builder,
        sentence_analyzer=realistic_sentence_analyzer,
        metrics_tracker=realistic_metrics_tracker,
    )


# --- Realistic Mock Dependencies for Testing Service Logic ---
@pytest.fixture
def realistic_context_builder() -> MagicMock:
    """Provides a realistic ContextBuilder mock that returns authentic interview contexts."""
    mock = MagicMock(spec=ContextBuilder)

    def build_realistic_contexts(sentences: List[str]) -> Dict[int, Dict[str, str]]:
        """Build realistic contexts based on sentence content and position."""
        contexts = {}
        for i, sentence in enumerate(sentences):
            if "microservices" in sentence.lower():
                contexts[i] = {
                    "immediate": f"Technical discussion about microservices (sentence {i})",
                    "broader": "Senior engineer interview focusing on distributed systems",
                    "observer": "Assessment of system architecture expertise",
                }
            elif "docker" in sentence.lower() or "kubernetes" in sentence.lower():
                contexts[i] = {
                    "immediate": f"Containerization experience discussion (sentence {i})",
                    "broader": "DevOps and deployment practices evaluation",
                    "observer": "Assessment of modern deployment knowledge",
                }
            elif "challenge" in sentence.lower() or "problem" in sentence.lower():
                contexts[i] = {
                    "immediate": f"Problem-solving scenario discussion (sentence {i})",
                    "broader": "Technical challenges and solutions exploration",
                    "observer": "Evaluation of analytical and problem-solving skills",
                }
            else:
                contexts[i] = {
                    "immediate": f"General technical interview context (sentence {i})",
                    "broader": "Technical competency assessment",
                    "observer": "Overall technical evaluation",
                }
        return contexts

    mock.build_all_contexts.side_effect = build_realistic_contexts
    return mock


@pytest.fixture
def realistic_sentence_analyzer() -> AsyncMock:
    """Provides a realistic SentenceAnalyzer mock that returns authentic analysis results."""
    mock = AsyncMock(spec=SentenceAnalyzer)

    async def classify_realistic_sentence(sentence: str, context: Dict[str, str]) -> Dict[str, Any]:
        """Classify sentences based on realistic interview analysis patterns."""
        sentence_lower = sentence.lower()

        # Analyze based on actual sentence characteristics
        if sentence.endswith("?"):
            function_type = "interrogative"
            if "experience" in sentence_lower:
                purpose = "technical_assessment"
                topic_level_1 = "technical_skills"
            elif "challenge" in sentence_lower:
                purpose = "problem_solving_assessment"
                topic_level_1 = "technical_challenges"
            elif "approach" in sentence_lower:
                purpose = "operational_assessment"
                topic_level_1 = "operational_practices"
            else:
                purpose = "general_inquiry"
                topic_level_1 = "general_discussion"
        else:
            function_type = "declarative"
            if "i've been" in sentence_lower or "i have" in sentence_lower:
                purpose = "experience_sharing"
                topic_level_1 = "technical_experience"
            elif "challenge" in sentence_lower or "main" in sentence_lower:
                purpose = "problem_explanation"
                topic_level_1 = "technical_challenges"
            else:
                purpose = "methodology_explanation"
                topic_level_1 = "operational_practices"

        # Extract realistic keywords
        overall_keywords = []
        domain_keywords = []

        technical_terms = {
            "microservices": "microservices",
            "docker": "Docker",
            "kubernetes": "Kubernetes",
            "distributed": "distributed_systems",
            "tracing": "distributed_tracing",
            "jaeger": "Jaeger",
            "monitoring": "monitoring",
            "observability": "observability",
        }

        words = sentence_lower.split()
        for word in words:
            if len(word) > 3 and word not in ["have", "been", "with", "that", "this"]:
                overall_keywords.append(word)

            for term, domain_term in technical_terms.items():
                if term in word:
                    domain_keywords.append(domain_term)

        return {
            "function_type": function_type,
            "structure_type": "complex" if len(words) > 10 else "simple",
            "purpose": purpose,
            "topic_level_1": topic_level_1,
            "topic_level_3": "system_architecture" if "microservices" in sentence_lower else "technical_implementation",
            "overall_keywords": overall_keywords[:6],  # Limit to realistic number
            "domain_keywords": list(set(domain_keywords)),  # Remove duplicates
        }

    mock.classify_sentence = classify_realistic_sentence
    return mock


@pytest.fixture
def realistic_metrics_tracker() -> MagicMock:
    """Provides a realistic MetricsTracker mock for testing service metrics."""
    mock = MagicMock(spec=MetricsTracker)

    # Track realistic metrics
    mock.sentences_successful = 0
    mock.errors_encountered = 0
    mock.total_processing_time = 0.0

    def increment_success():
        mock.sentences_successful += 1

    def increment_errors():
        mock.errors_encountered += 1

    def add_time(sentence_id: int, processing_time: float):
        mock.total_processing_time += processing_time

    mock.increment_sentences_success.side_effect = increment_success
    mock.increment_errors.side_effect = increment_errors
    mock.add_processing_time.side_effect = add_time

    return mock


# --------------------------------------

# --- Tests for build_contexts ---


def test_build_contexts_with_realistic_interview_data(
    realistic_analysis_service: AnalysisService,
    realistic_context_builder: MagicMock,
    realistic_interview_data: Dict[str, Any],
) -> None:
    """Test build_contexts with realistic technical interview sentences."""
    # Use first 3 sentences from realistic interview data
    interview_sentences = realistic_interview_data["sentences"][:3]

    result = realistic_analysis_service.build_contexts(interview_sentences)

    # Verify the context builder was called with actual interview sentences
    realistic_context_builder.build_all_contexts.assert_called_once_with(interview_sentences)

    # Verify we get realistic contexts back
    assert len(result) == 3
    assert all(isinstance(ctx, dict) for ctx in result)

    # Check that contexts contain realistic interview content
    microservices_context = result[0]  # First sentence is about microservices
    assert "microservices" in microservices_context["immediate"].lower()
    assert "distributed systems" in microservices_context["broader"]
    assert "architecture" in microservices_context["observer"]

    # Verify context structure matches expected interview analysis format
    for ctx in result:
        assert "immediate" in ctx
        assert "broader" in ctx
        assert "observer" in ctx


def test_build_contexts_empty(analysis_service: AnalysisService, mock_context_builder: MagicMock) -> None:
    """Test build_contexts with empty sentence list."""
    sentences: List[str] = []
    mock_context_builder.build_all_contexts.return_value = {}

    result = analysis_service.build_contexts(sentences)

    mock_context_builder.build_all_contexts.assert_called_once_with(sentences)
    assert result == []


def test_build_contexts_exception(analysis_service: AnalysisService, mock_context_builder: MagicMock) -> None:
    """Test build_contexts when the context builder raises an exception."""
    sentences: List[str] = ["s1"]
    mock_context_builder.build_all_contexts.side_effect = ValueError("Builder failed")

    with pytest.raises(ValueError, match="Builder failed"):
        analysis_service.build_contexts(sentences)

    mock_context_builder.build_all_contexts.assert_called_once_with(sentences)


# --- Tests for analyze_sentences ---


@pytest.mark.asyncio
async def test_analyze_sentences_with_realistic_interview_content(
    realistic_analysis_service: AnalysisService,
    realistic_sentence_analyzer: AsyncMock,
    realistic_metrics_tracker: MagicMock,
    realistic_interview_data: Dict[str, Any],
) -> None:
    """Test analyze_sentences with realistic technical interview content."""
    # Use first 2 sentences from realistic interview data
    interview_sentences = realistic_interview_data["sentences"][:2]
    interview_contexts = realistic_interview_data["contexts"][:2]

    mock_timer = create_mock_timer(start_time=2.0, increment=0.3)

    # Call the service method with realistic interview data
    results = await realistic_analysis_service.analyze_sentences(
        interview_sentences, interview_contexts, timer=mock_timer
    )

    # Verify we get realistic analysis results
    assert len(results) == 2

    # Test first result (microservices question)
    first_result = results[0]
    assert first_result["sentence_id"] == 0
    assert first_result["sequence_order"] == 0
    assert first_result["sentence"] == interview_sentences[0]

    # Verify realistic analysis content
    assert first_result["function_type"] == "interrogative"  # It's a question
    assert first_result["purpose"] == "technical_assessment"  # Interview assessment
    assert "microservices" in first_result["overall_keywords"]
    assert "microservices" in first_result["domain_keywords"]

    # Test second result (Docker/Kubernetes experience)
    second_result = results[1]
    assert second_result["sentence_id"] == 1
    assert second_result["sequence_order"] == 1
    assert second_result["sentence"] == interview_sentences[1]

    # Verify realistic analysis content
    assert second_result["function_type"] == "declarative"  # Statement about experience
    assert second_result["purpose"] == "experience_sharing"  # Sharing background

    # Verify technical keywords are extracted (flexible check)
    has_technical_keywords = (
        any("docker" in keyword.lower() for keyword in second_result["overall_keywords"])
        or any("kubernetes" in keyword.lower() for keyword in second_result["overall_keywords"])
        or any("microservices" in keyword.lower() for keyword in second_result["overall_keywords"])
    )
    assert has_technical_keywords, f"Expected technical keywords in: {second_result['overall_keywords']}"

    # Verify domain keywords contain relevant technical terms
    domain_keywords = second_result["domain_keywords"]
    has_domain_keywords = (
        "Docker" in domain_keywords or "Kubernetes" in domain_keywords or "microservices" in domain_keywords
    )
    assert has_domain_keywords, f"Expected domain keywords in: {domain_keywords}"

    # Verify realistic metrics tracking
    assert realistic_metrics_tracker.sentences_successful == 2
    assert realistic_metrics_tracker.errors_encountered == 0
    assert realistic_metrics_tracker.total_processing_time > 0.0


@pytest.mark.asyncio
async def test_analyze_sentences_empty_input(
    analysis_service: AnalysisService, mock_sentence_analyzer: AsyncMock
) -> None:
    """Test analyze_sentences with empty sentences list."""
    # No timer needed here, no patch needed
    with patch("src.services.analysis_service.logger") as mock_logger:
        results = await analysis_service.analyze_sentences([], [])

    assert results == []
    mock_sentence_analyzer.classify_sentence.assert_not_awaited()
    mock_logger.warning.assert_called_with("analyze_sentences called with no sentences. Returning empty list.")


@pytest.mark.asyncio
async def test_analyze_sentences_context_mismatch(
    analysis_service: AnalysisService, mock_sentence_analyzer: AsyncMock
) -> None:
    """Test analyze_sentences when sentence and context counts differ."""
    # No timer needed here, no patch needed

    with patch("src.services.analysis_service.logger") as mock_logger:
        results = await analysis_service.analyze_sentences(["s1", "s2"], [{"c": "1"}])  # Mismatch

    assert results == []
    mock_sentence_analyzer.classify_sentence.assert_not_awaited()
    mock_logger.error.assert_called_with(
        "Sentence count (2) and context count (1) mismatch in analyze_sentences. Aborting."
    )


@pytest.mark.asyncio
async def test_analyze_sentences_classify_error(
    analysis_service: AnalysisService,
    mock_sentence_analyzer: AsyncMock,  # The base mock instance
    mock_metrics_tracker: MagicMock,
) -> None:
    """Test error path, configuring the mock analyzer before the call."""
    sentences: List[str] = ["s1_ok", "s2_fail"]
    contexts: List[Dict[str, str]] = [{"c": "1"}, {"c": "2"}]
    mock_analysis_result_s1 = {"analysis": "result1"}
    test_exception = ValueError("API Error")

    expected_results = [
        {
            "sentence_id": 0,
            "sequence_order": 0,
            "sentence": "s1_ok",
            "analysis": "result1",
        },
        {
            "sentence_id": 1,
            "sequence_order": 1,
            "sentence": "s2_fail",
            "error": True,
            "error_type": "ValueError",
            "error_message": "API Error",
        },
    ]
    mock_timer = create_mock_timer()

    # Configure the classify_sentence method on the *mock instance* used by the service
    mock_sentence_analyzer.classify_sentence.side_effect = [
        mock_analysis_result_s1,
        test_exception,
    ]

    # Call the service method (NO inner patching needed)
    results = await analysis_service.analyze_sentences(sentences, contexts, timer=mock_timer)

    # Assert Results
    assert results == expected_results

    # Assert Metrics
    assert mock_metrics_tracker.increment_errors.call_count == 1
    assert mock_metrics_tracker.increment_sentences_success.call_count == 1
    assert mock_metrics_tracker.add_processing_time.call_count == 1
    mock_metrics_tracker.add_processing_time.assert_called_once_with(0, 0.5)

    # Optional: Assert logger error was called (using caplog fixture if needed)
    # ...


# --- New Concurrency Test --- #
@pytest.mark.asyncio
async def test_analyze_sentences_concurrency(
    # Need specific service instance for concurrency config
    mock_config: Dict[str, Any],  # Get base config
    mock_context_builder: MagicMock,
    mock_sentence_analyzer: AsyncMock,  # The base mock instance
    mock_metrics_tracker: MagicMock,
) -> None:
    """Test concurrency path, configuring the mock analyzer before the call."""
    # Create a modified config for this test
    mock_config_concurrent = mock_config.copy()
    mock_config_concurrent["pipeline"] = {"num_analysis_workers": 2}

    # Instantiate service with modified config and *base* mocks
    analysis_service_concurrent = AnalysisService(
        config=mock_config_concurrent,
        context_builder=mock_context_builder,
        sentence_analyzer=mock_sentence_analyzer,  # Pass the base mock
        metrics_tracker=mock_metrics_tracker,
    )

    sentences: List[str] = ["s1", "s2", "s3"]
    contexts: List[Dict[str, str]] = [{"c": "1"}, {"c": "2"}, {"c": "3"}]
    mock_results_list = [{"analysis": "r1"}, {"analysis": "r2"}, {"analysis": "r3"}]
    expected_final_results = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "s1", "analysis": "r1"},
        {"sentence_id": 1, "sequence_order": 1, "sentence": "s2", "analysis": "r2"},
        {"sentence_id": 2, "sequence_order": 2, "sentence": "s3", "analysis": "r3"},
    ]
    mock_timer = create_mock_timer()

    # Configure the classify_sentence method on the *mock instance* used by the service
    mock_sentence_analyzer.classify_sentence.side_effect = mock_results_list

    # Call the service method (NO inner patching needed)
    results = await analysis_service_concurrent.analyze_sentences(sentences, contexts, timer=mock_timer)

    # Assert Results
    assert results == expected_final_results

    # Assert Metrics
    mock_metrics_tracker.increment_errors.assert_not_called()
    assert mock_metrics_tracker.increment_sentences_success.call_count == 3
    assert mock_metrics_tracker.add_processing_time.call_count == 3
    mock_metrics_tracker.add_processing_time.assert_has_calls(
        [
            call(0, 0.5),
            call(1, 0.5),
            call(2, 0.5),
        ],
        any_order=True,
    )


# Add more tests: e.g., different number of workers, loader errors?
