"""
Tests for the AnalysisService.
"""

import itertools  # Import itertools for counter
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

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


# Fixture for mock config
@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provides a mock configuration dictionary for tests."""
    return {
        "paths": {
            "output_dir": "mock_output",
            "map_dir": "mock_maps",
            "map_suffix": "_map.jsonl",
            "analysis_suffix": "_analysis.jsonl",
        },
        "pipeline": {
            "num_analysis_workers": 1  # Use 1 worker for simplicity in unit tests
        },
        "preprocessing": {
            "context_windows": {"immediate": 1, "broader": 3, "observer": 5}
        },
        # Add other config sections if needed by service or its dependencies
    }


# Fixture for AnalysisService instance
@pytest.fixture
def analysis_service(
    mock_config: Dict[str, Any],
    mock_context_builder: MagicMock,
    mock_sentence_analyzer: AsyncMock,  # Inject base mock
    mock_metrics_tracker: MagicMock,
) -> AnalysisService:
    """Provides an AnalysisService instance with *base* mocked dependencies.
    Specific mock behaviors (like classify_sentence side_effect) should be
    configured *within the tests* using this instance or its mocks.
    """
    # NOTE: We no longer configure side_effects here. Tests will configure the
    # mock_sentence_analyzer passed *into* this fixture as needed,
    # typically right before calling the service method.
    return AnalysisService(
        config=mock_config,
        context_builder=mock_context_builder,
        sentence_analyzer=mock_sentence_analyzer,  # Pass the base mock
        metrics_tracker=mock_metrics_tracker,
    )


# --- NEW Fixtures for Mock Dependencies ---
@pytest.fixture
def mock_context_builder() -> MagicMock:
    """Provides a mock ContextBuilder instance."""
    mock = MagicMock(spec=ContextBuilder)
    # Configure default mock behavior if needed, e.g.:
    mock.build_all_contexts.return_value = {}
    return mock


@pytest.fixture
def mock_sentence_analyzer() -> AsyncMock:
    """Provides a *base* mock SentenceAnalyzer instance."""
    mock = AsyncMock(spec=SentenceAnalyzer)
    # Ensure the attribute exists and is an AsyncMock, but DO NOT configure side_effect here.
    mock.classify_sentence = AsyncMock()
    return mock


@pytest.fixture
def mock_metrics_tracker() -> MagicMock:
    """Provides a mock MetricsTracker instance."""
    mock = MagicMock(spec=MetricsTracker)
    # Configure default mock behavior (methods are usually called without return value check)
    mock.increment_sentences_processed = MagicMock()
    mock.increment_sentences_success = MagicMock()
    mock.increment_errors = MagicMock()
    mock.add_processing_time = MagicMock()
    return mock


# --------------------------------------

# --- Tests for build_contexts ---


def test_build_contexts_success(
    analysis_service: AnalysisService, mock_context_builder: MagicMock
) -> None:
    """Test build_contexts successfully converts dict from builder to list."""
    sentences: List[str] = ["s1", "s2"]
    mock_builder_result = {0: {"ctx": "c1"}, 1: {"ctx": "c2"}}
    expected_list = [{"ctx": "c1"}, {"ctx": "c2"}]
    mock_context_builder.build_all_contexts.return_value = mock_builder_result

    result = analysis_service.build_contexts(sentences)

    mock_context_builder.build_all_contexts.assert_called_once_with(sentences)
    assert result == expected_list


def test_build_contexts_empty(
    analysis_service: AnalysisService, mock_context_builder: MagicMock
) -> None:
    """Test build_contexts with empty sentence list."""
    sentences: List[str] = []
    mock_context_builder.build_all_contexts.return_value = {}

    result = analysis_service.build_contexts(sentences)

    mock_context_builder.build_all_contexts.assert_called_once_with(sentences)
    assert result == []


def test_build_contexts_exception(
    analysis_service: AnalysisService, mock_context_builder: MagicMock
) -> None:
    """Test build_contexts when the context builder raises an exception."""
    sentences: List[str] = ["s1"]
    mock_context_builder.build_all_contexts.side_effect = ValueError("Builder failed")

    with pytest.raises(ValueError, match="Builder failed"):
        analysis_service.build_contexts(sentences)

    mock_context_builder.build_all_contexts.assert_called_once_with(sentences)


# --- Tests for analyze_sentences ---


@pytest.mark.asyncio
async def test_analyze_sentences_success(
    analysis_service: AnalysisService,  # Uses the fixture above
    mock_sentence_analyzer: AsyncMock,  # The *same* base mock instance used by analysis_service
    mock_metrics_tracker: MagicMock,
) -> None:
    """Test success path, configuring the mock analyzer before the call."""
    sentences: List[str] = ["s1", "s2"]
    contexts: List[Dict[str, str]] = [{"c": "1"}, {"c": "2"}]
    mock_analysis_result_s1 = {"analysis": "result1"}
    mock_analysis_result_s2 = {"analysis": "result2"}
    results_to_return = [mock_analysis_result_s1, mock_analysis_result_s2]

    expected_results = [
        {
            "sentence_id": 0,
            "sequence_order": 0,
            "sentence": "s1",
            "analysis": "result1",
        },
        {
            "sentence_id": 1,
            "sequence_order": 1,
            "sentence": "s2",
            "analysis": "result2",
        },
    ]
    mock_timer = create_mock_timer()

    # Configure the classify_sentence method on the *mock instance* used by the service
    mock_sentence_analyzer.classify_sentence.side_effect = results_to_return

    # Call the service method (NO inner patching needed now)
    results = await analysis_service.analyze_sentences(
        sentences, contexts, timer=mock_timer
    )

    # Assert Results
    assert results == expected_results

    # Assert Metrics
    mock_metrics_tracker.increment_errors.assert_not_called()
    assert mock_metrics_tracker.increment_sentences_success.call_count == len(sentences)
    assert mock_metrics_tracker.add_processing_time.call_count == len(sentences)
    mock_metrics_tracker.add_processing_time.assert_any_call(0, 0.5)
    mock_metrics_tracker.add_processing_time.assert_any_call(1, 0.5)


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
    mock_logger.warning.assert_called_with(
        "analyze_sentences called with no sentences. Returning empty list."
    )


@pytest.mark.asyncio
async def test_analyze_sentences_context_mismatch(
    analysis_service: AnalysisService, mock_sentence_analyzer: AsyncMock
) -> None:
    """Test analyze_sentences when sentence and context counts differ."""
    # No timer needed here, no patch needed

    with patch("src.services.analysis_service.logger") as mock_logger:
        results = await analysis_service.analyze_sentences(
            ["s1", "s2"], [{"c": "1"}]
        )  # Mismatch

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
    results = await analysis_service.analyze_sentences(
        sentences, contexts, timer=mock_timer
    )

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
    results = await analysis_service_concurrent.analyze_sentences(
        sentences, contexts, timer=mock_timer
    )

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
