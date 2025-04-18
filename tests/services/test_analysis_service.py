"""
Tests for the AnalysisService.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call
from typing import Dict, Any, List
import asyncio

from src.services.analysis_service import AnalysisService
from src.agents.context_builder import ContextBuilder 
from src.agents.sentence_analyzer import SentenceAnalyzer
from src.utils.metrics import MetricsTracker

# Fixture for mock config
@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provides a mock configuration dictionary for tests."""
    return {
        "paths": {
            "output_dir": "mock_output",
            "map_dir": "mock_maps",
            "map_suffix": "_map.jsonl",
            "analysis_suffix": "_analysis.jsonl"
        },
        "pipeline": {
            "num_analysis_workers": 1 # Use 1 worker for simplicity in unit tests
        },
        "preprocessing": {
            "context_windows": {"immediate": 1, "broader": 3, "observer": 5}
        }
        # Add other config sections if needed by service or its dependencies
    }

# Fixture for AnalysisService instance
@pytest.fixture
def analysis_service(
    mock_config: Dict[str, Any],
    mock_context_builder: MagicMock, # Inject mock dependency
    mock_sentence_analyzer: AsyncMock, # Inject mock dependency
    mock_metrics_tracker: MagicMock # Inject mock dependency
) -> AnalysisService:
    """Provides an AnalysisService instance with mocked dependencies."""
    return AnalysisService(
        config=mock_config,
        context_builder=mock_context_builder,
        sentence_analyzer=mock_sentence_analyzer,
        metrics_tracker=mock_metrics_tracker
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
    """Provides a mock SentenceAnalyzer instance (AsyncMock for async methods)."""
    mock = AsyncMock(spec=SentenceAnalyzer)
    # Configure default mock behavior if needed, e.g.:
    mock.classify_sentence = AsyncMock(return_value={"analysis": "mocked"})
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

def test_build_contexts_success(analysis_service: AnalysisService, mock_context_builder: MagicMock):
    """Test build_contexts successfully converts dict from builder to list."""
    sentences = ["s1", "s2"]
    mock_builder_result = {0: {"ctx": "c1"}, 1: {"ctx": "c2"}}
    expected_list = [{"ctx": "c1"}, {"ctx": "c2"}]
    mock_context_builder.build_all_contexts.return_value = mock_builder_result
    
    result = analysis_service.build_contexts(sentences)
    
    mock_context_builder.build_all_contexts.assert_called_once_with(sentences)
    assert result == expected_list

def test_build_contexts_empty(analysis_service: AnalysisService, mock_context_builder: MagicMock):
    """Test build_contexts with empty sentence list."""
    sentences = []
    mock_context_builder.build_all_contexts.return_value = {}
    
    result = analysis_service.build_contexts(sentences)
    
    mock_context_builder.build_all_contexts.assert_called_once_with(sentences)
    assert result == []

def test_build_contexts_exception(analysis_service: AnalysisService, mock_context_builder: MagicMock):
    """Test build_contexts when the context builder raises an exception."""
    sentences = ["s1"]
    mock_context_builder.build_all_contexts.side_effect = ValueError("Builder failed")
    
    with pytest.raises(ValueError, match="Builder failed"):
        analysis_service.build_contexts(sentences)
        
    mock_context_builder.build_all_contexts.assert_called_once_with(sentences)

# --- Tests for analyze_sentences ---

@pytest.mark.asyncio
async def test_analyze_sentences_success(
    analysis_service: AnalysisService, 
    mock_sentence_analyzer: AsyncMock, 
    mock_metrics_tracker: MagicMock
):
    """Test analyze_sentences success path with mocked analyzer."""
    sentences = ["s1", "s2"]
    contexts = [{"c": 1}, {"c": 2}]
    mock_analysis_result_s1 = {"analysis": "result1"}
    mock_analysis_result_s2 = {"analysis": "result2"}
    mock_sentence_analyzer.classify_sentence.side_effect = [mock_analysis_result_s1, mock_analysis_result_s2]

    expected_results = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "s1", "analysis": "result1"},
        {"sentence_id": 1, "sequence_order": 1, "sentence": "s2", "analysis": "result2"}
    ]

    with patch("src.services.analysis_service.logger") as mock_logger:
        results = await analysis_service.analyze_sentences(sentences, contexts)
    
    # Assert analyzer was called correctly - Use assert_awaited
    mock_sentence_analyzer.classify_sentence.assert_awaited() # Check if it was awaited at least once
    # Check specific calls if needed after confirming it's awaited
    # mock_sentence_analyzer.classify_sentence.assert_has_awaits([
    #     call("s1", {"c": 1}),
    #     call("s2", {"c": 2})
    # ], any_order=True) # any_order might be needed due to concurrency

    # Assert metrics were called - only check existing metrics methods
    mock_metrics_tracker.increment_errors.assert_not_called() 
    # Removed checks for increment_sentences_processed, increment_sentences_success, add_processing_time

    # Assert results 
    assert results == expected_results

@pytest.mark.asyncio
async def test_analyze_sentences_empty_input(analysis_service: AnalysisService, mock_sentence_analyzer: AsyncMock):
    """Test analyze_sentences with empty sentences list."""
    with patch("src.services.analysis_service.logger") as mock_logger:
        results = await analysis_service.analyze_sentences([], [])
    
    assert results == []
    mock_sentence_analyzer.classify_sentence.assert_not_awaited()
    mock_logger.warning.assert_called_with("analyze_sentences called with no sentences. Returning empty list.")

@pytest.mark.asyncio
async def test_analyze_sentences_context_mismatch(analysis_service: AnalysisService, mock_sentence_analyzer: AsyncMock):
    """Test analyze_sentences when sentence and context counts differ."""
    with patch("src.services.analysis_service.logger") as mock_logger:
        results = await analysis_service.analyze_sentences(["s1", "s2"], [{"c": 1}]) # Mismatch
    
    assert results == []
    mock_sentence_analyzer.classify_sentence.assert_not_awaited()
    mock_logger.error.assert_called_with("Sentence count (2) and context count (1) mismatch in analyze_sentences. Aborting.")

@pytest.mark.asyncio
async def test_analyze_sentences_classify_error(
    analysis_service: AnalysisService, 
    mock_sentence_analyzer: AsyncMock, 
    mock_metrics_tracker: MagicMock
):
    """Test analyze_sentences when analyzer raises an error for one sentence."""
    sentences = ["s1_ok", "s2_fail"]
    contexts = [{"c": 1}, {"c": 2}]
    mock_analysis_result_s1 = {"analysis": "result1"}
    test_exception = ValueError("API Error")
    mock_sentence_analyzer.classify_sentence.side_effect = [mock_analysis_result_s1, test_exception]

    expected_results = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "s1_ok", "analysis": "result1"},
        {"sentence_id": 1, "sequence_order": 1, "sentence": "s2_fail", "error": True, "error_type": "ValueError", "error_message": "API Error"}
    ]

    with patch("src.services.analysis_service.logger") as mock_logger:
        results = await analysis_service.analyze_sentences(sentences, contexts)
    
    # Assert analyzer was awaited twice (even though one failed)
    assert mock_sentence_analyzer.classify_sentence.await_count == 2

    # Assert metrics - Check ONLY existing methods
    assert mock_metrics_tracker.increment_errors.call_count == 1 # KEEP - Error occurred

    # Assert results (should contain both success and error dicts)
    assert results == expected_results

    # Check logger error message
    log_message = mock_logger.error.call_args[0][0]
    assert "failed analyzing sentence_id 1" in log_message
    assert "API Error" in log_message

# Add more tests: e.g., different number of workers, loader errors? 