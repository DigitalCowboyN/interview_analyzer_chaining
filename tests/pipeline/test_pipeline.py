from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.io.protocols import (
    ConversationMapStorage,
    SentenceAnalysisWriter,
    TextDataSource,
)

# Import necessary components from the source code
from src.pipeline import (
    PipelineOrchestrator,  # NEW: Import orchestrator
    analyze_specific_sentences,
    # segment_text is likely not needed directly if patched via utils
    run_pipeline,
)
from src.services.analysis_service import AnalysisService

# from src.config import Config # Config likely not needed directly in tests anymore
from src.utils.metrics import MetricsTracker

# from src.utils.text_processing import segment_text # Import locally in test


# Define the missing helper function
def create_mock_analysis(sentence_id: int, sequence_order: int, sentence_text: str) -> Dict[str, Any]:
    """
    Helper function to generate a consistent mock analysis result dictionary.

    Used within pipeline tests to simulate the output of `SentenceAnalyzer`.

    Args:
        sentence_id (int): The ID of the sentence.
        sequence_order (int): The sequence order of the sentence.
        sentence_text (str): The text content of the sentence.

    Returns:
        dict: A dictionary containing mock analysis fields.
    """
    return {
        "sentence_id": sentence_id,
        "sequence_order": sequence_order,
        "sentence": sentence_text,
        "function_type": "mock_declarative",
        "structure_type": "mock_simple",
        # ... other analysis fields ...
    }

# Define missing fixtures


@pytest.fixture
def sample_text_file(tmp_path):
    """
    Pytest fixture creating a temporary sample text file.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        Path: Path object pointing to the created temp file.
    """
    file_content = "First sentence. Second sentence."
    test_file = tmp_path / "test_input.txt"
    test_file.write_text(file_content)
    return test_file


@pytest.fixture
def mock_config(tmp_path: Path) -> Dict[str, Any]:
    """
    Pytest fixture providing a mock configuration dictionary.

    Includes nested structures for paths, pipeline settings, preprocessing,
    classification, and domain keywords used by various components.
    Uses tmp_path to provide realistic, test-specific absolute paths.
    """
    map_dir = tmp_path / "mock_maps"
    output_dir = tmp_path / "mock_output"
    logs_dir = tmp_path / "mock_logs"
    return {
        "paths": {
            "output_dir": str(output_dir),
            "map_dir": str(map_dir),
            "map_suffix": "_map.jsonl",
            "analysis_suffix": "_analysis.jsonl",
            "logs_dir": str(logs_dir)
        },
        "pipeline": {
            "num_analysis_workers": 2,
            "num_concurrent_files": 1  # Add this if orchestrator uses it
        },
        # Add nested structure expected by dependencies
        "preprocessing": {
            "context_windows": {
                "immediate": 1,
                "broader": 3,
                "observer": 5
            }
        },
        "classification": {
            "local": {
                "prompt_files": {
                    "no_context": "mock/path/to/prompts.yaml"
                }
            }
        },
        "domain_keywords": ["mock_keyword"]
    }


@pytest.fixture
def mock_analysis_service() -> MagicMock:
    """
    Provides a mock `AnalysisService` instance with mocked async/sync methods
    and attributes like context_builder.
    """
    mock = MagicMock(spec=AnalysisService)

    # Mock the build_contexts method directly on the service mock
    # This aligns with the call `analysis_service.build_contexts(sentences)` in process_file
    mock.build_contexts = MagicMock(return_value=[
        {"ctx": "mock_ctx_0"},
        {"ctx": "mock_ctx_1"}
        # Tests can override this return value if needed
    ])

    # Also mock the context_builder attribute and its method for analyze_specific_sentences
    mock.context_builder = MagicMock()
    mock.context_builder.build_all_contexts = MagicMock(return_value={
        0: {"ctx": "all_ctx_0"}, 1: {"ctx": "all_ctx_1"}, 2: {"ctx": "all_ctx_2"}, 3: {"ctx": "all_ctx_3"}
        # Use a potentially different return value for easier differentiation if needed
    })

    # Configure analyze_sentences directly on the main mock
    mock.analyze_sentences = AsyncMock(return_value=[{"result": "mock"}])  # Async method
    # Ensure metrics_tracker attribute exists if needed by tests (though pipeline passes it explicitly now)
    mock.metrics_tracker = MagicMock(spec=MetricsTracker)
    return mock


# --- ADDED from tests/test_pipeline.py START ---
# These tests ideally belong in tests/utils/test_text_processing.py
def test_segment_text():
    """Tests `segment_text` for basic sentence splitting (uses default spaCy model)."""
    # Assuming segment_text is available, potentially needs import or patch adjustment
    from src.utils.text_processing import segment_text  # Adjust import if needed
    test_text = "Hello world. How are you today? This pipeline is running well!"
    sentences = segment_text(test_text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you today?"
    assert sentences[2] == "This pipeline is running well!"


def test_segment_text_empty():
    """Tests `segment_text` returns an empty list for empty input."""
    from src.utils.text_processing import segment_text  # Adjust import if needed
    sentences = segment_text("")
    assert sentences == []
# --- ADDED from tests/test_pipeline.py END ---

# --- Test analyze_specific_sentences (Refactored) ---
# These tests check the orchestration logic of analyze_specific_sentences,
# mocking the internal refactored helpers.


@pytest.mark.asyncio
async def test_analyze_specific_sentences_success(
    tmp_path: Path,
    mock_analysis_service: MagicMock
) -> None:
    """Tests analyze_specific_sentences successfully orchestrates analysis."""
    # --- Setup ---
    map_file_path = tmp_path / "mock_map_specific_success.jsonl"
    task_id = "task-specific-ok-new"
    sentence_ids_to_analyze: List[int] = [1, 3]

    # Mock map storage (don't need real file for this orchestration test)
    mock_map_storage = AsyncMock(spec=ConversationMapStorage)
    mock_map_storage.get_identifier.return_value = str(map_file_path)

    # Expected intermediate values from helpers
    all_sentences_text: List[str] = ["S0.", "S1.", "S2.", "S3."]
    target_indices_sorted: List[int] = [1, 3]
    target_sentences_sorted: List[str] = ["S1.", "S3."]
    target_contexts_sorted: List[Dict[str, str]] = [{"ctx": "target_ctx_1"}, {"ctx": "target_ctx_3"}]
    mock_service_results: List[Dict[str, str]] = [
        {"analysis": "for_s1"},
        {"analysis": "for_s3"}
    ]
    expected_final_results: List[Dict[str, Any]] = [
        {"sentence_id": 1, "sequence_order": 1, "analysis": "for_s1"},
        {"sentence_id": 3, "sequence_order": 3, "analysis": "for_s3"}
    ]

    # Configure mock service return value
    mock_analysis_service.analyze_sentences = AsyncMock(return_value=mock_service_results)

    # Patch the internal helpers
    with patch("src.pipeline._prepare_data_for_specific_analysis", new_callable=AsyncMock) as mock_prepare, \
         patch("src.pipeline._build_contexts_for_specific_analysis") as mock_build_ctx, \
         patch("src.pipeline._post_process_specific_results") as mock_post_process, \
         patch("src.pipeline.logger") as mock_logger:

        # Configure mock returns for helpers
        mock_prepare.return_value = (target_sentences_sorted, target_indices_sorted, all_sentences_text)
        mock_build_ctx.return_value = target_contexts_sorted
        mock_post_process.return_value = expected_final_results  # Simulate post-processing

        # --- Execute ---
        final_results = await analyze_specific_sentences(
            map_storage=mock_map_storage,
            sentence_ids=sentence_ids_to_analyze,
            analysis_service=mock_analysis_service,
            task_id=task_id
        )

        # --- Assertions ---
        # Assert helpers were called correctly
        mock_prepare.assert_awaited_once_with(mock_map_storage, sentence_ids_to_analyze, f"[Task {task_id}] ")
        mock_build_ctx.assert_called_once_with(
            all_sentences_text, target_indices_sorted, mock_analysis_service, f"[Task {task_id}] "
        )

        # Assert analysis service was called correctly
        mock_analysis_service.analyze_sentences.assert_awaited_once_with(
            target_sentences_sorted, target_contexts_sorted, task_id=task_id
        )

        # Assert post-processing was called correctly
        mock_post_process.assert_called_once_with(mock_service_results, target_indices_sorted, f"[Task {task_id}] ")

        # Verify logger was called with success messages
        mock_logger.info.assert_called()
        info_calls = [log_call.args[0] for log_call in mock_logger.info.call_args_list]
        assert any("Starting analysis for specific sentences" in log_call for log_call in info_calls), (
            "Pipeline should log analysis start"
        )
        assert any("Finished specific sentence analysis" in log_call for log_call in info_calls), (
            "Pipeline should log analysis completion"
        )

        # Final result check
        assert final_results == expected_final_results


@pytest.mark.asyncio
async def test_analyze_specific_sentences_prepare_error(
    tmp_path: Path,
    mock_analysis_service: MagicMock
) -> None:
    """Tests analyze_specific_sentences handles errors from _prepare_data helper."""
    mock_map_storage = AsyncMock(spec=ConversationMapStorage)
    sentence_ids: List[int] = [1, 5]  # ID 5 will be missing
    task_id = "task-specific-prep-err"
    prepare_error = ValueError("Sentence IDs not found")

    with patch("src.pipeline._prepare_data_for_specific_analysis",
               new_callable=AsyncMock, side_effect=prepare_error) as mock_prepare, \
         patch("src.pipeline._build_contexts_for_specific_analysis") as mock_build_ctx, \
         patch("src.pipeline._post_process_specific_results") as mock_post_process, \
         patch("src.pipeline.logger") as mock_logger:

        with pytest.raises(ValueError, match="Sentence IDs not found"):
            await analyze_specific_sentences(
                map_storage=mock_map_storage,
                sentence_ids=sentence_ids,
                analysis_service=mock_analysis_service,
                task_id=task_id
            )

        mock_prepare.assert_awaited_once()
        mock_build_ctx.assert_not_called()  # Should not be called if prepare fails
        mock_analysis_service.analyze_sentences.assert_not_awaited()
        mock_post_process.assert_not_called()

        # Verify logger was called with the error message
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "Specific analysis failed" in log_message
        assert "Sentence IDs not found" in log_message


# === Unit tests for run_pipeline ===
@pytest.mark.asyncio
async def test_run_pipeline_instantiates_and_executes_orchestrator(mock_config: Dict[str, Any]) -> None:
    """Tests that run_pipeline instantiates PipelineOrchestrator and calls execute."""
    input_dir = "fake/input"
    output_dir = "fake/output"
    map_dir = "fake/map"
    specific_file = "file.txt"
    task_id = "run-task-123"

    # Patch the orchestrator itself
    with patch("src.pipeline.PipelineOrchestrator", autospec=True) as MockPipelineOrchestrator:
        # Get the mock instance that __init__ returns
        mock_orchestrator_instance = MockPipelineOrchestrator.return_value
        # Make execute an AsyncMock
        mock_orchestrator_instance.execute = AsyncMock()

        await run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            specific_file=specific_file,
            config_dict=mock_config,
            task_id=task_id
        )

        # Assert orchestrator was instantiated correctly
        MockPipelineOrchestrator.assert_called_once_with(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config_dict=mock_config,
            task_id=task_id
        )
        # Assert execute was called on the instance
        mock_orchestrator_instance.execute.assert_awaited_once_with(specific_file=specific_file)


@pytest.mark.asyncio
async def test_run_pipeline_handles_orchestrator_init_error(mock_config: Dict[str, Any]) -> None:
    """Tests run_pipeline handles exceptions during orchestrator initialization."""
    input_dir = "fake/input"
    init_error = ValueError("Bad config")

    with patch("src.pipeline.PipelineOrchestrator", side_effect=init_error), \
         patch("src.pipeline.logger") as mock_logger:

        with pytest.raises(ValueError, match="Bad config"):
            await run_pipeline(input_dir=input_dir, config_dict=mock_config)

        mock_logger.critical.assert_called_once()
        log_message = mock_logger.critical.call_args[0][0]
        assert "Pipeline setup failed" in log_message
        assert "Bad config" in log_message


# === Direct PipelineOrchestrator Tests ===
# These tests verify the actual behavior of PipelineOrchestrator methods
# and ensure type compatibility between components

@pytest.mark.asyncio
async def test_pipeline_orchestrator_build_contexts_returns_list(tmp_path: Path, mock_config: Dict[str, Any]) -> None:
    """Tests that _build_contexts actually returns a list format contexts."""
    # Create a real orchestrator with real AnalysisService
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create a test file
    test_file = input_dir / "test.txt"
    test_file.write_text("Hello world. How are you?")

    orchestrator = PipelineOrchestrator(
        input_dir=input_dir,
        config_dict=mock_config,
        task_id="test-build-contexts"
    )

    # Test that _build_contexts returns a list
    sentences: List[str] = ["Hello world.", "How are you?"]
    contexts = orchestrator._build_contexts(sentences, "test.txt")

    # Verify the return type and structure
    assert isinstance(contexts, list), f"Expected list, got {type(contexts)}"
    assert len(contexts) == len(sentences), f"Expected {len(sentences)} contexts, got {len(contexts)}"

    # Verify each context is a dict with expected keys
    for i, context in enumerate(contexts):
        assert isinstance(context, dict), f"Context {i} should be dict, got {type(context)}"
        # Check for expected context keys (based on real AnalysisService output)
        expected_keys = {"immediate", "broader", "observer"}
        assert all(key in context for key in expected_keys), f"Context {i} missing expected keys: {context}"


@pytest.mark.asyncio
async def test_pipeline_orchestrator_analyze_sentences_accepts_list_contexts(
    tmp_path: Path, mock_config: Dict[str, Any]
) -> None:
    """Tests that _analyze_and_save_results accepts list format contexts without type errors."""
    # Create a real orchestrator
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    orchestrator = PipelineOrchestrator(
        input_dir=input_dir,
        config_dict=mock_config,
        task_id="test-analyze-contexts"
    )

    # Create list format contexts (matching what _build_contexts returns)
    sentences: List[str] = ["Hello world.", "How are you?"]
    contexts: List[Dict[str, str]] = [
        {"immediate": "context1", "broader": "context1", "observer": "context1"},
        {"immediate": "context2", "broader": "context2", "observer": "context2"}
    ]

    # Mock the analysis writer to avoid file I/O
    mock_writer = AsyncMock(spec=SentenceAnalysisWriter)
    mock_writer.get_identifier.return_value = "mock_writer"

    # Mock the analysis service to return expected results
    mock_results = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "Hello world.", "analysis": "result1"},
        {"sentence_id": 1, "sequence_order": 1, "sentence": "How are you?", "analysis": "result2"}
    ]
    orchestrator.analysis_service.analyze_sentences = AsyncMock(return_value=mock_results)

    # This should not raise any type errors
    await orchestrator._analyze_and_save_results(
        sentences=sentences,
        contexts=contexts,  # List format - should be accepted
        analysis_writer=mock_writer,
        file_name="test.txt"
    )

    # Verify the analysis service was called with the correct parameters
    orchestrator.analysis_service.analyze_sentences.assert_awaited_once_with(
        sentences, contexts, task_id="test-analyze-contexts"
    )


@pytest.mark.asyncio
async def test_pipeline_orchestrator_type_compatibility_integration(
    tmp_path: Path, mock_config: Dict[str, Any]
) -> None:
    """Integration test verifying the complete data flow with correct types."""
    # Create a real orchestrator with real components
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create a test file
    test_file = input_dir / "test.txt"
    test_file.write_text("Hello world. How are you? This is a test.")

    orchestrator = PipelineOrchestrator(
        input_dir=input_dir,
        config_dict=mock_config,
        task_id="test-integration"
    )

    # Test the complete flow: sentences -> contexts -> analysis
    sentences: List[str] = ["Hello world.", "How are you?", "This is a test."]

    # Step 1: Build contexts (should return list)
    contexts = orchestrator._build_contexts(sentences, "test.txt")
    assert isinstance(contexts, list)
    assert len(contexts) == 3

    # Step 2: Verify contexts can be passed to analysis (should not raise type errors)
    mock_writer = AsyncMock(spec=SentenceAnalysisWriter)
    mock_writer.get_identifier.return_value = "mock_writer"

    # Mock analysis results
    mock_results: List[Dict[str, Any]] = [
        {"sentence_id": i, "sequence_order": i, "sentence": sentences[i], "analysis": f"result{i}"}
        for i in range(len(sentences))
    ]
    orchestrator.analysis_service.analyze_sentences = AsyncMock(return_value=mock_results)

    # This should work without type errors
    await orchestrator._analyze_and_save_results(
        sentences=sentences,
        contexts=contexts,
        analysis_writer=mock_writer,
        file_name="test.txt"
    )

    # Verify the complete flow worked
    orchestrator.analysis_service.analyze_sentences.assert_awaited_once_with(
        sentences, contexts, task_id="test-integration"
    )


def test_pipeline_orchestrator_context_format_consistency(
    tmp_path: Path, mock_config: Dict[str, Any]
) -> None:
    """Tests that the context format is consistent between build and analysis."""
    # Create orchestrator
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    orchestrator = PipelineOrchestrator(
        input_dir=input_dir,
        config_dict=mock_config,
        task_id="test-consistency"
    )

    sentences: List[str] = ["Hello world.", "How are you?"]

    # Build contexts
    contexts = orchestrator._build_contexts(sentences, "test.txt")

    # Verify format consistency
    assert isinstance(contexts, list)
    assert len(contexts) == 2

    # Each context should be a dict with the same structure
    for i, context in enumerate(contexts):
        assert isinstance(context, dict), f"Context {i} should be dict"
        # All contexts should have the same keys
        if i == 0:
            expected_keys = set(context.keys())
        else:
            assert set(context.keys()) == expected_keys, f"Context {i} has different keys: {context.keys()}"

    # Verify the context structure matches what the analysis service expects
    # (This tests the actual integration point)
    assert all(isinstance(ctx, dict) for ctx in contexts), "All contexts should be dicts"
    assert all(len(ctx) > 0 for ctx in contexts), "All contexts should have content"


def test_pipeline_orchestrator_wrong_type_annotations_would_fail(
    tmp_path: Path, mock_config: Dict[str, Any]
) -> None:
    """
    Demonstrates what would happen if we had the wrong type annotations.
    This test shows the value of our type compatibility tests.
    """
    # Create orchestrator
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    orchestrator = PipelineOrchestrator(
        input_dir=input_dir,
        config_dict=mock_config,
        task_id="test-wrong-types"
    )

    sentences: List[str] = ["Hello world.", "How are you?"]

    # Build contexts (returns list format)
    contexts = orchestrator._build_contexts(sentences, "test.txt")

    # Verify it's actually a list (this is what we fixed)
    assert isinstance(contexts, list), "Contexts should be a list"

    # Demonstrate what would happen if we expected dict format:
    # If the type annotations were wrong, this would be the expected behavior:
    # contexts = {0: {...}, 1: {...}}  # Dict format (WRONG)
    # context_0 = contexts[0]  # Would get first context, not context for sentence 0

    # But with the correct list format:
    context_0 = contexts[0]  # Gets context for sentence 0 (CORRECT)
    context_1 = contexts[1]  # Gets context for sentence 1 (CORRECT)

    assert isinstance(context_0, dict), "First context should be a dict"
    assert isinstance(context_1, dict), "Second context should be a dict"

    # Verify we can access contexts by index (list behavior)
    assert len(contexts) == 2, "Should have 2 contexts"
    assert contexts[0] is context_0, "Index 0 should return first context"
    assert contexts[1] is context_1, "Index 1 should return second context"

    # This demonstrates that the list format is correct for the analysis process
    # where contexts are accessed by index, not by sentence ID as keys


@pytest.mark.asyncio
async def test_pipeline_orchestrator_text_data_source_integration(
    tmp_path: Path, mock_config: Dict[str, Any]
) -> None:
    """Tests that the orchestrator correctly uses TextDataSource for reading files."""
    # Create orchestrator
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create a test file with content
    test_file = input_dir / "test.txt"
    test_content = "This is the first sentence. This is the second sentence."
    test_file.write_text(test_content)

    orchestrator = PipelineOrchestrator(
        input_dir=input_dir,
        config_dict=mock_config,
        task_id="test-text-source"
    )

    # Test the _setup_file_io method which creates TextDataSource
    data_source, map_storage, analysis_writer, paths = orchestrator._setup_file_io(test_file)

    # Verify TextDataSource is created correctly
    assert isinstance(data_source, TextDataSource), "Should create TextDataSource instance"
    assert data_source.get_identifier() == str(test_file), "Identifier should match file path"

    # Test reading text through the data source
    read_text = await data_source.read_text()
    assert read_text == test_content, "Should read the correct text content"

    # Test that the data source can be used in the read_and_segment_sentences method
    num_sentences, sentences = await orchestrator._read_and_segment_sentences(data_source)
    assert num_sentences == 2, "Should segment into 2 sentences"
    assert sentences == ["This is the first sentence.", "This is the second sentence."], (
        "Should segment text correctly"
    )
