"""
Unit tests for PipelineOrchestrator execution methods.

Tests the core execution flow: _process_single_file, _process_files_concurrently,
_log_summary, _run_verification, and execute().
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipeline import PipelineOrchestrator, run_pipeline


@pytest.fixture
def mock_orchestrator(tmp_path):
    """Create an orchestrator with mocked dependencies."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    map_dir = tmp_path / "maps"

    for d in [input_dir, output_dir, map_dir]:
        d.mkdir(exist_ok=True)

    # Create a test file
    test_file = input_dir / "test.txt"
    test_file.write_text("This is a test sentence. Here is another one.")

    orchestrator = PipelineOrchestrator(
        input_dir=input_dir,
        output_dir=output_dir,
        map_dir=map_dir,
        task_id="test_task",
    )

    return orchestrator, input_dir, test_file


@pytest.mark.asyncio
class TestProcessSingleFile:
    """Test _process_single_file method."""

    async def test_process_single_file_success(self, mock_orchestrator):
        """Test successful single file processing."""
        orchestrator, input_dir, test_file = mock_orchestrator

        # Mock the internal methods
        orchestrator._setup_file_io = MagicMock(return_value=(
            MagicMock(get_identifier=MagicMock(return_value=str(test_file))),
            MagicMock(get_identifier=MagicMock(return_value="map.jsonl")),
            MagicMock(get_identifier=MagicMock(return_value="analysis.jsonl")),
            MagicMock(),
        ))
        orchestrator._read_and_segment_sentences = AsyncMock(return_value=(2, ["Sentence 1.", "Sentence 2."]))
        orchestrator._write_map_file = AsyncMock()
        orchestrator._build_contexts = MagicMock(return_value=[{"context": "c1"}, {"context": "c2"}])
        orchestrator._analyze_and_save_results = AsyncMock()
        orchestrator.event_emitter = None  # No event emitter

        await orchestrator._process_single_file(test_file)

        # Verify method calls
        orchestrator._setup_file_io.assert_called_once()
        orchestrator._read_and_segment_sentences.assert_called_once()
        orchestrator._write_map_file.assert_called_once()
        orchestrator._build_contexts.assert_called_once()
        orchestrator._analyze_and_save_results.assert_called_once()

    async def test_process_single_file_empty_file(self, mock_orchestrator):
        """Test processing file with no sentences."""
        orchestrator, input_dir, test_file = mock_orchestrator

        # Create mock map storage
        mock_map_storage = AsyncMock()
        mock_map_storage.get_identifier.return_value = "map.jsonl"
        mock_map_storage.initialize = AsyncMock()
        mock_map_storage.finalize = AsyncMock()

        orchestrator._setup_file_io = MagicMock(return_value=(
            MagicMock(get_identifier=MagicMock(return_value=str(test_file))),
            mock_map_storage,
            MagicMock(get_identifier=MagicMock(return_value="analysis.jsonl")),
            MagicMock(),
        ))
        orchestrator._read_and_segment_sentences = AsyncMock(return_value=(0, []))
        orchestrator._write_map_file = AsyncMock()
        orchestrator._build_contexts = MagicMock()
        orchestrator._analyze_and_save_results = AsyncMock()
        orchestrator.event_emitter = None

        await orchestrator._process_single_file(test_file)

        # For empty files, should NOT call write_map_file, build_contexts, or analyze
        orchestrator._write_map_file.assert_not_called()
        orchestrator._build_contexts.assert_not_called()
        orchestrator._analyze_and_save_results.assert_not_called()

        # But should initialize and finalize the map storage
        mock_map_storage.initialize.assert_called_once()
        mock_map_storage.finalize.assert_called_once()

    async def test_process_single_file_with_event_emitter(self, mock_orchestrator):
        """Test file processing with event emitter."""
        orchestrator, input_dir, test_file = mock_orchestrator

        # Mock event emitter
        mock_emitter = AsyncMock()
        mock_emitter.emit_interview_created = AsyncMock()
        orchestrator.event_emitter = mock_emitter

        orchestrator._setup_file_io = MagicMock(return_value=(
            MagicMock(get_identifier=MagicMock(return_value=str(test_file))),
            MagicMock(get_identifier=MagicMock(return_value="map.jsonl")),
            MagicMock(get_identifier=MagicMock(return_value="analysis.jsonl")),
            MagicMock(),
        ))
        orchestrator._read_and_segment_sentences = AsyncMock(return_value=(1, ["Sentence."]))
        orchestrator._write_map_file = AsyncMock()
        orchestrator._build_contexts = MagicMock(return_value=[{"context": "c"}])
        orchestrator._analyze_and_save_results = AsyncMock()

        await orchestrator._process_single_file(test_file)

        # Event emitter should be called
        mock_emitter.emit_interview_created.assert_called_once()

    async def test_process_single_file_event_emission_failure(self, mock_orchestrator):
        """Test that event emission failure aborts processing."""
        orchestrator, input_dir, test_file = mock_orchestrator

        # Mock event emitter that fails
        mock_emitter = AsyncMock()
        mock_emitter.emit_interview_created = AsyncMock(side_effect=Exception("Event store unavailable"))
        orchestrator.event_emitter = mock_emitter

        with pytest.raises(RuntimeError, match="Event emission failed"):
            await orchestrator._process_single_file(test_file)

    async def test_process_single_file_analysis_error(self, mock_orchestrator):
        """Test error handling during analysis."""
        orchestrator, input_dir, test_file = mock_orchestrator

        orchestrator._setup_file_io = MagicMock(return_value=(
            MagicMock(get_identifier=MagicMock(return_value=str(test_file))),
            MagicMock(get_identifier=MagicMock(return_value="map.jsonl")),
            MagicMock(get_identifier=MagicMock(return_value="analysis.jsonl")),
            MagicMock(),
        ))
        orchestrator._read_and_segment_sentences = AsyncMock(return_value=(1, ["Sentence."]))
        orchestrator._write_map_file = AsyncMock()
        orchestrator._build_contexts = MagicMock(return_value=[{"context": "c"}])
        orchestrator._analyze_and_save_results = AsyncMock(side_effect=Exception("Analysis failed"))
        orchestrator.event_emitter = None

        with pytest.raises(Exception, match="Analysis failed"):
            await orchestrator._process_single_file(test_file)


@pytest.mark.asyncio
class TestProcessFilesConcurrently:
    """Test _process_files_concurrently method."""

    async def test_process_files_concurrently_success(self, mock_orchestrator):
        """Test concurrent processing of multiple files."""
        orchestrator, input_dir, _ = mock_orchestrator

        # Create multiple test files
        files = []
        for i in range(3):
            f = input_dir / f"test_{i}.txt"
            f.write_text(f"Sentence {i}.")
            files.append(f)

        # Mock _process_single_file to return quickly
        orchestrator._process_single_file = AsyncMock()

        results = await orchestrator._process_files_concurrently(files)

        assert len(results) == 3
        assert orchestrator._process_single_file.call_count == 3

    async def test_process_files_concurrently_with_failure(self, mock_orchestrator):
        """Test that failures in concurrent processing are captured."""
        orchestrator, input_dir, _ = mock_orchestrator

        files = []
        for i in range(3):
            f = input_dir / f"test_{i}.txt"
            f.write_text(f"Sentence {i}.")
            files.append(f)

        # Mock _process_single_file to fail on second file
        call_count = [0]

        async def mock_process(fp):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Processing failed")

        orchestrator._process_single_file = mock_process

        results = await orchestrator._process_files_concurrently(files)

        # Should have results for all files (some may be exceptions)
        assert len(results) == 3


@pytest.mark.asyncio
class TestLogSummary:
    """Test _log_summary method."""

    def test_log_summary_all_success(self, mock_orchestrator):
        """Test summary logging when all files succeed."""
        orchestrator, input_dir, _ = mock_orchestrator

        files = [input_dir / "a.txt", input_dir / "b.txt"]
        results = [files[0], files[1]]  # Paths indicate success

        # Should not raise
        orchestrator._log_summary(results, files)

    def test_log_summary_with_failures(self, mock_orchestrator):
        """Test summary logging with some failures."""
        orchestrator, input_dir, _ = mock_orchestrator

        files = [input_dir / "a.txt", input_dir / "b.txt", input_dir / "c.txt"]
        results = [files[0], Exception("Failed"), files[2]]

        # Should not raise, but should log warnings
        orchestrator._log_summary(results, files)

    def test_log_summary_all_failures(self, mock_orchestrator):
        """Test summary logging when all files fail."""
        orchestrator, input_dir, _ = mock_orchestrator

        files = [input_dir / "a.txt", input_dir / "b.txt"]
        results = [Exception("Failed 1"), Exception("Failed 2")]

        # Should not raise
        orchestrator._log_summary(results, files)


@pytest.mark.asyncio
class TestRunVerification:
    """Test _run_verification method."""

    async def test_run_verification_no_files(self, mock_orchestrator):
        """Test verification with no files."""
        orchestrator, _, _ = mock_orchestrator

        # Should not raise with empty list
        await orchestrator._run_verification([])

    async def test_run_verification_success(self, mock_orchestrator):
        """Test successful verification."""
        orchestrator, input_dir, test_file = mock_orchestrator

        # Create map and analysis files
        map_file = orchestrator.map_dir_path / "test_map.jsonl"
        analysis_file = orchestrator.output_dir_path / "test_analysis.jsonl"

        import json

        # Write matching map and analysis files
        with map_file.open("w") as f:
            f.write(json.dumps({"sentence_id": 0, "sentence": "Test."}) + "\n")

        with analysis_file.open("w") as f:
            f.write(json.dumps({"sentence_id": 0, "sentence": "Test.", "analysis": {}}) + "\n")

        await orchestrator._run_verification([test_file])

    async def test_run_verification_missing_analysis(self, mock_orchestrator):
        """Test verification when analysis file is missing entries."""
        orchestrator, input_dir, test_file = mock_orchestrator

        # Create map file with entries but no analysis file
        map_file = orchestrator.map_dir_path / "test_map.jsonl"

        import json

        with map_file.open("w") as f:
            f.write(json.dumps({"sentence_id": 0, "sentence": "Test."}) + "\n")
            f.write(json.dumps({"sentence_id": 1, "sentence": "Another."}) + "\n")

        # Create empty analysis file
        analysis_file = orchestrator.output_dir_path / "test_analysis.jsonl"
        analysis_file.touch()

        # Should complete without raising (just log warnings)
        await orchestrator._run_verification([test_file])


@pytest.mark.asyncio
class TestExecute:
    """Test execute() method."""

    async def test_execute_no_files_found(self, mock_orchestrator):
        """Test execute when no files are found."""
        orchestrator, input_dir, test_file = mock_orchestrator

        # Remove test file
        test_file.unlink()

        # Should complete without error
        await orchestrator.execute()

    async def test_execute_full_flow(self, mock_orchestrator):
        """Test full execute flow with mocked processing."""
        orchestrator, input_dir, test_file = mock_orchestrator

        # Mock the processing methods
        orchestrator._process_files_concurrently = AsyncMock(return_value=[test_file])
        orchestrator._run_verification = AsyncMock()
        orchestrator._log_summary = MagicMock()

        await orchestrator.execute()

        orchestrator._process_files_concurrently.assert_called_once()
        orchestrator._run_verification.assert_called_once()
        orchestrator._log_summary.assert_called_once()

    async def test_execute_specific_file(self, mock_orchestrator):
        """Test execute with specific file targeting."""
        orchestrator, input_dir, test_file = mock_orchestrator

        orchestrator._process_files_concurrently = AsyncMock(return_value=[test_file])
        orchestrator._run_verification = AsyncMock()
        orchestrator._log_summary = MagicMock()

        await orchestrator.execute(specific_file="test.txt")

        # Should still process
        orchestrator._process_files_concurrently.assert_called_once()


@pytest.mark.asyncio
class TestRunPipelineFunction:
    """Test the standalone run_pipeline function."""

    async def test_run_pipeline_creates_orchestrator(self, tmp_path):
        """Test that run_pipeline creates and runs an orchestrator."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create a test file
        test_file = input_dir / "test.txt"
        test_file.write_text("Test sentence.")

        with patch.object(PipelineOrchestrator, "execute", new_callable=AsyncMock) as mock_execute:
            await run_pipeline(input_dir=input_dir, task_id="test_run")
            mock_execute.assert_called_once()

    async def test_run_pipeline_propagates_errors(self, tmp_path):
        """Test that run_pipeline propagates orchestrator errors."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            await run_pipeline(input_dir=nonexistent)


@pytest.mark.asyncio
class TestAnalyzeAndSaveResults:
    """Test _analyze_and_save_results method."""

    async def test_analyze_and_save_results_success(self, mock_orchestrator):
        """Test successful analysis and save."""
        orchestrator, _, _ = mock_orchestrator

        sentences = ["Test sentence."]
        contexts = [{"context": "ctx"}]
        mock_writer = AsyncMock()
        mock_writer.get_identifier.return_value = "analysis.jsonl"

        # Mock the analysis service
        orchestrator.analysis_service = AsyncMock()
        orchestrator.analysis_service.analyze_sentences = AsyncMock(
            return_value=[{"sentence_id": 0, "sentence": "Test sentence."}]
        )
        orchestrator._save_analysis_results = AsyncMock()

        await orchestrator._analyze_and_save_results(
            sentences, contexts, mock_writer, "test.txt"
        )

        orchestrator.analysis_service.analyze_sentences.assert_called_once()
        orchestrator._save_analysis_results.assert_called_once()

    async def test_analyze_and_save_results_with_ids(self, mock_orchestrator):
        """Test analysis with interview_id and correlation_id."""
        orchestrator, _, _ = mock_orchestrator

        sentences = ["Test sentence."]
        contexts = [{"context": "ctx"}]
        mock_writer = AsyncMock()
        mock_writer.get_identifier.return_value = "analysis.jsonl"

        orchestrator.analysis_service = AsyncMock()
        orchestrator.analysis_service.analyze_sentences = AsyncMock(
            return_value=[{"sentence_id": 0, "sentence": "Test sentence."}]
        )
        orchestrator._save_analysis_results = AsyncMock()

        await orchestrator._analyze_and_save_results(
            sentences, contexts, mock_writer, "test.txt",
            interview_id="int-123", correlation_id="corr-456"
        )

        # Verify IDs are passed to save
        call_kwargs = orchestrator._save_analysis_results.call_args.kwargs
        assert call_kwargs.get("interview_id") == "int-123"
        assert call_kwargs.get("correlation_id") == "corr-456"

    async def test_analyze_and_save_results_error(self, mock_orchestrator):
        """Test error handling in analyze and save."""
        orchestrator, _, _ = mock_orchestrator

        sentences = ["Test sentence."]
        contexts = [{"context": "ctx"}]
        mock_writer = AsyncMock()
        mock_writer.get_identifier.return_value = "analysis.jsonl"

        orchestrator.analysis_service = AsyncMock()
        orchestrator.analysis_service.analyze_sentences = AsyncMock(
            side_effect=Exception("Analysis service failed")
        )

        with pytest.raises(Exception, match="Analysis service failed"):
            await orchestrator._analyze_and_save_results(
                sentences, contexts, mock_writer, "test.txt"
            )
