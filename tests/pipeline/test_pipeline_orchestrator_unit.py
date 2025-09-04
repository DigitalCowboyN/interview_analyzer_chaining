"""
tests/pipeline/test_pipeline_orchestrator_unit.py

Unit tests for PipelineOrchestrator class methods using real data and minimal mocking.

These tests follow the cardinal rules:
1. Test actual functionality, not mock interactions
2. Use realistic data derived from real processing, not hardcoded values
3. Only mock external dependencies that would be expensive or unreliable in tests
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline import (
    PipelineOrchestrator,
    _build_contexts_for_specific_analysis,
    _log_prefix,
    _post_process_specific_results,
    _prepare_data_for_specific_analysis,
)
from tests.pipeline.conftest import (
    create_realistic_analysis_results,
    create_realistic_map_entries,
)


class TestLogPrefix:
    """Test the _log_prefix helper function with various inputs."""

    def test_log_prefix_functionality(self):
        """Test log prefix generation with realistic task IDs."""
        test_cases = [
            ("pipeline_task_001", "[Task pipeline_task_001] "),
            ("user-analysis-2024", "[Task user-analysis-2024] "),
            ("batch_process_123", "[Task batch_process_123] "),
            (None, ""),
            ("", ""),
        ]

        for task_id, expected in test_cases:
            result = _log_prefix(task_id)
            assert result == expected


class TestPipelineOrchestratorInitialization:
    """Test PipelineOrchestrator initialization with realistic configurations."""

    def test_orchestrator_initialization_with_realistic_config(self, tmp_path, realistic_config):
        """Test orchestrator initialization with realistic configuration."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        map_dir = tmp_path / "maps"

        # Create directories
        for directory in [input_dir, output_dir, map_dir]:
            directory.mkdir(exist_ok=True)

        orchestrator = PipelineOrchestrator(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config_dict=realistic_config,
            task_id="realistic_test",
        )

        # Test that actual attributes are set correctly
        assert orchestrator.input_dir_path == input_dir
        assert orchestrator.output_dir_path == output_dir
        assert orchestrator.map_dir_path == map_dir
        assert orchestrator.task_id == "realistic_test"
        assert orchestrator.prefix == "[Task realistic_test] "
        assert orchestrator.config == realistic_config

        # Test that analysis service is properly initialized
        assert orchestrator.analysis_service is not None
        assert hasattr(orchestrator.analysis_service, "analyze_sentences")
        assert hasattr(orchestrator.analysis_service, "context_builder")

        # Test that metrics tracker is available
        assert orchestrator.metrics_tracker is not None

    def test_orchestrator_with_default_paths(self, tmp_path, realistic_config):
        """Test orchestrator initialization with default path behavior."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        orchestrator = PipelineOrchestrator(
            input_dir=input_dir,
            config_dict=realistic_config,
        )

        # Should use config defaults for output and map directories
        expected_output_dir = Path(realistic_config["paths"]["output_dir"])
        expected_map_dir = Path(realistic_config["paths"]["map_dir"])

        assert orchestrator.output_dir_path == expected_output_dir
        assert orchestrator.map_dir_path == expected_map_dir

    def test_orchestrator_error_with_invalid_input_path(self, tmp_path):
        """Test error handling with invalid input directory."""
        nonexistent_path = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError):
            PipelineOrchestrator(input_dir=nonexistent_path)


class TestPipelineOrchestratorFileOperations:
    """Test file discovery and I/O setup with real filesystem operations."""

    def test_file_discovery_with_mixed_file_types(self, tmp_path):
        """Test file discovery filters correctly by file type."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        # Create various file types
        text_files = ["document1.txt", "analysis.txt", "report.txt"]
        other_files = ["image.jpg", "data.csv", "config.json", "readme.md"]

        for filename in text_files + other_files:
            (input_dir / filename).write_text(f"Content of {filename}")

        orchestrator = PipelineOrchestrator(input_dir=input_dir)
        discovered_files = orchestrator._discover_files_to_process()

        # Should only find .txt files
        discovered_names = {f.name for f in discovered_files}
        expected_names = set(text_files)

        assert discovered_names == expected_names
        assert len(discovered_files) == len(text_files)

    def test_file_discovery_specific_file_targeting(self, tmp_path, multiple_text_files):
        """Test targeting a specific file among many."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        # Copy multiple files to input directory
        file_names = []
        for i, source_file in enumerate(multiple_text_files):
            target_name = f"doc_{i}.txt"
            target_file = input_dir / target_name
            target_file.write_text(source_file.read_text())
            file_names.append(target_name)

        orchestrator = PipelineOrchestrator(input_dir=input_dir)

        # Target the first file specifically
        target_file = file_names[0]
        discovered_files = orchestrator._discover_files_to_process(specific_file=target_file)

        assert len(discovered_files) == 1
        assert discovered_files[0].name == target_file

    def test_file_io_setup_creates_proper_components(self, tmp_path, realistic_config):
        """Test that file I/O setup creates properly configured components."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        map_dir = tmp_path / "maps"

        for directory in [input_dir, output_dir, map_dir]:
            directory.mkdir(exist_ok=True)

        test_file = input_dir / "test_document.txt"
        test_file.write_text("Test content for I/O setup validation.")

        orchestrator = PipelineOrchestrator(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config_dict=realistic_config,
        )

        data_source, map_storage, analysis_writer, paths = orchestrator._setup_file_io(test_file)

        # Verify components are properly configured
        assert data_source.get_identifier() == str(test_file)
        assert "test_document_map.jsonl" in map_storage.get_identifier()
        assert "test_document_analysis.jsonl" in analysis_writer.get_identifier()

        # Verify paths are realistic
        assert paths.map_file.parent == map_dir
        assert paths.analysis_file.parent == output_dir


class TestPipelineOrchestratorTextProcessing:
    """Test text reading and processing with real content."""

    async def test_text_reading_and_segmentation_integration(self, tmp_path, sample_text_content, expected_sentences):
        """Test reading and segmenting text with real content and segmentation."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        test_file = input_dir / "real_content.txt"
        test_file.write_text(sample_text_content)

        orchestrator = PipelineOrchestrator(input_dir=input_dir)

        # Setup real data source
        data_source, _, _, _ = orchestrator._setup_file_io(test_file)

        # Test actual reading and segmentation
        num_sentences, sentences = await orchestrator._read_and_segment_sentences(data_source)

        # Validate against expected results from real segmentation
        assert num_sentences == len(expected_sentences)
        assert sentences == expected_sentences

        # Verify all sentences are non-empty strings
        assert all(isinstance(s, str) and len(s.strip()) > 0 for s in sentences)

    async def test_map_file_writing_with_real_sentences(self, tmp_path, expected_sentences):
        """Test map file writing with real sentence data."""
        input_dir = tmp_path / "input"
        map_dir = tmp_path / "maps"

        for directory in [input_dir, map_dir]:
            directory.mkdir(exist_ok=True)

        test_file = input_dir / "map_test.txt"
        test_file.write_text(" ".join(expected_sentences))

        orchestrator = PipelineOrchestrator(
            input_dir=input_dir,
            map_dir=map_dir,
        )

        # Setup map storage
        _, map_storage, _, _ = orchestrator._setup_file_io(test_file)

        # Write map file with real sentences
        await orchestrator._write_map_file(
            expected_sentences,
            map_storage,
            test_file.name,
        )

        # Verify map file was created and contains correct data
        expected_map_file = map_dir / "map_test_map.jsonl"
        assert expected_map_file.exists()

        # Read and validate map file content
        import json

        with expected_map_file.open() as f:
            lines = f.readlines()

        assert len(lines) == len(expected_sentences)

        for i, line in enumerate(lines):
            entry = json.loads(line.strip())
            assert entry["sentence_id"] == i
            assert entry["sequence_order"] == i
            assert entry["sentence"] == expected_sentences[i]

    async def test_context_building_with_real_analysis_service(
        self, tmp_path, expected_sentences, real_analysis_service_with_mocked_llm
    ):
        """Test context building with real analysis service and real sentences."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        orchestrator = PipelineOrchestrator(input_dir=input_dir)

        # Use real analysis service instead of mocks
        orchestrator.analysis_service = real_analysis_service_with_mocked_llm

        # Test context building with real sentences
        contexts = orchestrator._build_contexts(expected_sentences, "test_document.txt")

        # Validate context structure
        assert isinstance(contexts, list)
        assert len(contexts) == len(expected_sentences)

        # Validate context content is realistic
        for i, context in enumerate(contexts):
            assert isinstance(context, dict)

            # Check for expected context window keys from real ContextBuilder
            expected_keys = {
                "structure_analysis",
                "immediate_context",
                "observer_context",
                "broader_context",
                "overall_context",
            }
            assert all(key in context for key in expected_keys)

            # Verify context contains actual sentences from our test data
            for key, context_sentences in context.items():
                assert isinstance(context_sentences, list)
                for ctx_sentence in context_sentences:
                    if ctx_sentence.strip():  # Skip empty context
                        assert ctx_sentence in expected_sentences


class TestSpecificAnalysisHelperFunctions:
    """Test helper functions for specific sentence analysis with realistic data."""

    async def test_prepare_data_with_realistic_map_entries(self, expected_sentences):
        """Test data preparation with realistic map entries derived from real sentences."""
        # Create realistic map storage
        mock_map_storage = MagicMock()
        mock_map_storage.get_identifier.return_value = "realistic_test_map.jsonl"

        # Use real sentences to create map entries
        map_entries = create_realistic_map_entries(expected_sentences, "test_document.txt")
        mock_map_storage.read_all_entries = AsyncMock(return_value=map_entries)

        # Test with realistic sentence IDs
        sentence_ids_to_analyze = [0, 2] if len(expected_sentences) >= 3 else [0]
        prefix = "[Test Analysis] "

        target_sentences, target_indices, full_context_list = await _prepare_data_for_specific_analysis(
            mock_map_storage, sentence_ids_to_analyze, prefix
        )

        # Validate results are based on real data
        assert len(target_sentences) == len(sentence_ids_to_analyze)
        assert target_indices == sorted(sentence_ids_to_analyze)

        # Verify sentences match expected content
        for i, sentence_id in enumerate(sorted(sentence_ids_to_analyze)):
            assert target_sentences[i] == expected_sentences[sentence_id]

        # Verify full context list contains all sentences
        assert len(full_context_list) == len(expected_sentences)
        for i, sentence in enumerate(expected_sentences):
            assert full_context_list[i] == sentence

    def test_context_building_for_specific_analysis(self, expected_sentences, real_analysis_service_with_mocked_llm):
        """Test context building for specific analysis with real service."""
        full_sentence_list = expected_sentences
        target_indices = [0, 2] if len(expected_sentences) >= 3 else [0]
        prefix = "[Context Test] "

        target_contexts = _build_contexts_for_specific_analysis(
            full_sentence_list,
            target_indices,
            real_analysis_service_with_mocked_llm,
            prefix,
        )

        # Validate context structure
        assert len(target_contexts) == len(target_indices)

        for context in target_contexts:
            assert isinstance(context, dict)
            # Should have context windows from real ContextBuilder
            expected_keys = {
                "structure_analysis",
                "immediate_context",
                "observer_context",
                "broader_context",
                "overall_context",
            }
            assert all(key in context for key in expected_keys)

    def test_post_processing_with_realistic_results(self, expected_sentences):
        """Test post-processing with realistic analysis results."""
        # Create realistic analysis results
        analysis_results = create_realistic_analysis_results(expected_sentences[:2])
        target_indices = [5, 10]  # Different from original indices
        prefix = "[Post Process] "

        final_results = _post_process_specific_results(analysis_results, target_indices, prefix)

        # Validate post-processing worked correctly
        assert len(final_results) == len(target_indices)

        for i, result in enumerate(final_results):
            # Verify IDs were remapped correctly
            assert result["sentence_id"] == target_indices[i]
            assert result["sequence_order"] == target_indices[i]

            # Verify original sentence content is preserved
            assert result["sentence"] == expected_sentences[i]

            # Verify realistic analysis fields are present
            assert "word_count" in result
            assert "character_count" in result
            assert "function_type" in result


class TestPipelineOrchestratorErrorHandling:
    """Test error handling with realistic scenarios."""

    async def test_empty_file_handling(self, tmp_path):
        """Test handling of empty files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        empty_file = input_dir / "empty.txt"
        empty_file.write_text("")

        orchestrator = PipelineOrchestrator(input_dir=input_dir)

        # Setup I/O for empty file
        data_source, _, _, _ = orchestrator._setup_file_io(empty_file)

        # Should handle empty content gracefully
        num_sentences, sentences = await orchestrator._read_and_segment_sentences(data_source)

        assert num_sentences == 0
        assert sentences == []

    async def test_file_read_error_handling(self, tmp_path):
        """Test error handling when file reading fails."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        # Create a mock data source that fails to read
        mock_data_source = MagicMock()
        mock_data_source.read_text = AsyncMock(side_effect=IOError("File read failed"))

        orchestrator = PipelineOrchestrator(input_dir=input_dir)

        # Should propagate the read error
        with pytest.raises(IOError, match="File read failed"):
            await orchestrator._read_and_segment_sentences(mock_data_source)

    def test_invalid_map_entries_handling(self, expected_sentences):
        """Test handling of invalid entries in map storage."""
        mock_map_storage = MagicMock()
        mock_map_storage.get_identifier.return_value = "mixed_validity_map.jsonl"

        # Mix valid and invalid entries
        valid_entries = create_realistic_map_entries(expected_sentences[:2])
        invalid_entries = [
            {"sentence_id": "not_an_integer", "sequence_order": 2, "sentence": "Invalid ID"},
            {"sentence_id": 3, "sequence_order": 3},  # Missing sentence
            {"sequence_order": 4, "sentence": "Missing ID"},  # Missing sentence_id
        ]

        all_entries = valid_entries + invalid_entries
        mock_map_storage.read_all_entries = AsyncMock(return_value=all_entries)

        # Should work with valid entries only
        async def test_with_invalid_entries():
            sentence_ids = [0, 1]  # Only request valid IDs
            prefix = "[Error Test] "

            target_sentences, target_indices, full_context_list = await _prepare_data_for_specific_analysis(
                mock_map_storage, sentence_ids, prefix
            )

            # Should get results for valid entries only
            assert len(target_sentences) == 2
            assert target_indices == [0, 1]
            assert target_sentences == expected_sentences[:2]

        # Run the async test
        import asyncio

        asyncio.run(test_with_invalid_entries())
