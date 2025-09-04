"""
tests/pipeline/test_pipeline_integration.py

Integration tests for the complete pipeline workflow.

These tests validate end-to-end pipeline functionality with real file operations,
actual text processing, and realistic data flow. No hardcoded mock values.
"""

import pytest

from src.pipeline import PipelineOrchestrator


class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""

    def test_orchestrator_initialization_with_real_paths(self, tmp_path, realistic_config):
        """Test orchestrator initialization with real filesystem paths."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        map_dir = tmp_path / "maps"

        # Create directories
        for directory in [input_dir, output_dir, map_dir]:
            directory.mkdir(exist_ok=True)

        # Should initialize successfully with real paths
        orchestrator = PipelineOrchestrator(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config_dict=realistic_config,
            task_id="integration-test",
        )

        # Verify paths were set correctly
        assert orchestrator.input_dir_path == input_dir
        assert orchestrator.output_dir_path == output_dir
        assert orchestrator.map_dir_path == map_dir
        assert orchestrator.task_id == "integration-test"
        assert orchestrator.config == realistic_config

    def test_file_discovery_with_real_files(self, tmp_path, multiple_text_files):
        """Test file discovery with actual files on filesystem."""
        # Move files to input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        actual_files = []
        for i, source_file in enumerate(multiple_text_files):
            target_file = input_dir / f"doc_{i}.txt"
            target_file.write_text(source_file.read_text())
            actual_files.append(target_file)

        orchestrator = PipelineOrchestrator(input_dir=input_dir)
        discovered_files = orchestrator._discover_files_to_process()

        # Should find all .txt files
        assert len(discovered_files) == len(actual_files)

        discovered_names = {f.name for f in discovered_files}
        expected_names = {f.name for f in actual_files}
        assert discovered_names == expected_names

    def test_file_discovery_with_specific_file(self, tmp_path, sample_text_file):
        """Test file discovery when targeting a specific file."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        # Copy sample file to input directory
        target_file = input_dir / "target_document.txt"
        target_file.write_text(sample_text_file.read_text())

        # Create other files that should be ignored
        (input_dir / "other1.txt").write_text("Other content 1")
        (input_dir / "other2.txt").write_text("Other content 2")

        orchestrator = PipelineOrchestrator(input_dir=input_dir)
        discovered_files = orchestrator._discover_files_to_process(specific_file="target_document.txt")

        # Should find only the specific file
        assert len(discovered_files) == 1
        assert discovered_files[0].name == "target_document.txt"

    async def test_text_reading_and_segmentation_integration(self, tmp_path, sample_text_file, expected_sentences):
        """Test the integration of text reading and sentence segmentation."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        # Copy file to input directory
        test_file = input_dir / "test_doc.txt"
        test_file.write_text(sample_text_file.read_text())

        orchestrator = PipelineOrchestrator(input_dir=input_dir)

        # Test file I/O setup
        data_source, _, _, _ = orchestrator._setup_file_io(test_file)

        # Test reading and segmentation
        num_sentences, sentences = await orchestrator._read_and_segment_sentences(data_source)

        # Validate against expected results from actual segmentation
        assert num_sentences == len(expected_sentences)
        assert sentences == expected_sentences
        assert all(isinstance(sentence, str) for sentence in sentences)
        assert all(len(sentence.strip()) > 0 for sentence in sentences)

    async def test_map_file_writing_integration(self, tmp_path, expected_sentences):
        """Test map file writing with real file operations."""
        input_dir = tmp_path / "input"
        map_dir = tmp_path / "maps"

        for directory in [input_dir, map_dir]:
            directory.mkdir()

        orchestrator = PipelineOrchestrator(
            input_dir=input_dir,
            map_dir=map_dir,
        )

        # Create a test file
        test_file = input_dir / "integration_test.txt"
        test_file.write_text("Test content for map writing.")

        # Setup I/O components
        _, map_storage, _, _ = orchestrator._setup_file_io(test_file)

        # Write map file
        await orchestrator._write_map_file(
            expected_sentences,
            map_storage,
            test_file.name,
        )

        # Verify map file was created
        expected_map_file = map_dir / "integration_test_map.jsonl"
        assert expected_map_file.exists()

        # Read and verify map file content
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
        """Test context building integration with real AnalysisService."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        orchestrator = PipelineOrchestrator(input_dir=input_dir)

        # Replace with real analysis service
        orchestrator.analysis_service = real_analysis_service_with_mocked_llm

        # Test context building
        contexts = orchestrator._build_contexts(expected_sentences, "test.txt")

        # Validate context structure and content
        assert isinstance(contexts, list)
        assert len(contexts) == len(expected_sentences)

        for i, context in enumerate(contexts):
            assert isinstance(context, dict)

            # Check for expected context window keys
            expected_keys = {
                "structure_analysis",
                "immediate_context",
                "observer_context",
                "broader_context",
                "overall_context",
            }
            assert all(key in context for key in expected_keys)

            # Context should be derived from actual sentences, not hardcoded
            for key, context_sentences in context.items():
                assert isinstance(context_sentences, list)
                # Context sentences should be from our actual sentence list
                for ctx_sentence in context_sentences:
                    if ctx_sentence:  # Skip empty context
                        assert ctx_sentence in expected_sentences

    async def test_analysis_with_real_service_integration(
        self, tmp_path, expected_sentences, real_analysis_service_with_mocked_llm
    ):
        """Test sentence analysis integration with realistic analysis service."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        for directory in [input_dir, output_dir]:
            directory.mkdir(exist_ok=True)

        orchestrator = PipelineOrchestrator(
            input_dir=input_dir,
            output_dir=output_dir,
        )

        # Use real analysis service
        orchestrator.analysis_service = real_analysis_service_with_mocked_llm

        # Test file setup
        test_file = input_dir / "analysis_test.txt"
        test_file.write_text(" ".join(expected_sentences))

        # Setup I/O
        _, _, analysis_writer, _ = orchestrator._setup_file_io(test_file)

        # Build contexts
        contexts = orchestrator._build_contexts(expected_sentences, test_file.name)

        # Run analysis
        await orchestrator._analyze_and_save_results(
            expected_sentences,
            contexts,
            analysis_writer,
            test_file.name,
        )

        # Verify analysis file was created
        expected_analysis_file = output_dir / "analysis_test_analysis.jsonl"
        assert expected_analysis_file.exists()

        # Read and validate analysis results
        import json

        with expected_analysis_file.open() as f:
            lines = f.readlines()

        assert len(lines) == len(expected_sentences)

        for i, line in enumerate(lines):
            result = json.loads(line.strip())

            # Verify structure
            assert "sentence" in result
            assert "function_type" in result
            assert "structure_type" in result
            assert "sentence_id" in result

            # Verify content is based on actual sentence
            assert result["sentence"] == expected_sentences[i]
            assert result["sentence_id"] == i

            # Verify analysis is realistic, not hardcoded
            assert result["function_type"] in ["declarative", "interrogative", "exclamatory", "request"]
            assert result["structure_type"] in ["simple", "compound", "complex"]

    def test_error_handling_with_nonexistent_input_directory(self, tmp_path):
        """Test error handling when input directory doesn't exist."""
        nonexistent_dir = tmp_path / "does_not_exist"

        # Should raise FileNotFoundError due to strict path resolution
        with pytest.raises(FileNotFoundError):
            PipelineOrchestrator(input_dir=nonexistent_dir)

    def test_empty_directory_handling(self, tmp_path):
        """Test behavior with empty input directory."""
        input_dir = tmp_path / "empty_input"
        input_dir.mkdir(exist_ok=True)

        orchestrator = PipelineOrchestrator(input_dir=input_dir)
        files = orchestrator._discover_files_to_process()

        # Should return empty list for empty directory
        assert files == []

    async def test_empty_file_handling(self, tmp_path, empty_text_file):
        """Test handling of empty text files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        # Copy empty file to input directory
        empty_file = input_dir / "empty.txt"
        empty_file.write_text("")

        orchestrator = PipelineOrchestrator(input_dir=input_dir)

        # Setup I/O
        data_source, _, _, _ = orchestrator._setup_file_io(empty_file)

        # Test reading empty file
        num_sentences, sentences = await orchestrator._read_and_segment_sentences(data_source)

        # Should handle empty content gracefully
        assert num_sentences == 0
        assert sentences == []
