"""
Integration tests using real MetricsTracker with actual pipeline components.
Tests end-to-end metrics accumulation in realistic scenarios.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.context_builder import ContextBuilder
from src.agents.sentence_analyzer import SentenceAnalyzer
from src.pipeline import PipelineOrchestrator
from src.services.analysis_service import AnalysisService
from src.utils.metrics import MetricsTracker


class TestMetricsIntegrationWithRealComponents:
    """Integration tests with real pipeline components."""

    @pytest.fixture
    def real_metrics_tracker(self):
        """Provide a real MetricsTracker instance for integration testing."""
        tracker = MetricsTracker()
        tracker.reset()
        return tracker

    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
        return {
            "paths": {
                "output_dir": "test_output",
                "map_dir": "test_maps",
                "map_suffix": "_map.jsonl",
                "analysis_suffix": "_analysis.jsonl",
            },
            "pipeline": {"num_concurrent_files": 2},
            "preprocessing": {"context_windows": {"immediate": 1, "broader": 3, "observer": 5}},
            "classification": {"local": {"prompt_files": {"no_context": "dummy"}}},
            "domain_keywords": ["test", "integration"],
            "openai": {"model_name": "gpt-4"},
        }

    @pytest.mark.asyncio
    async def test_analysis_service_metrics_integration(self, real_metrics_tracker, integration_config):
        """Test that AnalysisService correctly updates real MetricsTracker."""
        # Create real components
        context_builder = ContextBuilder(integration_config)

        # Mock the sentence analyzer to avoid API calls
        sentence_analyzer = AsyncMock(spec=SentenceAnalyzer)
        sentence_analyzer.classify_sentence.return_value = {
            "function_type": "declarative",
            "structure_type": "simple",
            "purpose": "testing",
        }

        # Create service with real metrics tracker
        service = AnalysisService(
            config=integration_config,
            context_builder=context_builder,
            sentence_analyzer=sentence_analyzer,
            metrics_tracker=real_metrics_tracker,
        )

        # Test data
        sentences = ["This is test sentence one.", "This is test sentence two."]
        contexts = [{"immediate": "context1"}, {"immediate": "context2"}]

        # Mock timer for consistent timing
        with patch("time.time") as mock_time:
            mock_time.side_effect = [1.0, 1.5, 2.0, 2.3]  # 0.5s and 0.3s processing times

            # Execute analysis
            results = await service.analyze_sentences(sentences, contexts)

        # Verify real metrics accumulation
        assert len(results) == 2
        assert real_metrics_tracker.custom_metrics["pipeline"]["sentences_success"] == 2

        # Verify processing times were recorded
        assert "sentences" in real_metrics_tracker.custom_metrics
        processing_times = real_metrics_tracker.custom_metrics["sentences"]
        assert "processing_time_0" in processing_times
        assert "processing_time_1" in processing_times
        assert processing_times["processing_time_0"] == 0.5
        assert processing_times["processing_time_1"] == 0.3

    @pytest.mark.asyncio
    async def test_pipeline_orchestrator_metrics_integration(self, tmp_path, real_metrics_tracker, integration_config):
        """Test that PipelineOrchestrator correctly accumulates metrics."""
        # Setup test directories
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        map_dir = tmp_path / "maps"

        for dir_path in [input_dir, output_dir, map_dir]:
            dir_path.mkdir()

        # Create test file
        test_file = input_dir / "test.txt"
        test_file.write_text("Hello world. This is a test sentence.")

        # Update config with test paths
        test_config = integration_config.copy()
        test_config["paths"]["output_dir"] = str(output_dir)
        test_config["paths"]["map_dir"] = str(map_dir)

        # Create orchestrator with real metrics tracker
        orchestrator = PipelineOrchestrator(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config_dict=test_config,
            task_id="metrics-integration-test",
        )

        # Replace the metrics tracker with our real one
        orchestrator.metrics_tracker = real_metrics_tracker

        # Mock the analysis service to avoid API calls
        mock_analysis_service = AsyncMock()
        mock_analysis_service.analyze_sentences.return_value = [
            {"sentence_id": 0, "sequence_order": 0, "sentence": "Hello world.", "function_type": "declarative"},
            {
                "sentence_id": 1,
                "sequence_order": 1,
                "sentence": "This is a test sentence.",
                "function_type": "declarative",
            },
        ]
        orchestrator.analysis_service = mock_analysis_service

        # Execute pipeline
        with patch("time.time") as mock_time:
            mock_time.side_effect = [
                1000.0,  # Pipeline start
                1001.0,  # File timer start
                1003.5,  # File timer stop
                1010.0,  # Pipeline stop
            ]

            await orchestrator.execute()

        # Verify metrics accumulation
        summary = real_metrics_tracker.get_summary()

        # Pipeline-level metrics
        assert summary["pipeline_duration_seconds"] == 10.0
        assert summary["custom_metrics"]["pipeline"]["files_processed"] == 1

        # File-level metrics
        assert "test.txt" in summary["custom_metrics"]
        file_metrics = summary["custom_metrics"]["test.txt"]
        assert file_metrics["processing_time_seconds"] == 2.5
        assert file_metrics["sentences_segmented"] == 2
        assert file_metrics["results_processed"] == 2

    def test_error_accumulation_across_components(self, real_metrics_tracker):
        """Test that errors from different components accumulate correctly."""

        # Simulate errors from different parts of the system
        # Agent validation errors
        real_metrics_tracker.increment_errors("validation_error_function_type")
        real_metrics_tracker.increment_errors("validation_error_structure_type")

        # Pipeline errors
        real_metrics_tracker.increment_errors("file_read_error")
        real_metrics_tracker.increment_errors("map_write_error", 2)

        # Service errors
        real_metrics_tracker.increment_errors()  # Generic error

        # Verify total error count
        assert real_metrics_tracker.errors == 6

        # Verify summary includes errors
        summary = real_metrics_tracker.get_summary()
        assert summary["total_errors"] == 6


class TestMetricsWithRealPipelineScenarios:
    """Test metrics with realistic pipeline failure and success scenarios."""

    @pytest.fixture
    def scenario_metrics_tracker(self):
        tracker = MetricsTracker()
        tracker.reset()
        return tracker

    def test_mixed_success_failure_scenario(self, scenario_metrics_tracker):
        """Test metrics accumulation in mixed success/failure scenarios."""
        tracker = scenario_metrics_tracker

        # Simulate processing 3 files: 2 success, 1 failure
        tracker.set_metric("pipeline", "files_discovered", 3)

        # File 1: Success
        tracker.start_file_timer("file1.txt")
        tracker.set_metric("file1.txt", "sentences_segmented", 5)
        tracker.increment_results_processed("file1.txt", 5)
        tracker.increment_sentences_success(5)

        with patch("src.utils.metrics.time") as mock_time:
            mock_time.side_effect = [100.0, 103.0]
            tracker.start_file_timer("file1.txt")
            tracker.stop_file_timer("file1.txt")

        # File 2: Success
        tracker.set_metric("file2.txt", "sentences_segmented", 3)
        tracker.increment_results_processed("file2.txt", 3)
        tracker.increment_sentences_success(3)

        # File 3: Failure
        tracker.increment_errors("file_processing_error")
        tracker.increment_files_failed(1)

        # Pipeline completion
        tracker.increment_files_processed(2)  # Only successful files
        tracker.set_metric("pipeline", "verification_errors", 1)

        # Verify final state
        summary = tracker.get_summary()
        assert summary["custom_metrics"]["pipeline"]["files_discovered"] == 3
        assert summary["custom_metrics"]["pipeline"]["files_processed"] == 2
        assert summary["custom_metrics"]["pipeline"]["files_failed"] == 1
        assert summary["custom_metrics"]["pipeline"]["sentences_success"] == 8
        assert summary["total_errors"] == 1
        assert summary["custom_metrics"]["file1.txt"]["processing_time_seconds"] == 3.0

    def test_verification_failure_scenario(self, scenario_metrics_tracker):
        """Test metrics during output verification failures."""
        tracker = scenario_metrics_tracker

        # Simulate verification process
        tracker.set_metric("pipeline", "verification_total_missing", 5)
        tracker.set_metric("pipeline", "verification_errors", 2)

        # Simulate per-file verification results
        tracker.set_metric("file1.txt", "expected_sentences", 10)
        tracker.set_metric("file1.txt", "missing_sentences", 2)
        tracker.set_metric("file2.txt", "expected_sentences", 8)
        tracker.set_metric("file2.txt", "missing_sentences", 3)

        summary = tracker.get_summary()
        assert summary["custom_metrics"]["pipeline"]["verification_total_missing"] == 5
        assert summary["custom_metrics"]["pipeline"]["verification_errors"] == 2
        assert summary["custom_metrics"]["file1.txt"]["missing_sentences"] == 2

    def test_concurrent_processing_metrics(self, scenario_metrics_tracker):
        """Test metrics accumulation during concurrent file processing."""
        tracker = scenario_metrics_tracker

        # Simulate concurrent processing of multiple files
        files = ["file1.txt", "file2.txt", "file3.txt"]

        for i, filename in enumerate(files):
            # Each file has different processing characteristics
            sentences = 5 + i * 2
            processing_time = 1.0 + i * 0.5

            tracker.set_metric(filename, "sentences_segmented", sentences)
            tracker.increment_results_processed(filename, sentences)
            tracker.increment_sentences_success(sentences)

            # Mock file timing
            with patch("src.utils.metrics.time") as mock_time:
                start_time = 100.0 + i * 10
                mock_time.side_effect = [start_time, start_time + processing_time]
                tracker.start_file_timer(filename)
                tracker.stop_file_timer(filename)

        # Verify accumulated metrics
        summary = tracker.get_summary()

        # Total sentences across all files: 5 + 7 + 9 = 21
        assert summary["custom_metrics"]["pipeline"]["sentences_success"] == 21

        # Verify per-file metrics
        assert summary["custom_metrics"]["file1.txt"]["sentences_segmented"] == 5
        assert summary["custom_metrics"]["file2.txt"]["sentences_segmented"] == 7
        assert summary["custom_metrics"]["file3.txt"]["sentences_segmented"] == 9

        # Verify processing times
        assert summary["custom_metrics"]["file1.txt"]["processing_time_seconds"] == 1.0
        assert summary["custom_metrics"]["file2.txt"]["processing_time_seconds"] == 1.5
        assert summary["custom_metrics"]["file3.txt"]["processing_time_seconds"] == 2.0
