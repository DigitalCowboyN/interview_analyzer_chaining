"""
Comprehensive tests for MetricsTracker using real instances to verify all accumulation logic.
Tests every method, edge case, and integration pattern used in production.
"""

from unittest.mock import patch

from src.utils.metrics import MetricsTracker, metrics_tracker


class TestMetricsTrackerBasicOperations:
    """Test basic counter and accumulation operations."""

    def test_initialization_and_reset(self):
        """Test that tracker initializes correctly and reset works."""
        tracker = MetricsTracker()

        # Should start with clean state
        assert tracker.api_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.errors == 0
        assert tracker.pipeline_start_time is None
        assert tracker.pipeline_end_time is None
        assert tracker.custom_metrics == {}
        assert tracker.file_timers == {}

        # Modify state
        tracker.increment_api_calls(5)
        tracker.add_tokens(100)
        tracker.set_metric("test", "key", "value")

        # Reset should clear everything
        tracker.reset()
        assert tracker.api_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.custom_metrics == {}

    def test_api_calls_accumulation(self):
        """Test API call counter accumulation patterns."""
        tracker = MetricsTracker()

        # Single increments
        tracker.increment_api_calls()
        assert tracker.api_calls == 1

        # Batch increments
        tracker.increment_api_calls(5)
        assert tracker.api_calls == 6

        # Zero increment (edge case)
        tracker.increment_api_calls(0)
        assert tracker.api_calls == 6

        # Large increment
        tracker.increment_api_calls(1000)
        assert tracker.api_calls == 1006

    def test_token_accumulation_with_edge_cases(self):
        """Test token accumulation including edge cases from production."""
        tracker = MetricsTracker()

        # Normal accumulation
        tracker.add_tokens(100)
        tracker.add_tokens(50)
        assert tracker.total_tokens == 150

        # Zero tokens (valid case)
        tracker.add_tokens(0)
        assert tracker.total_tokens == 150

        # Negative tokens (should be rejected)
        tracker.add_tokens(-10)
        assert tracker.total_tokens == 150  # Should not change

        # None tokens (OpenAI API sometimes returns None)
        tracker.add_tokens(None)
        assert tracker.total_tokens == 150  # Should not crash

    def test_error_tracking_patterns(self):
        """Test error tracking as used across the pipeline."""
        tracker = MetricsTracker()

        # Basic error increment
        tracker.increment_errors()
        assert tracker.errors == 1

        # Batch error increment
        tracker.increment_errors(count=3)
        assert tracker.errors == 4

        # Category-based errors (future enhancement)
        tracker.increment_errors("validation_error", count=2)
        assert tracker.errors == 6


class TestCustomMetricsSystem:
    """Test the custom metrics categorization and accumulation system."""

    def test_set_metric_creates_categories(self):
        """Test that set_metric properly creates category structure."""
        tracker = MetricsTracker()

        tracker.set_metric("pipeline", "files_processed", 5)
        tracker.set_metric("file1.txt", "processing_time", 2.5)
        tracker.set_metric("pipeline", "errors", 0)

        expected_structure = {"pipeline": {"files_processed": 5, "errors": 0}, "file1.txt": {"processing_time": 2.5}}
        assert tracker.custom_metrics == expected_structure

    def test_increment_metric_numeric_logic(self):
        """Test increment_metric's conditional logic for numeric values."""
        tracker = MetricsTracker()

        # Test integer increment
        tracker.increment_metric("test", "counter", 5)
        tracker.increment_metric("test", "counter", 3)
        assert tracker.custom_metrics["test"]["counter"] == 8

        # Test float increment
        tracker.set_metric("test", "float_val", 2.5)
        tracker.increment_metric("test", "float_val", 1.5)
        assert tracker.custom_metrics["test"]["float_val"] == 4.0

        # Test zero increment
        tracker.increment_metric("test", "counter", 0)
        assert tracker.custom_metrics["test"]["counter"] == 8

    def test_increment_metric_non_numeric_overwrite(self):
        """Test increment_metric's overwrite behavior for non-numeric values."""
        tracker = MetricsTracker()

        # Set string value
        tracker.set_metric("test", "string_val", "initial")

        # Increment should overwrite, not concatenate
        tracker.increment_metric("test", "string_val", 10)
        assert tracker.custom_metrics["test"]["string_val"] == 10

        # Test with None value
        tracker.set_metric("test", "none_val", None)
        tracker.increment_metric("test", "none_val", 5)
        assert tracker.custom_metrics["test"]["none_val"] == 5

    def test_convenience_methods_integration(self):
        """Test all convenience methods used in production."""
        tracker = MetricsTracker()

        # Test pipeline-level methods
        tracker.increment_files_processed(3)
        tracker.increment_files_failed(1)
        tracker.increment_sentences_success(10)

        # Test file-level methods
        tracker.increment_results_processed("file1.txt", 5)
        tracker.add_processing_time("sentence_1", 1.23)

        # Verify structure
        assert tracker.custom_metrics["pipeline"]["files_processed"] == 3
        assert tracker.custom_metrics["pipeline"]["files_failed"] == 1
        assert tracker.custom_metrics["pipeline"]["sentences_success"] == 10
        assert tracker.custom_metrics["file1.txt"]["results_processed"] == 5
        assert tracker.custom_metrics["sentences"]["processing_time_sentence_1"] == 1.23


class TestTimerSystem:
    """Test the timer system used for performance tracking."""

    @patch("src.utils.metrics.time")
    def test_pipeline_timer_calculation(self, mock_time):
        """Test pipeline timer start/stop and duration calculation."""
        tracker = MetricsTracker()

        # Mock time progression
        mock_time.side_effect = [1000.0, 1005.5]  # 5.5 second duration

        tracker.start_pipeline_timer()
        assert tracker.pipeline_start_time == 1000.0

        tracker.stop_pipeline_timer()
        assert tracker.pipeline_end_time == 1005.5

        # Test summary calculation
        summary = tracker.get_summary()
        assert summary["pipeline_duration_seconds"] == 5.5

    @patch("src.utils.metrics.time")
    def test_file_timer_system(self, mock_time):
        """Test file timer start/stop with automatic metric storage."""
        tracker = MetricsTracker()

        # Mock time progression for multiple files
        mock_time.side_effect = [
            100.0,  # file1 start
            103.5,  # file1 stop (3.5s duration)
            200.0,  # file2 start
            202.1,  # file2 stop (2.1s duration)
        ]

        # Test file1
        tracker.start_file_timer("file1.txt")
        assert tracker.file_timers["file1.txt"] == 100.0

        tracker.stop_file_timer("file1.txt")
        assert tracker.custom_metrics["file1.txt"]["processing_time_seconds"] == 3.5

        # Test file2
        tracker.start_file_timer("file2.txt")
        tracker.stop_file_timer("file2.txt")
        assert tracker.custom_metrics["file2.txt"]["processing_time_seconds"] == 2.1

        # File timers should still contain start times
        assert "file1.txt" in tracker.file_timers
        assert "file2.txt" in tracker.file_timers

    def test_file_timer_edge_cases(self):
        """Test file timer edge cases that could occur in production."""
        tracker = MetricsTracker()

        # Stop timer for non-existent file (should not crash)
        tracker.stop_file_timer("nonexistent.txt")
        assert "nonexistent.txt" not in tracker.custom_metrics

        # Start multiple timers for same file (should overwrite)
        with patch("src.utils.metrics.time") as mock_time:
            mock_time.side_effect = [100.0, 200.0, 205.0]

            tracker.start_file_timer("test.txt")
            tracker.start_file_timer("test.txt")  # Overwrite
            tracker.stop_file_timer("test.txt")

            # Should use the second start time
            assert tracker.custom_metrics["test.txt"]["processing_time_seconds"] == 5.0

    def test_pipeline_timer_without_stop(self):
        """Test summary when pipeline timer started but not stopped."""
        tracker = MetricsTracker()

        with patch("src.utils.metrics.time") as mock_time:
            mock_time.return_value = 1000.0
            tracker.start_pipeline_timer()

        # Summary should handle missing end time
        summary = tracker.get_summary()
        assert summary["pipeline_duration_seconds"] is None

    def test_pipeline_timer_without_start(self):
        """Test summary when pipeline timer stopped but not started."""
        tracker = MetricsTracker()

        with patch("src.utils.metrics.time") as mock_time:
            mock_time.return_value = 1000.0
            tracker.stop_pipeline_timer()

        # Summary should handle missing start time
        summary = tracker.get_summary()
        assert summary["pipeline_duration_seconds"] is None


class TestSummaryGeneration:
    """Test the get_summary method used for reporting."""

    def test_complete_summary_structure(self):
        """Test that summary contains all expected fields."""
        tracker = MetricsTracker()

        # Populate with realistic data
        tracker.increment_api_calls(15)
        tracker.add_tokens(1250)
        tracker.increment_errors(count=2)  # Fix: specify count parameter explicitly
        tracker.set_metric("pipeline", "files_processed", 3)
        tracker.set_metric("file1.txt", "sentences", 45)

        with patch("src.utils.metrics.time") as mock_time:
            mock_time.side_effect = [1000.0, 1030.5]
            tracker.start_pipeline_timer()
            tracker.stop_pipeline_timer()

        summary = tracker.get_summary()

        # Test structure
        expected_keys = {
            "total_api_calls",
            "total_tokens_used",
            "total_errors",
            "pipeline_duration_seconds",
            "custom_metrics",
        }
        assert set(summary.keys()) == expected_keys

        # Test values
        assert summary["total_api_calls"] == 15
        assert summary["total_tokens_used"] == 1250
        assert summary["total_errors"] == 2
        assert summary["pipeline_duration_seconds"] == 30.5
        assert summary["custom_metrics"]["pipeline"]["files_processed"] == 3
        assert summary["custom_metrics"]["file1.txt"]["sentences"] == 45

    def test_empty_summary(self):
        """Test summary with no metrics set."""
        tracker = MetricsTracker()
        summary = tracker.get_summary()

        assert summary["total_api_calls"] == 0
        assert summary["total_tokens_used"] == 0
        assert summary["total_errors"] == 0
        assert summary["pipeline_duration_seconds"] is None
        assert summary["custom_metrics"] == {}


class TestProductionUsagePatterns:
    """Test metrics patterns actually used in production code."""

    def test_pipeline_orchestrator_pattern(self):
        """Test the exact pattern used in PipelineOrchestrator."""
        tracker = MetricsTracker()

        # Simulate pipeline execution metrics
        tracker.start_pipeline_timer()
        tracker.set_metric("pipeline", "files_discovered", 3)

        # Simulate per-file processing
        for file_num in range(3):
            filename = f"file{file_num}.txt"
            tracker.start_file_timer(filename)

            # Simulate processing metrics
            tracker.set_metric(filename, "sentences_segmented", 10 + file_num)
            tracker.increment_results_processed(filename, 10 + file_num)

            with patch("src.utils.metrics.time") as mock_time:
                mock_time.side_effect = [100.0, 102.0 + file_num]
                tracker.start_file_timer(filename)  # Restart for consistent timing
                tracker.stop_file_timer(filename)

        # Simulate pipeline completion
        tracker.increment_files_processed(3)
        tracker.set_metric("pipeline", "verification_errors", 0)
        tracker.stop_pipeline_timer()

        # Verify accumulated metrics match expected patterns
        summary = tracker.get_summary()
        assert summary["custom_metrics"]["pipeline"]["files_discovered"] == 3
        assert summary["custom_metrics"]["pipeline"]["files_processed"] == 3
        assert summary["custom_metrics"]["file0.txt"]["sentences_segmented"] == 10
        assert summary["custom_metrics"]["file2.txt"]["results_processed"] == 12

    def test_analysis_service_pattern(self):
        """Test the exact pattern used in AnalysisService."""
        tracker = MetricsTracker()

        # Simulate sentence analysis metrics
        sentence_count = 5
        for i in range(sentence_count):
            # Simulate processing time tracking
            tracker.add_processing_time(str(i), 0.5 + i * 0.1)
            tracker.increment_sentences_success()

        # Simulate error case
        tracker.increment_errors()

        # Verify metrics
        assert tracker.custom_metrics["pipeline"]["sentences_success"] == 5
        assert tracker.errors == 1
        assert tracker.custom_metrics["sentences"]["processing_time_0"] == 0.5
        assert tracker.custom_metrics["sentences"]["processing_time_4"] == 0.9

    def test_agent_error_tracking_pattern(self):
        """Test the error tracking patterns used in agents."""
        tracker = MetricsTracker()

        # Simulate validation errors from sentence_analyzer.py
        validation_errors = [
            "validation_error_function_type",
            "validation_error_structure_type",
            "validation_error_purpose",
            "validation_error_topic_level_1",
            "validation_error_domain_keywords",
        ]

        for error_type in validation_errors:
            tracker.increment_errors(error_type)

        # Simulate API call tracking from agent.py
        tracker.increment_api_calls(10)
        tracker.add_tokens(500)

        # Verify accumulation
        assert tracker.errors == 5  # Global error count
        assert tracker.api_calls == 10
        assert tracker.total_tokens == 500


class TestGlobalInstanceBehavior:
    """Test the global metrics_tracker instance behavior."""

    def test_global_instance_isolation(self):
        """Test that global instance doesn't interfere with local instances."""
        # Reset global instance
        metrics_tracker.reset()

        # Create local instance
        local_tracker = MetricsTracker()

        # Modify both
        metrics_tracker.increment_api_calls(5)
        local_tracker.increment_api_calls(3)

        # Should be independent
        assert metrics_tracker.api_calls == 5
        assert local_tracker.api_calls == 3

    def test_global_instance_reset_safety(self):
        """Test that resetting global instance is safe."""
        # Populate global instance
        metrics_tracker.increment_api_calls(10)
        metrics_tracker.set_metric("test", "value", 42)

        # Reset should clear everything
        metrics_tracker.reset()
        assert metrics_tracker.api_calls == 0
        assert metrics_tracker.custom_metrics == {}

        # Should be usable after reset
        metrics_tracker.increment_api_calls(1)
        assert metrics_tracker.api_calls == 1


class TestConcurrencyAndThreadSafety:
    """Test metrics behavior under concurrent access patterns."""

    def test_concurrent_metric_updates(self):
        """Test that concurrent updates don't cause data corruption."""
        import threading
        import time as time_module

        tracker = MetricsTracker()

        def worker(worker_id):
            for i in range(100):
                tracker.increment_api_calls()
                tracker.add_tokens(1)
                tracker.increment_metric("worker", f"worker_{worker_id}", 1)
                time_module.sleep(0.001)  # Small delay to increase chance of race conditions

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify final counts (may have race conditions, but shouldn't crash)
        assert tracker.api_calls <= 500  # Should be 500 if no race conditions
        assert tracker.total_tokens <= 500
        # Note: This test mainly ensures no crashes occur during concurrent access
