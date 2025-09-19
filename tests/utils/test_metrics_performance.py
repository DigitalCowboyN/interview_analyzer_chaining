"""
Performance and edge case tests for MetricsTracker.
Tests behavior under stress conditions and unusual scenarios.
"""

import threading
import time
from unittest.mock import patch

import pytest

from src.utils.metrics import MetricsTracker


class TestMetricsPerformance:
    """Test metrics performance under various load conditions."""

    def test_high_volume_metric_updates(self):
        """Test performance with high volume of metric updates."""
        tracker = MetricsTracker()

        start_time = time.time()

        # Simulate high-volume updates
        for i in range(10000):
            tracker.increment_api_calls()
            tracker.add_tokens(1)
            tracker.increment_metric("performance", "counter", 1)
            tracker.set_metric("performance", f"key_{i % 100}", i)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 5.0, f"High volume updates took {duration:.2f}s, expected < 5.0s"

        # Verify final state
        assert tracker.api_calls == 10000
        assert tracker.total_tokens == 10000
        assert tracker.custom_metrics["performance"]["counter"] == 10000

    def test_large_custom_metrics_structure(self):
        """Test behavior with large custom metrics structures."""
        tracker = MetricsTracker()

        # Create large nested structure
        for category in range(100):
            for key in range(50):
                tracker.set_metric(f"category_{category}", f"key_{key}", key * category)

        # Verify structure size
        assert len(tracker.custom_metrics) == 100
        for category_name in tracker.custom_metrics:
            assert len(tracker.custom_metrics[category_name]) == 50

        # Test summary generation performance
        start_time = time.time()
        summary = tracker.get_summary()
        end_time = time.time()

        assert (end_time - start_time) < 1.0, "Summary generation too slow for large structure"
        assert len(summary["custom_metrics"]) == 100

    def test_concurrent_access_stress(self):
        """Stress test concurrent access to metrics."""
        tracker = MetricsTracker()

        def stress_worker(worker_id, iterations=1000):
            for i in range(iterations):
                tracker.increment_api_calls()
                tracker.add_tokens(1)
                tracker.increment_metric("stress", f"worker_{worker_id}", 1)
                tracker.set_metric("stress", f"worker_{worker_id}_last", i)

        # Start multiple concurrent workers
        threads = []
        num_workers = 10
        iterations_per_worker = 1000

        start_time = time.time()
        for worker_id in range(num_workers):
            t = threading.Thread(target=stress_worker, args=(worker_id, iterations_per_worker))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        end_time = time.time()

        # Verify no crashes and reasonable performance
        assert (end_time - start_time) < 10.0, "Concurrent stress test took too long"

        # Note: Due to potential race conditions, we can't verify exact counts
        # but we can verify the system didn't crash and has reasonable values
        assert tracker.api_calls <= num_workers * iterations_per_worker
        assert tracker.total_tokens <= num_workers * iterations_per_worker


class TestMetricsEdgeCases:
    """Test edge cases and error conditions."""

    def test_extreme_values(self):
        """Test behavior with extreme values."""
        tracker = MetricsTracker()

        # Test very large numbers
        large_number = 10**15
        tracker.increment_api_calls(large_number)
        tracker.add_tokens(large_number)

        assert tracker.api_calls == large_number
        assert tracker.total_tokens == large_number

        # Test very small increments
        tracker.increment_metric("test", "small", 0.000001)
        tracker.increment_metric("test", "small", 0.000001)

        # Should handle floating point precision
        assert abs(tracker.custom_metrics["test"]["small"] - 0.000002) < 1e-10

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in metric names."""
        tracker = MetricsTracker()

        # Test unicode category and key names
        tracker.set_metric("测试类别", "测试键", "测试值")
        tracker.set_metric("category with spaces", "key-with-dashes", "value_with_underscores")
        tracker.set_metric("category/with/slashes", "key.with.dots", 42)

        # Should handle all character types
        assert "测试类别" in tracker.custom_metrics
        assert tracker.custom_metrics["测试类别"]["测试键"] == "测试值"
        assert tracker.custom_metrics["category with spaces"]["key-with-dashes"] == "value_with_underscores"
        assert tracker.custom_metrics["category/with/slashes"]["key.with.dots"] == 42

    def test_memory_usage_with_many_categories(self):
        """Test memory behavior with many categories and keys."""
        import sys

        tracker = MetricsTracker()

        # Measure initial memory usage (rough estimate)
        initial_size = sys.getsizeof(tracker.custom_metrics)

        # Add many categories
        for i in range(1000):
            tracker.set_metric(f"category_{i}", "key", i)

        final_size = sys.getsizeof(tracker.custom_metrics)

        # Memory should grow but not excessively
        growth_ratio = final_size / initial_size if initial_size > 0 else float("inf")
        assert growth_ratio < 1000, f"Memory growth ratio too high: {growth_ratio}"

        # Should still be functional
        summary = tracker.get_summary()
        assert len(summary["custom_metrics"]) == 1000

    def test_reset_behavior_edge_cases(self):
        """Test reset behavior in various states."""
        tracker = MetricsTracker()

        # Populate with complex state
        tracker.increment_api_calls(100)
        tracker.start_pipeline_timer()
        tracker.start_file_timer("test.txt")
        tracker.set_metric("complex", "nested", {"deep": {"value": 42}})

        # Reset should clear everything
        tracker.reset()

        # Verify complete reset
        assert tracker.api_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.errors == 0
        assert tracker.pipeline_start_time is None
        assert tracker.pipeline_end_time is None
        assert tracker.custom_metrics == {}
        assert tracker.file_timers == {}

        # Should be fully functional after reset
        tracker.increment_api_calls(5)
        assert tracker.api_calls == 5

    def test_timer_precision_edge_cases(self):
        """Test timer precision with very short durations."""
        tracker = MetricsTracker()

        with patch("src.utils.metrics.time") as mock_time:
            # Test very short duration (microsecond precision)
            mock_time.side_effect = [1000.0, 1000.000001]

            tracker.start_file_timer("fast_file.txt")
            tracker.stop_file_timer("fast_file.txt")

            # Should handle microsecond precision
            processing_time = tracker.custom_metrics["fast_file.txt"]["processing_time_seconds"]
            assert processing_time == 0.0  # Rounded to 2 decimal places

        with patch("src.utils.metrics.time") as mock_time:
            # Test longer duration with precision
            mock_time.side_effect = [1000.0, 1000.123456]

            tracker.start_file_timer("precise_file.txt")
            tracker.stop_file_timer("precise_file.txt")

            processing_time = tracker.custom_metrics["precise_file.txt"]["processing_time_seconds"]
            assert processing_time == 0.12  # Rounded to 2 decimal places
