"""
Unit tests for the projection metrics module.

Tests the ProjectionMetrics class and MetricsTimer context manager.
"""

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.projections.metrics import (
    ProjectionMetrics,
    MetricsTimer,
    get_metrics,
    METRIC_EVENTS_PROCESSED,
    METRIC_EVENTS_FAILED,
    METRIC_PROCESSING_LATENCY,
)


class TestProjectionMetricsCounters:
    """Tests for counter functionality."""

    def test_counter_starts_at_zero(self):
        """Uninitialized counter should return 0."""
        metrics = ProjectionMetrics()

        assert metrics.get_counter("nonexistent_counter") == 0

    def test_increment_counter_increases_value_by_one(self):
        """increment_counter with default should increase by 1."""
        metrics = ProjectionMetrics()

        metrics.increment_counter("test_counter")

        assert metrics.get_counter("test_counter") == 1

    def test_increment_counter_increases_value_by_specified_amount(self):
        """increment_counter should increase by the specified value."""
        metrics = ProjectionMetrics()

        metrics.increment_counter("test_counter", 5)

        assert metrics.get_counter("test_counter") == 5

    def test_increment_counter_accumulates_values(self):
        """Multiple increments should accumulate."""
        metrics = ProjectionMetrics()

        metrics.increment_counter("test_counter", 3)
        metrics.increment_counter("test_counter", 7)

        assert metrics.get_counter("test_counter") == 10

    def test_multiple_counters_are_independent(self):
        """Different counters should track independently."""
        metrics = ProjectionMetrics()

        metrics.increment_counter("counter_a", 5)
        metrics.increment_counter("counter_b", 10)

        assert metrics.get_counter("counter_a") == 5
        assert metrics.get_counter("counter_b") == 10


class TestProjectionMetricsGauges:
    """Tests for gauge functionality."""

    def test_gauge_starts_at_zero(self):
        """Uninitialized gauge should return 0.0."""
        metrics = ProjectionMetrics()

        assert metrics.get_gauge("nonexistent_gauge") == 0.0

    def test_set_gauge_stores_value(self):
        """set_gauge should store the provided value."""
        metrics = ProjectionMetrics()

        metrics.set_gauge("test_gauge", 42.5)

        assert metrics.get_gauge("test_gauge") == 42.5

    def test_set_gauge_overwrites_previous_value(self):
        """set_gauge should replace the previous value."""
        metrics = ProjectionMetrics()

        metrics.set_gauge("test_gauge", 10.0)
        metrics.set_gauge("test_gauge", 20.0)

        assert metrics.get_gauge("test_gauge") == 20.0

    def test_multiple_gauges_are_independent(self):
        """Different gauges should track independently."""
        metrics = ProjectionMetrics()

        metrics.set_gauge("gauge_a", 1.5)
        metrics.set_gauge("gauge_b", 2.5)

        assert metrics.get_gauge("gauge_a") == 1.5
        assert metrics.get_gauge("gauge_b") == 2.5


class TestProjectionMetricsHistograms:
    """Tests for histogram functionality."""

    def test_histogram_stats_returns_zeros_for_empty_histogram(self):
        """Empty histogram should return zeros for all stats."""
        metrics = ProjectionMetrics()

        stats = metrics.get_histogram_stats("nonexistent_histogram")

        assert stats["count"] == 0
        assert stats["min"] == 0
        assert stats["max"] == 0
        assert stats["avg"] == 0
        assert stats["p50"] == 0
        assert stats["p95"] == 0
        assert stats["p99"] == 0

    def test_record_histogram_stores_values(self):
        """record_histogram should store recorded values."""
        metrics = ProjectionMetrics()

        metrics.record_histogram("test_histogram", 10.0)
        metrics.record_histogram("test_histogram", 20.0)
        metrics.record_histogram("test_histogram", 30.0)

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["count"] == 3

    def test_histogram_calculates_min_correctly(self):
        """Histogram min should be the smallest recorded value."""
        metrics = ProjectionMetrics()

        for value in [50, 10, 30, 5, 100]:
            metrics.record_histogram("test_histogram", value)

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["min"] == 5

    def test_histogram_calculates_max_correctly(self):
        """Histogram max should be the largest recorded value."""
        metrics = ProjectionMetrics()

        for value in [50, 10, 30, 5, 100]:
            metrics.record_histogram("test_histogram", value)

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["max"] == 100

    def test_histogram_calculates_average_correctly(self):
        """Histogram avg should be the mean of all recorded values."""
        metrics = ProjectionMetrics()

        values = [10, 20, 30, 40, 50]
        for value in values:
            metrics.record_histogram("test_histogram", value)

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["avg"] == 30.0  # (10+20+30+40+50)/5 = 30

    def test_histogram_calculates_p50_correctly(self):
        """Histogram p50 should be the median value."""
        metrics = ProjectionMetrics()

        # Record 100 sequential values
        for i in range(100):
            metrics.record_histogram("test_histogram", i)

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["p50"] == 50  # Middle value of 0-99

    def test_histogram_enforces_memory_limit(self):
        """Histogram should keep only last 1000 values."""
        metrics = ProjectionMetrics()

        # Record 1500 values
        for i in range(1500):
            metrics.record_histogram("test_histogram", i)

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["count"] == 1000
        # Min should be 500 (first 500 values dropped)
        assert stats["min"] == 500

    def test_histogram_p95_uses_fallback_for_small_samples(self):
        """For small samples (<20), p95 should return max value."""
        metrics = ProjectionMetrics()

        for i in range(10):
            metrics.record_histogram("test_histogram", i)

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["p95"] == 9  # Max value

    def test_histogram_p99_uses_fallback_for_small_samples(self):
        """For small samples (<100), p99 should return max value."""
        metrics = ProjectionMetrics()

        for i in range(50):
            metrics.record_histogram("test_histogram", i)

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["p99"] == 49  # Max value


class TestProjectionMetricsGetAllMetrics:
    """Tests for get_all_metrics method."""

    def test_get_all_metrics_includes_uptime(self):
        """get_all_metrics should include uptime_seconds."""
        metrics = ProjectionMetrics()

        all_metrics = metrics.get_all_metrics()

        assert "uptime_seconds" in all_metrics
        assert all_metrics["uptime_seconds"] >= 0

    def test_get_all_metrics_includes_counters(self):
        """get_all_metrics should include all counters."""
        metrics = ProjectionMetrics()
        metrics.increment_counter("test_counter", 5)

        all_metrics = metrics.get_all_metrics()

        assert "counters" in all_metrics
        assert all_metrics["counters"]["test_counter"] == 5

    def test_get_all_metrics_includes_gauges(self):
        """get_all_metrics should include all gauges."""
        metrics = ProjectionMetrics()
        metrics.set_gauge("test_gauge", 42.5)

        all_metrics = metrics.get_all_metrics()

        assert "gauges" in all_metrics
        assert all_metrics["gauges"]["test_gauge"] == 42.5

    def test_get_all_metrics_includes_histogram_stats(self):
        """get_all_metrics should include histogram statistics."""
        metrics = ProjectionMetrics()
        metrics.record_histogram("test_histogram", 10.0)

        all_metrics = metrics.get_all_metrics()

        assert "histograms" in all_metrics
        assert "test_histogram" in all_metrics["histograms"]
        assert all_metrics["histograms"]["test_histogram"]["count"] == 1


class TestProjectionMetricsReset:
    """Tests for reset method."""

    def test_reset_clears_counters(self):
        """reset should clear all counters."""
        metrics = ProjectionMetrics()
        metrics.increment_counter("test_counter", 10)

        metrics.reset()

        assert metrics.get_counter("test_counter") == 0

    def test_reset_clears_gauges(self):
        """reset should clear all gauges."""
        metrics = ProjectionMetrics()
        metrics.set_gauge("test_gauge", 42.5)

        metrics.reset()

        assert metrics.get_gauge("test_gauge") == 0.0

    def test_reset_clears_histograms(self):
        """reset should clear all histograms."""
        metrics = ProjectionMetrics()
        metrics.record_histogram("test_histogram", 10.0)

        metrics.reset()

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["count"] == 0

    def test_reset_resets_started_at(self):
        """reset should reset the started_at timestamp."""
        metrics = ProjectionMetrics()
        original_started_at = metrics.started_at

        time.sleep(0.01)  # Small delay to ensure time difference
        metrics.reset()

        assert metrics.started_at > original_started_at


class TestMetricsTimer:
    """Tests for MetricsTimer context manager."""

    def test_metrics_timer_records_elapsed_time(self):
        """MetricsTimer should record elapsed time in milliseconds."""
        metrics = ProjectionMetrics()

        with MetricsTimer("test_timing", metrics):
            time.sleep(0.05)  # 50ms

        stats = metrics.get_histogram_stats("test_timing")
        assert stats["count"] == 1
        assert stats["min"] >= 40  # At least 40ms (allowing for timing variance)
        assert stats["max"] < 200  # Less than 200ms

    def test_metrics_timer_uses_global_metrics_by_default(self):
        """MetricsTimer should use global metrics if none provided."""
        global_metrics = get_metrics()
        initial_count = global_metrics.get_histogram_stats("default_timer_test")["count"]

        with MetricsTimer("default_timer_test"):
            pass

        final_count = global_metrics.get_histogram_stats("default_timer_test")["count"]
        assert final_count == initial_count + 1

    def test_metrics_timer_records_even_on_exception(self):
        """MetricsTimer should record timing even when exception occurs."""
        metrics = ProjectionMetrics()

        with pytest.raises(ValueError):
            with MetricsTimer("exception_timing", metrics):
                raise ValueError("Test exception")

        stats = metrics.get_histogram_stats("exception_timing")
        assert stats["count"] == 1

    def test_metrics_timer_returns_self(self):
        """MetricsTimer __enter__ should return self."""
        metrics = ProjectionMetrics()

        with MetricsTimer("test_timing", metrics) as timer:
            assert isinstance(timer, MetricsTimer)


class TestGetMetrics:
    """Tests for global metrics singleton accessor."""

    def test_get_metrics_returns_projection_metrics_instance(self):
        """get_metrics should return a ProjectionMetrics instance."""
        metrics = get_metrics()

        assert isinstance(metrics, ProjectionMetrics)

    def test_get_metrics_returns_same_instance_on_multiple_calls(self):
        """get_metrics should return the same singleton instance."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()

        assert metrics1 is metrics2

    def test_get_metrics_state_persists_across_calls(self):
        """State set via one call should be visible in subsequent calls."""
        metrics1 = get_metrics()
        metrics1.increment_counter("persistence_test", 42)

        metrics2 = get_metrics()

        assert metrics2.get_counter("persistence_test") == 42


class TestMetricConstants:
    """Tests for metric name constants."""

    def test_metric_constants_are_strings(self):
        """All metric constants should be strings."""
        assert isinstance(METRIC_EVENTS_PROCESSED, str)
        assert isinstance(METRIC_EVENTS_FAILED, str)
        assert isinstance(METRIC_PROCESSING_LATENCY, str)

    def test_metric_constants_can_be_used_with_metrics(self):
        """Metric constants should work with metrics methods."""
        metrics = ProjectionMetrics()

        metrics.increment_counter(METRIC_EVENTS_PROCESSED, 1)
        metrics.increment_counter(METRIC_EVENTS_FAILED, 2)
        metrics.record_histogram(METRIC_PROCESSING_LATENCY, 10.5)

        assert metrics.get_counter(METRIC_EVENTS_PROCESSED) == 1
        assert metrics.get_counter(METRIC_EVENTS_FAILED) == 2
        assert metrics.get_histogram_stats(METRIC_PROCESSING_LATENCY)["count"] == 1
