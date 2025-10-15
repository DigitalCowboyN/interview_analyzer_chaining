"""
Metrics tracking for projection service.

Tracks events processed, errors, latency, and queue depths.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict

logger = logging.getLogger(__name__)


class ProjectionMetrics:
    """
    Metrics collector for projection service.

    Tracks counters, gauges, and histograms for monitoring.
    """

    def __init__(self):
        """Initialize metrics."""
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = defaultdict(list)
        self.started_at = datetime.now(timezone.utc)

    def increment_counter(self, name: str, value: int = 1):
        """
        Increment a counter.

        Args:
            name: Counter name
            value: Amount to increment by
        """
        self.counters[name] += value

    def set_gauge(self, name: str, value: float):
        """
        Set a gauge value.

        Args:
            name: Gauge name
            value: Value to set
        """
        self.gauges[name] = value

    def record_histogram(self, name: str, value: float):
        """
        Record a histogram value.

        Args:
            name: Histogram name
            value: Value to record
        """
        self.histograms[name].append(value)

        # Keep only last 1000 values to prevent memory growth
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """Get gauge value."""
        return self.gauges.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> Dict:
        """
        Get histogram statistics.

        Args:
            name: Histogram name

        Returns:
            Dict: Statistics (count, min, max, avg, p50, p95, p99)
        """
        values = self.histograms.get(name, [])
        if not values:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
            }

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "count": count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.50)],
            "p95": sorted_values[int(count * 0.95)] if count > 20 else sorted_values[-1],
            "p99": sorted_values[int(count * 0.99)] if count > 100 else sorted_values[-1],
        }

    def get_all_metrics(self) -> Dict:
        """Get all metrics."""
        uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()

        return {
            "uptime_seconds": uptime,
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in self.histograms.keys()
            },
        }

    def reset(self):
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.started_at = datetime.now(timezone.utc)


# Global metrics instance
_metrics: ProjectionMetrics = ProjectionMetrics()


def get_metrics() -> ProjectionMetrics:
    """Get the global metrics instance."""
    return _metrics


# Metric names (constants for consistency)
METRIC_EVENTS_PROCESSED = "events_processed"
METRIC_EVENTS_FAILED = "events_failed"
METRIC_EVENTS_PARKED = "events_parked"
METRIC_EVENTS_SKIPPED = "events_skipped"
METRIC_PROCESSING_LATENCY = "processing_latency_ms"
METRIC_QUEUE_DEPTH = "queue_depth"
METRIC_NEO4J_WRITE_ERRORS = "neo4j_write_errors"


class MetricsTimer:
    """Context manager for timing operations."""

    def __init__(self, metric_name: str, metrics: ProjectionMetrics = None):
        """
        Initialize timer.

        Args:
            metric_name: Name of the metric to record to
            metrics: Metrics instance (uses global if not provided)
        """
        self.metric_name = metric_name
        self.metrics = metrics or get_metrics()
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        if self.start_time:
            elapsed_ms = (time.time() - self.start_time) * 1000
            self.metrics.record_histogram(self.metric_name, elapsed_ms)
