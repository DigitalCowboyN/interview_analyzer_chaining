"""
metrics.py

Provides a simple centralized tracker for key performance and cost metrics.
"""

from time import time
from typing import Any, Dict


class MetricsTracker:
    """
    A simple singleton-like class (by convention) to track key operational metrics.

    Provides methods to increment counters for API calls, token usage, and errors,
    and to time the overall pipeline execution. Also allows setting arbitrary custom metrics.
    A global instance `metrics_tracker` is provided for convenience.

    Attributes:
        api_calls (int): Count of successful API calls made.
        total_tokens (int): Total number of tokens processed (input + output).
        pipeline_start_time (float | None): Timestamp when the pipeline started.
        pipeline_end_time (float | None): Timestamp when the pipeline ended.
        errors (int): Count of errors encountered during processing.
        custom_metrics (Dict[str, Dict[str, Any]]): Storage for arbitrary metrics categorized by scope.
    """

    def __init__(self):
        """Initializes the MetricsTracker by resetting all counters."""
        self.reset()

    def reset(self):
        """Resets all tracked metrics to their initial state (0 or None)."""
        self.api_calls: int = 0
        self.total_tokens: int = 0
        self.pipeline_start_time: float | None = None
        self.pipeline_end_time: float | None = None
        self.errors: int = 0  # Basic error tracking
        self.custom_metrics: Dict[str, Dict[str, Any]] = (
            {}
        )  # Initialize custom metrics storage
        self.file_timers: Dict[str, float] = (
            {}
        )  # Track start times for individual files

    def increment_api_calls(self, count: int = 1):
        """
        Increments the total number of successful API calls.

        Args:
            count (int): The number of calls to add. Defaults to 1.
        """
        self.api_calls += count

    def add_tokens(self, count: int):
        """
        Adds to the total token count, ensuring the count is a valid non-negative integer.

        Args:
            count (int): The number of tokens to add.
        """
        # Check if count is valid (OpenAI usage object might be None)
        if isinstance(count, int) and count >= 0:
            self.total_tokens += count
        # Optionally log a warning if count is invalid/None

    def start_pipeline_timer(self):
        """Records the start time of the pipeline using `time.time()`."""
        self.pipeline_start_time = time()

    def stop_pipeline_timer(self):
        """Records the end time of the pipeline using `time.time()`."""
        self.pipeline_end_time = time()

    def increment_errors(self, category: str = "global", count: int = 1):
        """
        Increments the count of errors encountered, optionally by category.

        Args:
            category (str): The category or scope of the error (e.g., 'pipeline', 'file_name'). Defaults to 'global'.
            count (int): The number of errors to add. Defaults to 1.
        """
        self.errors += count  # Keep global count for simplicity in summary? Or make it category based?
        # Optionally track errors per category
        # self.set_metric(category, "errors", self.custom_metrics.get(category, {}).get("errors", 0) + count)

    # --- Add start/stop file timer ---
    def start_file_timer(self, filename: str):
        """Records the start time for processing a specific file."""
        self.file_timers[filename] = time()

    def stop_file_timer(self, filename: str):
        """Calculates and stores the elapsed time for a processed file."""
        if filename in self.file_timers:
            elapsed = time() - self.file_timers[filename]
            self.set_metric(filename, "processing_time_seconds", round(elapsed, 2))
            # Remove from active timers once stopped
            # del self.file_timers[filename]  # Keep it? Or clear? Keep for now.
        else:
            # Log a warning?
            pass

    # --- Add set_metric method ---
    def set_metric(self, category: str, key: str, value: Any):
        """
        Sets or updates a custom metric within a specific category.

        Args:
            category (str): The category for the metric (e.g., 'pipeline', filename).
            key (str): The name of the metric (e.g., 'files_processed', 'verification_errors').
            value (Any): The value of the metric.
        """
        if category not in self.custom_metrics:
            self.custom_metrics[category] = {}
        self.custom_metrics[category][key] = value

    # --- Add increment_metric method (optional helper) ---
    def increment_metric(self, category: str, key: str, increment_by: int = 1):
        """
        Increments a numeric custom metric within a specific category.

        Args:
            category (str): The category for the metric.
            key (str): The name of the metric.
            increment_by (int): The value to add to the metric. Defaults to 1.
        """
        current_value = self.custom_metrics.get(category, {}).get(key, 0)
        if isinstance(current_value, (int, float)):
            self.set_metric(category, key, current_value + increment_by)
        else:
            # Log warning: trying to increment non-numeric metric
            self.set_metric(category, key, increment_by)  # Overwrite if not numeric?

    # --- Add increment_results_processed (specific helper needed in pipeline) ---
    def increment_results_processed(self, filename: str, count: int = 1):
        """Helper to increment results processed count for a file."""
        self.increment_metric(filename, "results_processed", count)

    # --- Add convenience methods for backward compatibility with tests ---
    def increment_files_processed(self, count: int = 1):
        """Helper to increment files processed count."""
        self.increment_metric("pipeline", "files_processed", count)

    def increment_files_failed(self, count: int = 1):
        """Helper to increment files failed count."""
        self.increment_metric("pipeline", "files_failed", count)

    def add_processing_time(self, sentence_id: str, processing_time: float):
        """
        Records the processing time for a specific sentence.

        Args:
            sentence_id (str): The identifier for the sentence.
            processing_time (float): The processing time in seconds.
        """
        self.set_metric("sentences", f"processing_time_{sentence_id}", processing_time)

    def increment_sentences_success(self, count: int = 1):
        """Helper to increment successful sentences count."""
        self.increment_metric("pipeline", "sentences_success", count)

    # --- Dual-Write Metrics (M2.2 Event-First Architecture) ---

    def increment_event_emission_success(self, event_type: str, count: int = 1):
        """
        Increments the count of successful event emissions by event type.

        Args:
            event_type (str): The type of event (e.g., 'InterviewCreated', 'SentenceCreated').
            count (int): The number of successful emissions to add. Defaults to 1.
        """
        self.increment_metric("events", f"emission_success_{event_type}", count)

    def increment_event_emission_failure(self, event_type: str, count: int = 1):
        """
        Increments the count of failed event emissions by event type.

        Args:
            event_type (str): The type of event (e.g., 'InterviewCreated', 'SentenceCreated').
            count (int): The number of failed emissions to add. Defaults to 1.
        """
        self.increment_metric("events", f"emission_failure_{event_type}", count)

    def increment_dual_write_event_first_success(self, count: int = 1):
        """
        Increments the count of successful event-first writes (event succeeded, proceeding to Neo4j).

        Args:
            count (int): The number of successful event-first writes. Defaults to 1.
        """
        self.increment_metric("dual_write", "event_first_success", count)

    def increment_dual_write_event_first_failure(self, count: int = 1):
        """
        Increments the count of failed event-first writes (event failed, operation aborted).

        Args:
            count (int): The number of failed event-first writes. Defaults to 1.
        """
        self.increment_metric("dual_write", "event_first_failure", count)

    def increment_dual_write_neo4j_after_event_success(self, count: int = 1):
        """
        Increments the count of successful Neo4j writes after event emission.

        Args:
            count (int): The number of successful Neo4j writes. Defaults to 1.
        """
        self.increment_metric("dual_write", "neo4j_after_event_success", count)

    def increment_dual_write_neo4j_after_event_failure(self, count: int = 1):
        """
        Increments the count of failed Neo4j writes after successful event emission.

        Args:
            count (int): The number of failed Neo4j writes. Defaults to 1.
        """
        self.increment_metric("dual_write", "neo4j_after_event_failure", count)

    def increment_projection_duplicate_skipped(self, aggregate_type: str, count: int = 1):
        """
        Increments the count of duplicate nodes skipped by projection service.

        Args:
            aggregate_type (str): The type of aggregate (e.g., 'Interview', 'Sentence').
            count (int): The number of duplicates skipped. Defaults to 1.
        """
        self.increment_metric("projections", f"duplicate_skipped_{aggregate_type}", count)

    def increment_projection_version_conflict(self, aggregate_type: str, count: int = 1):
        """
        Increments the count of version conflicts encountered by projection service.

        Args:
            aggregate_type (str): The type of aggregate (e.g., 'Interview', 'Sentence').
            count (int): The number of version conflicts. Defaults to 1.
        """
        self.increment_metric("projections", f"version_conflict_{aggregate_type}", count)

    def get_summary(self) -> dict:
        """
        Returns a dictionary summarizing the tracked metrics, including custom ones.

        Calculates the pipeline duration if start and end times are available.

        Returns:
            dict: A dictionary containing standard metrics and custom metrics.
        """
        duration = None
        if self.pipeline_start_time and self.pipeline_end_time:
            duration = self.pipeline_end_time - self.pipeline_start_time

        summary = {
            "total_api_calls": self.api_calls,
            "total_tokens_used": self.total_tokens,
            "total_errors": self.errors,  # Overall error count
            "pipeline_duration_seconds": duration,
            "custom_metrics": self.custom_metrics,  # Include all custom metrics
        }
        return summary


# Global instance
metrics_tracker = MetricsTracker()
