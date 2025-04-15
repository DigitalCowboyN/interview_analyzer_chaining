"""
metrics.py

Provides a simple centralized tracker for key performance and cost metrics.
"""

from time import time

class MetricsTracker:
    """
    A simple singleton-like class (by convention) to track key operational metrics.

    Provides methods to increment counters for API calls, token usage, and errors,
    and to time the overall pipeline execution. A summary of metrics can be retrieved.
    A global instance `metrics_tracker` is provided for convenience.

    Attributes:
        api_calls (int): Count of successful API calls made.
        total_tokens (int): Total number of tokens processed (input + output).
        pipeline_start_time (float | None): Timestamp when the pipeline started.
        pipeline_end_time (float | None): Timestamp when the pipeline ended.
        errors (int): Count of errors encountered during processing.
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
        self.errors: int = 0 # Basic error tracking

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

    def increment_errors(self, count: int = 1):
         """
         Increments the count of errors encountered.

         Args:
            count (int): The number of errors to add. Defaults to 1.
         """
         self.errors += count

    def get_summary(self) -> dict:
        """
        Returns a dictionary summarizing the tracked metrics.

        Calculates the pipeline duration if start and end times are available.

        Returns:
            dict: A dictionary containing keys like 'total_api_calls',
                  'total_tokens_used', 'total_errors', and 
                  'pipeline_duration_seconds'.
        """
        duration = None
        if self.pipeline_start_time and self.pipeline_end_time:
            duration = self.pipeline_end_time - self.pipeline_start_time
        
        return {
            "total_api_calls": self.api_calls,
            "total_tokens_used": self.total_tokens,
            "total_errors": self.errors,
            "pipeline_duration_seconds": duration
        }

# Global instance
metrics_tracker = MetricsTracker() 