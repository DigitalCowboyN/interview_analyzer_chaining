"""
metrics.py

Provides a simple centralized tracker for key performance and cost metrics.
"""

from time import time

class MetricsTracker:
    """
    A simple singleton-like class (by convention) to track metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracked metrics to their initial state."""
        self.api_calls: int = 0
        self.total_tokens: int = 0
        self.pipeline_start_time: float | None = None
        self.pipeline_end_time: float | None = None
        self.errors: int = 0 # Basic error tracking

    def increment_api_calls(self, count: int = 1):
        """Increments the total number of successful API calls."""
        self.api_calls += count

    def add_tokens(self, count: int):
        """Adds to the total token count."""
        # Check if count is valid (OpenAI usage object might be None)
        if isinstance(count, int) and count >= 0:
            self.total_tokens += count
        # Optionally log a warning if count is invalid/None

    def start_pipeline_timer(self):
        """Records the start time of the pipeline."""
        self.pipeline_start_time = time()

    def stop_pipeline_timer(self):
        """Records the end time of the pipeline."""
        self.pipeline_end_time = time()

    def increment_errors(self, count: int = 1):
         """Increments the count of errors encountered."""
         self.errors += count

    def get_summary(self) -> dict:
        """Returns a dictionary summarizing the tracked metrics."""
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