"""
Configuration for the projection service.

Defines lane count, subscription settings, retry policies, and event allowlists.
"""

import os
from typing import Dict, List

# Lane configuration
LANE_COUNT = int(os.getenv("PROJECTION_LANE_COUNT", "12"))
QUEUE_DEPTH_ALERT_THRESHOLD = int(os.getenv("PROJECTION_QUEUE_ALERT_THRESHOLD", "100"))

# Retry configuration
RETRY_MAX_ATTEMPTS = int(os.getenv("PROJECTION_RETRY_MAX_ATTEMPTS", "5"))
RETRY_INITIAL_DELAY = float(os.getenv("PROJECTION_RETRY_INITIAL_DELAY", "1.0"))
RETRY_MAX_DELAY = float(os.getenv("PROJECTION_RETRY_MAX_DELAY", "60.0"))
RETRY_EXPONENTIAL_BASE = float(os.getenv("PROJECTION_RETRY_EXPONENTIAL_BASE", "2.0"))

# Checkpoint configuration
CHECKPOINT_INTERVAL = int(os.getenv("PROJECTION_CHECKPOINT_INTERVAL", "100"))

# EventStoreDB configuration
ESDB_CONNECTION_STRING = os.getenv("ESDB_CONNECTION_STRING", "esdb://localhost:2113?tls=false")

# Feature flags
ENABLE_PROJECTION_SERVICE = os.getenv("ENABLE_PROJECTION_SERVICE", "true").lower() == "true"

# Subscription configuration
SUBSCRIPTION_CONFIG: Dict[str, Dict] = {
    "interview": {
        "stream": "$ce-Interview",  # Category stream for Interview aggregate
        "group": "neo4j-projection-interview-v1",
        "allowlist": [
            "InterviewCreated",
            "InterviewUpdated",
            "StatusChanged",
            "InterviewArchived",
            "InterviewDeleted",
        ],
    },
    "sentence": {
        "stream": "$ce-Sentence",  # Category stream for Sentence aggregate
        "group": "neo4j-projection-sentence-v1",
        "allowlist": [
            "SentenceCreated",
            "SentenceEdited",
            "SentenceRelocated",
            "AnalysisGenerated",
            "AnalysisRegenerated",
            "AnalysisOverridden",
            "AnalysisCleared",
            "SentenceTagged",
            "SentenceUntagged",
            "SentenceStatusChanged",
            "SentenceDeleted",
        ],
    },
}

# Parked events configuration
PARKED_EVENTS_STREAM_PREFIX = "parked-events"


def get_parked_stream_name(aggregate_type: str) -> str:
    """Get the parked events stream name for an aggregate type."""
    return f"{PARKED_EVENTS_STREAM_PREFIX}-{aggregate_type.lower()}"


def is_event_allowed(subscription_name: str, event_type: str) -> bool:
    """
    Check if an event type is allowed for a subscription.

    Args:
        subscription_name: Name of the subscription (interview/sentence)
        event_type: Type of event to check

    Returns:
        bool: True if event is allowed, False otherwise
    """
    config = SUBSCRIPTION_CONFIG.get(subscription_name)
    if not config:
        return False

    return event_type in config["allowlist"]


def get_all_allowed_event_types() -> List[str]:
    """Get all allowed event types across all subscriptions."""
    all_types = []
    for config in SUBSCRIPTION_CONFIG.values():
        all_types.extend(config["allowlist"])
    return list(set(all_types))  # Remove duplicates
