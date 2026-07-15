"""
Unit tests for projection service configuration.

Pins the subscription allowlist contract in src/projections/config.py.
"""


def test_project_subscription_allowlist():
    """The $ce-Project subscription must allow all 7 resolution event types."""
    from src.projections.config import SUBSCRIPTION_CONFIG, is_event_allowed

    assert SUBSCRIPTION_CONFIG["project"]["stream"] == "$ce-Project"
    for event_type in (
        "EntityCanonicalized", "EntityAliasAdded", "EntityMergeConfirmed",
        "EntitySplit", "PersonIdentified", "SpeakerLinkedToPerson",
        "PersonLinkRemoved",
    ):
        assert is_event_allowed("project", event_type)


def test_interview_subscription_allowlist_includes_segment_events():
    """The $ce-Interview subscription must allow the M4.5c segment event types."""
    from src.projections.config import is_event_allowed

    assert is_event_allowed("interview", "SegmentIdentified")
    assert is_event_allowed("interview", "SegmentRemoved")
