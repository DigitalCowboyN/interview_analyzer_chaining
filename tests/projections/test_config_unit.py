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
