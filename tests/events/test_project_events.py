"""Project event payload models and deterministic id helpers (M4.5b)."""

import uuid

import pytest
from pydantic import ValidationError

from src.events.envelope import AggregateType
from src.events.project_events import (
    EntityAliasAddedData,
    EntityCanonicalizedData,
    EntityMergeConfirmedData,
    EntitySplitData,
    PersonIdentifiedData,
    PersonLinkRemovedData,
    SpeakerLinkedToPersonData,
    canonical_entity_id,
    person_id_for,
    project_aggregate_id,
)


class TestIdHelpers:
    def test_project_aggregate_id_is_deterministic_uuid(self):
        a = project_aggregate_id("default-project")
        assert a == project_aggregate_id("default-project")
        uuid.UUID(a)  # valid UUID (envelope validator requires it)
        assert a != project_aggregate_id("other-project")

    def test_canonical_entity_id_matches_spec_derivation(self):
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, "p1:entity:acme corp:ORG"))
        assert canonical_entity_id("p1", "acme corp", "ORG") == expected

    def test_person_id_matches_spec_derivation(self):
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, "p1:person:jane doe"))
        assert person_id_for("p1", "jane doe") == expected


class TestAggregateType:
    def test_project_member_value_is_wire_frozen(self):
        assert AggregateType.PROJECT.value == "Project"


class TestDataModels:
    def test_entity_canonicalized_requires_surfaces(self):
        with pytest.raises(ValidationError):
            EntityCanonicalizedData(
                project_id="p1", canonical_id="c1", name="Acme",
                entity_type="ORG", surfaces=[], method="deterministic", confidence=1.0,
            )

    def test_method_literal_rejected(self):
        with pytest.raises(ValidationError):
            EntityAliasAddedData(
                project_id="p1", canonical_id="c1", surface="acme",
                method="llm", confidence=0.9,
            )

    def test_link_method_literal(self):
        ok = SpeakerLinkedToPersonData(
            project_id="p1", interview_id="i1", speaker_id="s1",
            person_id="per1", method="front_matter", confidence=1.0,
        )
        assert ok.method == "front_matter"
        with pytest.raises(ValidationError):
            SpeakerLinkedToPersonData(
                project_id="p1", interview_id="i1", speaker_id="s1",
                person_id="per1", method="fuzzy", confidence=1.0,
            )

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            EntityAliasAddedData(
                project_id="p1", canonical_id="c1", surface="acme",
                method="human", confidence=1.5,
            )

    def test_remaining_models_construct(self):
        EntityMergeConfirmedData(project_id="p1", canonical_id="c1", merged_canonical_id="c2")
        EntitySplitData(
            project_id="p1", canonical_id="c1", surfaces_removed=["a"],
            new_canonical_id="c3", new_name="A",
        )
        PersonIdentifiedData(project_id="p1", person_id="per1", display_name="Jane Doe")
        PersonLinkRemovedData(
            project_id="p1", interview_id="i1", speaker_id="s1", person_id="per1", note=None
        )
