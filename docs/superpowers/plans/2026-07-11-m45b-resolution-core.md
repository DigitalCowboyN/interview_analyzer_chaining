# M4.5b: Resolution Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cross-interview resolution: canonical entities with aliases and real Person identities, driven by a new event-sourced Project aggregate, a deterministic+embedding ResolutionEngine, human corrections API, and consumer upgrades (worklist suggestions, Person-grouped rollup, canonical-keyed OKF bundles).

**Architecture:** Second of three plans from `docs/superpowers/specs/2026-07-10-layer4-schema-v2-design.md` (M4.5b section). A new `Project` aggregate (`Project-{uuid}` streams) holds resolution state; seven new events project to `(:CanonicalEntity)` / `(:Person)` overlay nodes with `ALIAS_OF` / `IDENTIFIED_AS` edges. The engine mirrors LensEngine (idempotent re-runs via aggregate state + deterministic uuid5 ids; locked items skipped). Suggestions are computed on demand, never stored as events.

**Tech Stack:** Python 3.10, pydantic, EventStoreDB (esdbclient), Neo4j async driver, FastAPI, pytest.

## Global Constraints

- **Wire format frozen (existing):** event type names (`SentenceCreated`, …), `AggregateType.SENTENCE`/`INTERVIEW` values, `Sentence-{id}`/`Interview-{id}` streams never change.
- **New wire format minted by THIS plan — choose once, frozen on merge:** `AggregateType.PROJECT = "Project"`; stream `Project-{aggregate_id}`; event type names `EntityCanonicalized`, `EntityAliasAdded`, `EntityMergeConfirmed`, `EntitySplit`, `PersonIdentified`, `SpeakerLinkedToPerson`, `PersonLinkRemoved`.
- **Project aggregate identity:** `EventEnvelope.aggregate_id` must be a valid UUID, but `project_id` is a human string (e.g. `"default-project"`). Therefore `aggregate_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"project:{project_id}"))` (helper `project_aggregate_id` in Task 1). Every Project event **payload** carries the human `project_id` — graph handlers MERGE on it (the `(:Project {project_id})` node already exists from `InterviewCreatedHandler`).
- **New-event delivery checklist (hard-won M4.3 lesson):** a new event type is NOT delivered until bootstrap registration + subscription allowlist + lane routing + pin-test updates are ALL wired (Task 5). Project events route lanes by `event.aggregate_id` (all events of one project serialize in one lane).
- **Locking discipline:** human-method events lock their targets; the engine skips locked canonicals and never re-links blocked `(interview_id, speaker_id)` pairs. Locks protect against the ENGINE; humans may operate on human-locked items (M4.3 lens-override precedent). 409s come from unknown ids / already-merged / already-linked, mapped from domain `ValueError`.
- **Entity surfaces in events are graph-form:** M4.2 projects `(:Entity {surface: toLower(ent.text), entity_type})`, so every surface string in a Project event is exactly a graph `e.surface` value (already lowercased).
- **uuid5 derivations (spec):** `canonical_id = uuid5(NAMESPACE_DNS, f"{project_id}:entity:{normalized_name}:{entity_type}")`; `person_id = uuid5(NAMESPACE_DNS, f"{project_id}:person:{normalized_display_name}")`.
- **Thresholds:** `config.get("resolution", {})` keys `auto_merge_threshold` (default **0.92**) and `suggest_threshold` (default **0.80**); defaults live in code, no config-file change required.
- Test runner: `./scripts/test.sh [paths]` (unit), `./scripts/test-integration.sh [paths]` (live infra). TDD per task; commit per task; flake8 clean on touched files.

## File Structure

- `src/events/project_events.py` (new) — payload models + uuid5 helpers
- `src/events/envelope.py` (modify) — `AggregateType.PROJECT`
- `src/events/aggregates.py` (modify) — `class Project(AggregateRoot)` + `_add_event` branch
- `src/events/repository.py` (modify) — `ProjectRepository` + factory/getter
- `src/projections/handlers/resolution_handlers.py` (new) — 7 handlers
- `src/projections/bootstrap.py`, `src/projections/config.py`, `src/projections/lane_manager.py` (modify) — wiring
- `src/resolution/__init__.py`, `reader.py`, `candidates.py`, `engine.py`, `suggestions.py`, `__main__.py` (new)
- `src/api/routers/resolution.py` (new) + `src/main.py` (modify) — corrections API
- `src/api/routers/queries.py`, `src/export/reader.py`, `src/export/renderer.py`, `src/export/bundler.py` (modify) — consumer upgrades
- `tests/…` mirroring each (see tasks); `tests/integration/test_layer4_resolution_smoke.py` (new)

---

### Task 1: Project event payload models + AggregateType.PROJECT

**Files:**
- Create: `src/events/project_events.py`
- Modify: `src/events/envelope.py` (AggregateType)
- Test: `tests/events/test_project_events.py`

**Interfaces:**
- Produces: `project_aggregate_id(project_id) -> str`, `canonical_entity_id(project_id, normalized_name, entity_type) -> str`, `person_id_for(project_id, normalized_display_name) -> str`, the 7 `*Data` pydantic models, `AggregateType.PROJECT`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/events/test_project_events.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/events/test_project_events.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: src.events.project_events` (and missing `PROJECT` member).

- [ ] **Step 3: Implement**

In `src/events/envelope.py`, extend the enum (wire format — never change these values):

```python
class AggregateType(str, Enum):
    """Type of aggregate the event belongs to."""

    INTERVIEW = "Interview"
    SENTENCE = "Sentence"
    PROJECT = "Project"
```

Create `src/events/project_events.py`:

```python
"""Event payload models for the Project aggregate (M4.5b resolution core).

Wire format, frozen once merged: the event type names registered for these
payloads, AggregateType.PROJECT's "Project" value, and Project-{aggregate_id}
stream names. Envelope aggregate ids must be UUIDs, so the aggregate id is
uuid5 of the human project_id (project_aggregate_id); payloads carry the
human project_id for the graph handlers, which MERGE on it.
"""

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ResolutionMethod = Literal["deterministic", "human"]
LinkMethod = Literal["exact_name", "front_matter", "human"]


def project_aggregate_id(project_id: str) -> str:
    """Deterministic UUID for a project's event stream."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"project:{project_id}"))


def canonical_entity_id(project_id: str, normalized_name: str, entity_type: str) -> str:
    """Spec derivation: uuid5('{project_id}:entity:{normalized_name}:{entity_type}')."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{project_id}:entity:{normalized_name}:{entity_type}"))


def person_id_for(project_id: str, normalized_display_name: str) -> str:
    """Spec derivation: uuid5('{project_id}:person:{normalized_display_name}')."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{project_id}:person:{normalized_display_name}"))


class EntityCanonicalizedData(BaseModel):
    project_id: str
    canonical_id: str
    name: str
    entity_type: str
    surfaces: List[str] = Field(..., min_length=1)
    method: ResolutionMethod
    confidence: float = Field(..., ge=0.0, le=1.0)


class EntityAliasAddedData(BaseModel):
    project_id: str
    canonical_id: str
    surface: str
    method: ResolutionMethod
    confidence: float = Field(..., ge=0.0, le=1.0)


class EntityMergeConfirmedData(BaseModel):
    project_id: str
    canonical_id: str  # survivor
    merged_canonical_id: str


class EntitySplitData(BaseModel):
    project_id: str
    canonical_id: str
    surfaces_removed: List[str] = Field(..., min_length=1)
    new_canonical_id: str
    new_name: str


class PersonIdentifiedData(BaseModel):
    project_id: str
    person_id: str
    display_name: str


class SpeakerLinkedToPersonData(BaseModel):
    project_id: str
    interview_id: str
    speaker_id: str
    person_id: str
    method: LinkMethod
    confidence: float = Field(..., ge=0.0, le=1.0)


class PersonLinkRemovedData(BaseModel):
    project_id: str
    interview_id: str
    speaker_id: str
    person_id: str
    note: Optional[str] = None
```

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/events/test_project_events.py -q --no-cov` → all PASS.
Then `./scripts/test.sh tests/events -q --no-cov` → no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/events/project_events.py src/events/envelope.py tests/events/test_project_events.py
git commit -m "feat(resolution): Project event payload models + AggregateType.PROJECT (new wire format)"
```

---

### Task 2: Project aggregate

**Files:**
- Modify: `src/events/aggregates.py` (add `Project` after `Fragment`; extend `_add_event`)
- Test: `tests/events/test_project_aggregate.py`

**Interfaces:**
- Consumes: Task 1 models/helpers.
- Produces: `Project(AggregateRoot)` with state `project_id`, `canonical_entities: Dict[cid, {name, entity_type, surfaces, locked, merged_into}]`, `persons: Dict[pid, {display_name, links: List[[interview_id, speaker_id]]}]`, `blocked_links: set[(interview_id, speaker_id)]`; queries `canonical_for_surface(surface, entity_type) -> Optional[str]`, `link_for_speaker(interview_id, speaker_id) -> Optional[str]`; domain methods `canonicalize_entity`, `add_entity_alias`, `confirm_entity_merge`, `split_entity`, `identify_person`, `link_speaker_to_person`, `remove_person_link` — all take `project_id` as first arg and accept `**envelope_kwargs` (actor, correlation_id).

- [ ] **Step 1: Write the failing tests**

```python
# tests/events/test_project_aggregate.py
"""Project aggregate: domain invariants, locking, replay round-trip (M4.5b)."""

import pytest

from src.events.aggregates import Project
from src.events.project_events import project_aggregate_id

P = "default-project"


def _project() -> Project:
    return Project(project_aggregate_id(P))


def _canonicalize(p, cid="c1", surfaces=("acme corp",), method="deterministic"):
    p.canonicalize_entity(P, cid, "Acme Corp", "ORG", list(surfaces), method, 1.0)


class TestEntityInvariants:
    def test_canonicalize_and_state(self):
        p = _project()
        _canonicalize(p)
        assert p.project_id == P
        entry = p.canonical_entities["c1"]
        assert entry["surfaces"] == ["acme corp"]
        assert entry["locked"] is False
        assert p.canonical_for_surface("acme corp", "ORG") == "c1"
        assert p.canonical_for_surface("acme corp", "PERSON") is None

    def test_human_canonicalize_locks(self):
        p = _project()
        _canonicalize(p, method="human")
        assert p.canonical_entities["c1"]["locked"] is True

    def test_duplicate_canonical_id_raises(self):
        p = _project()
        _canonicalize(p)
        with pytest.raises(ValueError):
            _canonicalize(p)

    def test_surface_already_owned_raises(self):
        p = _project()
        _canonicalize(p)
        with pytest.raises(ValueError):
            p.canonicalize_entity(P, "c2", "Acme", "ORG", ["acme corp"], "deterministic", 1.0)

    def test_alias_add_and_locked_guard(self):
        p = _project()
        _canonicalize(p)
        p.add_entity_alias(P, "c1", "acme", "deterministic", 0.95)
        assert "acme" in p.canonical_entities["c1"]["surfaces"]
        p.add_entity_alias(P, "c1", "acme inc", "human", 1.0)  # human alias locks
        assert p.canonical_entities["c1"]["locked"] is True
        with pytest.raises(ValueError):  # engine may not touch locked
            p.add_entity_alias(P, "c1", "acme co", "deterministic", 0.99)

    def test_alias_unknown_canonical_raises(self):
        p = _project()
        with pytest.raises(ValueError):
            p.add_entity_alias(P, "nope", "x", "deterministic", 1.0)

    def test_merge_moves_surfaces_and_locks_both(self):
        p = _project()
        _canonicalize(p, "c1", ("acme corp",))
        p.canonicalize_entity(P, "c2", "Acme Inc", "ORG", ["acme inc"], "deterministic", 1.0)
        p.confirm_entity_merge(P, "c1", "c2")
        assert "acme inc" in p.canonical_entities["c1"]["surfaces"]
        assert p.canonical_entities["c1"]["locked"] is True
        assert p.canonical_entities["c2"]["merged_into"] == "c1"
        # merged-away canonical no longer owns surfaces
        assert p.canonical_for_surface("acme inc", "ORG") == "c1"

    def test_merge_unknown_or_merged_raises(self):
        p = _project()
        _canonicalize(p, "c1")
        with pytest.raises(ValueError):
            p.confirm_entity_merge(P, "c1", "ghost")
        p.canonicalize_entity(P, "c2", "B", "ORG", ["b"], "deterministic", 1.0)
        p.confirm_entity_merge(P, "c1", "c2")
        with pytest.raises(ValueError):  # c2 already merged away
            p.confirm_entity_merge(P, "c1", "c2")
        with pytest.raises(ValueError):  # self-merge
            p.confirm_entity_merge(P, "c1", "c1")

    def test_split_moves_surfaces_and_locks(self):
        p = _project()
        _canonicalize(p, "c1", ("acme corp", "acme berlin"))
        p.split_entity(P, "c1", ["acme berlin"], "c9", "Acme Berlin")
        assert p.canonical_entities["c1"]["surfaces"] == ["acme corp"]
        new = p.canonical_entities["c9"]
        assert new["surfaces"] == ["acme berlin"]
        assert new["locked"] is True and p.canonical_entities["c1"]["locked"] is True
        assert new["entity_type"] == "ORG"

    def test_split_requires_proper_subset(self):
        p = _project()
        _canonicalize(p, "c1", ("acme corp", "acme berlin"))
        with pytest.raises(ValueError):  # all surfaces
            p.split_entity(P, "c1", ["acme corp", "acme berlin"], "c9", "X")
        with pytest.raises(ValueError):  # not owned
            p.split_entity(P, "c1", ["ghost"], "c9", "X")


class TestPersonInvariants:
    def test_identify_and_link(self):
        p = _project()
        p.identify_person(P, "per1", "Jane Doe")
        p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        assert p.link_for_speaker("i1", "s1") == "per1"

    def test_duplicate_person_raises(self):
        p = _project()
        p.identify_person(P, "per1", "Jane Doe")
        with pytest.raises(ValueError):
            p.identify_person(P, "per1", "Jane Doe")

    def test_link_unknown_person_or_double_link_raises(self):
        p = _project()
        with pytest.raises(ValueError):
            p.link_speaker_to_person(P, "i1", "s1", "ghost", "exact_name", 1.0)
        p.identify_person(P, "per1", "Jane Doe")
        p.identify_person(P, "per2", "Jane D")
        p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        with pytest.raises(ValueError):
            p.link_speaker_to_person(P, "i1", "s1", "per2", "exact_name", 1.0)

    def test_unlink_blocks_engine_but_not_human(self):
        p = _project()
        p.identify_person(P, "per1", "Jane Doe")
        p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        p.remove_person_link(P, "i1", "s1", "per1", note="wrong Jane")
        assert p.link_for_speaker("i1", "s1") is None
        assert ("i1", "s1") in p.blocked_links
        with pytest.raises(ValueError):  # engine re-link blocked
            p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        p.link_speaker_to_person(P, "i1", "s1", "per1", "human", 1.0)  # human may
        assert ("i1", "s1") not in p.blocked_links

    def test_unlink_nonexistent_raises(self):
        p = _project()
        p.identify_person(P, "per1", "Jane Doe")
        with pytest.raises(ValueError):
            p.remove_person_link(P, "i1", "s1", "per1", note=None)


class TestReplay:
    def test_replay_round_trip(self):
        p = _project()
        _canonicalize(p, "c1", ("acme corp",))
        p.add_entity_alias(P, "c1", "acme", "deterministic", 0.95)
        p.canonicalize_entity(P, "c2", "B", "ORG", ["b"], "deterministic", 1.0)
        p.confirm_entity_merge(P, "c1", "c2")
        p.identify_person(P, "per1", "Jane Doe")
        p.link_speaker_to_person(P, "i1", "s1", "per1", "exact_name", 1.0)
        p.remove_person_link(P, "i1", "s1", "per1", note="oops")
        events = p.get_uncommitted_events()
        assert events[0].aggregate_type == "Project"  # wire value

        fresh = _project()
        fresh.load_from_history(events)
        assert fresh.canonical_entities == p.canonical_entities
        assert fresh.persons == p.persons
        assert fresh.blocked_links == p.blocked_links
        assert fresh.project_id == P
        assert fresh.version == len(events) - 1
```

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/events/test_project_aggregate.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'Project'`.

- [ ] **Step 3: Implement**

In `src/events/aggregates.py`: add to the imports block

```python
from .project_events import (
    EntityAliasAddedData,
    EntityCanonicalizedData,
    EntityMergeConfirmedData,
    EntitySplitData,
    PersonIdentifiedData,
    PersonLinkRemovedData,
    SpeakerLinkedToPersonData,
)
```

In `_add_event`, extend the aggregate-type dispatch (after the `Fragment` branch):

```python
        elif isinstance(self, Project):
            aggregate_type = AggregateType.PROJECT
```

Add the class after `Fragment` (before the deprecated `Sentence = Fragment` alias line; keep that alias LAST in the file):

```python
class Project(AggregateRoot):
    """Cross-interview resolution home (M4.5b).

    Holds canonical entities (with alias surfaces) and Person identities.
    aggregate_id is uuid5 of the human project_id (see
    project_events.project_aggregate_id); event payloads carry the human
    project_id for the graph handlers. Streams are Project-{aggregate_id}
    (wire format, frozen).
    """

    def __init__(self, aggregate_id: str):
        super().__init__(aggregate_id)
        self.project_id: Optional[str] = None
        # canonical_id -> {name, entity_type, surfaces: [str], locked: bool, merged_into: Optional[str]}
        self.canonical_entities: Dict[str, Dict[str, Any]] = {}
        # person_id -> {display_name, links: [[interview_id, speaker_id], ...]}
        self.persons: Dict[str, Dict[str, Any]] = {}
        # (interview_id, speaker_id) pairs a human unlinked: never auto re-linked
        self.blocked_links: set = set()
        self.updated_at: Optional[datetime] = None

    def apply_event(self, event: EventEnvelope) -> None:
        """Apply an event to update the project's state."""
        handlers = {
            "EntityCanonicalized": self._apply_entity_canonicalized,
            "EntityAliasAdded": self._apply_entity_alias_added,
            "EntityMergeConfirmed": self._apply_entity_merge_confirmed,
            "EntitySplit": self._apply_entity_split,
            "PersonIdentified": self._apply_person_identified,
            "SpeakerLinkedToPerson": self._apply_speaker_linked_to_person,
            "PersonLinkRemoved": self._apply_person_link_removed,
        }
        handler = handlers.get(event.event_type)
        if handler is None:
            raise ValueError(f"Unknown event type for Project: {event.event_type}")
        handler(event)

    # --- queries -----------------------------------------------------------

    def canonical_for_surface(self, surface: str, entity_type: str) -> Optional[str]:
        """The live (non-merged) canonical owning a surface, if any."""
        for cid, entry in self.canonical_entities.items():
            if entry["merged_into"] is not None:
                continue
            if entry["entity_type"] == entity_type and surface in entry["surfaces"]:
                return cid
        return None

    def link_for_speaker(self, interview_id: str, speaker_id: str) -> Optional[str]:
        """The person a speaker is linked to, if any."""
        for pid, person in self.persons.items():
            if [interview_id, speaker_id] in person["links"]:
                return pid
        return None

    # --- entity domain methods ---------------------------------------------

    def canonicalize_entity(
        self,
        project_id: str,
        canonical_id: str,
        name: str,
        entity_type: str,
        surfaces: List[str],
        method: str,
        confidence: float,
        **envelope_kwargs,
    ) -> EventEnvelope:
        if canonical_id in self.canonical_entities:
            raise ValueError(f"Canonical entity {canonical_id} already exists")
        for surface in surfaces:
            owner = self.canonical_for_surface(surface, entity_type)
            if owner is not None:
                raise ValueError(f"Surface {surface!r} already belongs to {owner}")
        data = EntityCanonicalizedData(
            project_id=project_id, canonical_id=canonical_id, name=name,
            entity_type=entity_type, surfaces=surfaces, method=method,
            confidence=confidence,
        ).model_dump()
        return self._add_event("EntityCanonicalized", data, project_id=project_id, **envelope_kwargs)

    def add_entity_alias(
        self, project_id: str, canonical_id: str, surface: str,
        method: str, confidence: float, **envelope_kwargs,
    ) -> EventEnvelope:
        entry = self.canonical_entities.get(canonical_id)
        if entry is None or entry["merged_into"] is not None:
            raise ValueError(f"Canonical entity {canonical_id} not found")
        if entry["locked"] and method != "human":
            raise ValueError(f"Canonical entity {canonical_id} is locked")
        owner = self.canonical_for_surface(surface, entry["entity_type"])
        if owner is not None:
            raise ValueError(f"Surface {surface!r} already belongs to {owner}")
        data = EntityAliasAddedData(
            project_id=project_id, canonical_id=canonical_id, surface=surface,
            method=method, confidence=confidence,
        ).model_dump()
        return self._add_event("EntityAliasAdded", data, project_id=project_id, **envelope_kwargs)

    def confirm_entity_merge(
        self, project_id: str, canonical_id: str, merged_canonical_id: str, **envelope_kwargs,
    ) -> EventEnvelope:
        if canonical_id == merged_canonical_id:
            raise ValueError("Cannot merge a canonical entity into itself")
        for cid in (canonical_id, merged_canonical_id):
            entry = self.canonical_entities.get(cid)
            if entry is None:
                raise ValueError(f"Canonical entity {cid} not found")
            if entry["merged_into"] is not None:
                raise ValueError(f"Canonical entity {cid} was already merged away")
        data = EntityMergeConfirmedData(
            project_id=project_id, canonical_id=canonical_id,
            merged_canonical_id=merged_canonical_id,
        ).model_dump()
        return self._add_event("EntityMergeConfirmed", data, project_id=project_id, **envelope_kwargs)

    def split_entity(
        self, project_id: str, canonical_id: str, surfaces_removed: List[str],
        new_canonical_id: str, new_name: str, **envelope_kwargs,
    ) -> EventEnvelope:
        entry = self.canonical_entities.get(canonical_id)
        if entry is None or entry["merged_into"] is not None:
            raise ValueError(f"Canonical entity {canonical_id} not found")
        if new_canonical_id in self.canonical_entities:
            raise ValueError(f"Canonical entity {new_canonical_id} already exists")
        missing = [s for s in surfaces_removed if s not in entry["surfaces"]]
        if missing:
            raise ValueError(f"Surfaces not owned by {canonical_id}: {missing}")
        if len(surfaces_removed) >= len(entry["surfaces"]):
            raise ValueError("Split must leave at least one surface behind")
        data = EntitySplitData(
            project_id=project_id, canonical_id=canonical_id,
            surfaces_removed=surfaces_removed, new_canonical_id=new_canonical_id,
            new_name=new_name,
        ).model_dump()
        return self._add_event("EntitySplit", data, project_id=project_id, **envelope_kwargs)

    # --- person domain methods ----------------------------------------------

    def identify_person(
        self, project_id: str, person_id: str, display_name: str, **envelope_kwargs,
    ) -> EventEnvelope:
        if person_id in self.persons:
            raise ValueError(f"Person {person_id} already exists")
        data = PersonIdentifiedData(
            project_id=project_id, person_id=person_id, display_name=display_name,
        ).model_dump()
        return self._add_event("PersonIdentified", data, project_id=project_id, **envelope_kwargs)

    def link_speaker_to_person(
        self, project_id: str, interview_id: str, speaker_id: str, person_id: str,
        method: str, confidence: float, **envelope_kwargs,
    ) -> EventEnvelope:
        if person_id not in self.persons:
            raise ValueError(f"Person {person_id} not found")
        linked_to = self.link_for_speaker(interview_id, speaker_id)
        if linked_to is not None:
            raise ValueError(
                f"Speaker ({interview_id}, {speaker_id}) already linked to {linked_to}"
            )
        if (interview_id, speaker_id) in self.blocked_links and method != "human":
            raise ValueError(
                f"Speaker ({interview_id}, {speaker_id}) is blocked from auto-linking"
            )
        data = SpeakerLinkedToPersonData(
            project_id=project_id, interview_id=interview_id, speaker_id=speaker_id,
            person_id=person_id, method=method, confidence=confidence,
        ).model_dump()
        return self._add_event("SpeakerLinkedToPerson", data, project_id=project_id, **envelope_kwargs)

    def remove_person_link(
        self, project_id: str, interview_id: str, speaker_id: str, person_id: str,
        note: Optional[str] = None, **envelope_kwargs,
    ) -> EventEnvelope:
        person = self.persons.get(person_id)
        if person is None or [interview_id, speaker_id] not in person["links"]:
            raise ValueError(
                f"Speaker ({interview_id}, {speaker_id}) is not linked to {person_id}"
            )
        data = PersonLinkRemovedData(
            project_id=project_id, interview_id=interview_id, speaker_id=speaker_id,
            person_id=person_id, note=note,
        ).model_dump()
        return self._add_event("PersonLinkRemoved", data, project_id=project_id, **envelope_kwargs)

    # --- apply methods -------------------------------------------------------

    def _apply_entity_canonicalized(self, event: EventEnvelope) -> None:
        d = event.data
        self.project_id = d["project_id"]
        self.canonical_entities[d["canonical_id"]] = {
            "name": d["name"],
            "entity_type": d["entity_type"],
            "surfaces": list(d["surfaces"]),
            "locked": d["method"] == "human",
            "merged_into": None,
        }
        self.updated_at = event.occurred_at

    def _apply_entity_alias_added(self, event: EventEnvelope) -> None:
        d = event.data
        self.project_id = d["project_id"]
        entry = self.canonical_entities[d["canonical_id"]]
        entry["surfaces"].append(d["surface"])
        if d["method"] == "human":
            entry["locked"] = True
        self.updated_at = event.occurred_at

    def _apply_entity_merge_confirmed(self, event: EventEnvelope) -> None:
        d = event.data
        self.project_id = d["project_id"]
        survivor = self.canonical_entities[d["canonical_id"]]
        merged = self.canonical_entities[d["merged_canonical_id"]]
        for surface in merged["surfaces"]:
            if surface not in survivor["surfaces"]:
                survivor["surfaces"].append(surface)
        survivor["locked"] = True
        merged["locked"] = True
        merged["merged_into"] = d["canonical_id"]
        self.updated_at = event.occurred_at

    def _apply_entity_split(self, event: EventEnvelope) -> None:
        d = event.data
        self.project_id = d["project_id"]
        old = self.canonical_entities[d["canonical_id"]]
        old["surfaces"] = [s for s in old["surfaces"] if s not in d["surfaces_removed"]]
        old["locked"] = True
        self.canonical_entities[d["new_canonical_id"]] = {
            "name": d["new_name"],
            "entity_type": old["entity_type"],
            "surfaces": list(d["surfaces_removed"]),
            "locked": True,
            "merged_into": None,
        }
        self.updated_at = event.occurred_at

    def _apply_person_identified(self, event: EventEnvelope) -> None:
        d = event.data
        self.project_id = d["project_id"]
        self.persons[d["person_id"]] = {"display_name": d["display_name"], "links": []}
        self.updated_at = event.occurred_at

    def _apply_speaker_linked_to_person(self, event: EventEnvelope) -> None:
        d = event.data
        self.project_id = d["project_id"]
        self.persons[d["person_id"]]["links"].append([d["interview_id"], d["speaker_id"]])
        if d["method"] == "human":
            self.blocked_links.discard((d["interview_id"], d["speaker_id"]))
        self.updated_at = event.occurred_at

    def _apply_person_link_removed(self, event: EventEnvelope) -> None:
        d = event.data
        self.project_id = d["project_id"]
        person = self.persons[d["person_id"]]
        person["links"] = [
            link for link in person["links"]
            if link != [d["interview_id"], d["speaker_id"]]
        ]
        self.blocked_links.add((d["interview_id"], d["speaker_id"]))
        self.updated_at = event.occurred_at
```

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/events -q --no-cov` → all PASS (including existing alias/aggregate tests).

- [ ] **Step 5: Commit**

```bash
git add src/events/aggregates.py tests/events/test_project_aggregate.py
git commit -m "feat(resolution): Project aggregate with locking invariants and replayable state"
```

---

### Task 3: ProjectRepository + factory/getter

**Files:**
- Modify: `src/events/repository.py`
- Test: `tests/events/test_project_repository.py`

**Interfaces:**
- Produces: `ProjectRepository(Repository[Project])` with stream `Project-{aggregate_id}`; `RepositoryFactory.create_project_repository()`; module-level `get_project_repository() -> ProjectRepository`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/events/test_project_repository.py
"""ProjectRepository stream naming and factory wiring (M4.5b)."""

from unittest.mock import MagicMock

from src.events.aggregates import Project
from src.events.repository import (
    ProjectRepository,
    RepositoryFactory,
    get_project_repository,
)


class TestProjectRepository:
    def test_stream_name_is_wire_frozen(self):
        repo = ProjectRepository(MagicMock())
        assert repo._get_stream_name("abc-123") == "Project-abc-123"

    def test_creates_project_aggregate(self):
        repo = ProjectRepository(MagicMock())
        aggregate = repo._create_aggregate("abc-123")
        assert isinstance(aggregate, Project)
        assert aggregate.aggregate_id == "abc-123"

    def test_factory_creates_project_repository(self):
        factory = RepositoryFactory(MagicMock())
        assert isinstance(factory.create_project_repository(), ProjectRepository)

    def test_module_getter_returns_repository(self, monkeypatch):
        import src.events.repository as repo_module

        monkeypatch.setattr(repo_module, "_global_factory", RepositoryFactory(MagicMock()))
        assert isinstance(get_project_repository(), ProjectRepository)
```

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/events/test_project_repository.py -q --no-cov`
Expected: FAIL — `ImportError: cannot import name 'ProjectRepository'`.

- [ ] **Step 3: Implement**

In `src/events/repository.py`: extend the aggregates import to `from .aggregates import AggregateRoot, Fragment, Interview, Project`. After `SentenceRepository`, add:

```python
class ProjectRepository(Repository[Project]):
    """Repository for Project aggregates (M4.5b resolution core)."""

    def _create_aggregate(self, aggregate_id: str) -> Project:
        """Create a new Project instance."""
        return Project(aggregate_id)

    def _get_stream_name(self, aggregate_id: str) -> str:
        """Get the stream name for a Project aggregate (wire format: "Project-<id>")."""
        return f"Project-{aggregate_id}"
```

In `RepositoryFactory` (next to `create_interview_repository`):

```python
    def create_project_repository(self) -> ProjectRepository:
        """
        Create a ProjectRepository instance.

        Returns:
            ProjectRepository: Configured repository instance
        """
        return ProjectRepository(self.event_store)
```

At module level (next to `get_interview_repository`):

```python
def get_project_repository() -> ProjectRepository:
    """
    Get a ProjectRepository instance using the global factory.

    Returns:
        ProjectRepository: Configured repository instance
    """
    factory = get_repository_factory()
    return factory.create_project_repository()
```

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/events -q --no-cov` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/events/repository.py tests/events/test_project_repository.py
git commit -m "feat(resolution): ProjectRepository with Project-{id} streams"
```

---

### Task 4: Projection handlers for the 7 Project events

**Files:**
- Create: `src/projections/handlers/resolution_handlers.py`
- Test: `tests/projections/test_resolution_handlers.py`

**Interfaces:**
- Consumes: `BaseProjectionHandler` (`src/projections/handlers/base_handler.py`), event payload shapes from Task 1.
- Produces: `EntityCanonicalizedHandler`, `EntityAliasAddedHandler`, `EntityMergeConfirmedHandler`, `EntitySplitHandler`, `PersonIdentifiedHandler`, `SpeakerLinkedToPersonHandler`, `PersonLinkRemovedHandler` — each `async def apply(self, tx, event)`.

**Handler discipline (M4.3):** cross-stream ordering is guarded by RAISING when a MATCH target is missing — the base handler's retry/park machinery does the rest. These handlers check explicit `RETURN count(...)` values instead of write counters (edge MERGEs are no-ops on replay, so counters can be legitimately zero). Entity nodes live on the Sentence lane, Project events on their own lane: on a full replay a resolution event can arrive before its `(:Entity)` rows — the raise-for-retry is load-bearing, not defensive fluff.

Graph shapes consumed (existing): `(:Entity {surface, entity_type})` (surface already lowercased by M4.2), `(:Speaker {speaker_id})`, `(:Project {project_id})`.
Graph shapes produced: `(:CanonicalEntity {canonical_id, name, entity_type, project_id, method, confidence, locked, merged_into?})`, `(:Person {person_id, display_name, project_id})`, `(:Entity)-[:ALIAS_OF {project_id, method, confidence}]->(:CanonicalEntity)`, `(:Speaker)-[:IDENTIFIED_AS {method, confidence}]->(:Person)`.

- [ ] **Step 1: Write the failing tests**

Mirror the mocked-tx pattern of `tests/projections/test_lens_handlers.py` (MagicMock tx whose `run` returns an object with `async single()` / `consume()`). Cover, at minimum:

```python
# tests/projections/test_resolution_handlers.py
"""Resolution projection handlers: Cypher params + ordering guards (M4.5b)."""
# Test list (implement with the same fake-tx fixtures used in test_lens_handlers.py):
# - EntityCanonicalizedHandler passes canonical_id/name/entity_type/project_id/method/
#   confidence/surfaces to tx.run and raises ValueError when RETURN links < len(surfaces)
# - EntityCanonicalizedHandler passes when links == len(surfaces)
# - EntityAliasAddedHandler raises when links == 0 (entity not yet projected)
# - EntityMergeConfirmedHandler runs 2 queries; raises when query 1 RETURN found == 0;
#   does NOT raise when query 2 moves zero edges
# - EntitySplitHandler runs 2 queries; raises when query 1 RETURN created == 0
# - PersonIdentifiedHandler MERGEs person (no ordering guard needed — self-contained)
# - SpeakerLinkedToPersonHandler raises when links == 0 (speaker or person missing)
# - PersonLinkRemovedHandler raises when removed == 0
```

Write each as a real test function asserting (a) the exact query parameters passed to `tx.run` and (b) the raise/no-raise behavior driven by the faked record values.

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/projections/test_resolution_handlers.py -q --no-cov`
Expected: FAIL — module `src.projections.handlers.resolution_handlers` not found.

- [ ] **Step 3: Implement**

```python
# src/projections/handlers/resolution_handlers.py
"""Projection handlers for Project-stream resolution events (M4.5b).

Overlay nodes: (:CanonicalEntity), (:Person); edges ALIAS_OF, IDENTIFIED_AS.
Ordering guards raise on missing MATCH targets (cross-lane: Entity/Speaker
nodes are projected from the Sentence/Interview lanes) so the base handler
retries and eventually parks. Guards check explicit RETURN counts, not write
counters — edge MERGEs are no-ops on replay and would false-positive.
"""

from src.events.envelope import EventEnvelope
from src.projections.handlers.base_handler import BaseProjectionHandler
from src.utils.logger import get_logger

logger = get_logger()


class EntityCanonicalizedHandler(BaseProjectionHandler):
    """Create the canonical node and alias every surface entity to it."""

    async def apply(self, tx, event: EventEnvelope):
        d = event.data
        query = """
        MERGE (c:CanonicalEntity {canonical_id: $canonical_id})
        SET c.name = $name, c.entity_type = $entity_type,
            c.project_id = $project_id, c.method = $method,
            c.confidence = $confidence, c.locked = ($method = 'human')
        WITH c
        UNWIND $surfaces AS surface
        MATCH (e:Entity {surface: surface, entity_type: $entity_type})
        MERGE (e)-[a:ALIAS_OF {project_id: $project_id}]->(c)
        SET a.method = $method, a.confidence = $confidence
        RETURN count(a) AS links
        """
        result = await tx.run(
            query,
            canonical_id=d["canonical_id"], name=d["name"],
            entity_type=d["entity_type"], project_id=d["project_id"],
            method=d["method"], confidence=d["confidence"], surfaces=d["surfaces"],
        )
        record = await result.single()
        links = record["links"] if record else 0
        if links < len(d["surfaces"]):
            raise ValueError(
                f"EntityCanonicalized {d['canonical_id']}: only {links}/"
                f"{len(d['surfaces'])} surface entities projected yet"
            )


class EntityAliasAddedHandler(BaseProjectionHandler):
    """Attach one more surface entity to an existing canonical."""

    async def apply(self, tx, event: EventEnvelope):
        d = event.data
        query = """
        MATCH (c:CanonicalEntity {canonical_id: $canonical_id})
        MATCH (e:Entity {surface: $surface, entity_type: c.entity_type})
        MERGE (e)-[a:ALIAS_OF {project_id: $project_id}]->(c)
        SET a.method = $method, a.confidence = $confidence,
            c.locked = (c.locked OR $method = 'human')
        RETURN count(a) AS links
        """
        result = await tx.run(
            query,
            canonical_id=d["canonical_id"], surface=d["surface"],
            project_id=d["project_id"], method=d["method"], confidence=d["confidence"],
        )
        record = await result.single()
        if not record or record["links"] == 0:
            raise ValueError(
                f"EntityAliasAdded {d['canonical_id']}: canonical or entity "
                f"{d['surface']!r} not yet projected"
            )


class EntityMergeConfirmedHandler(BaseProjectionHandler):
    """Lock both canonicals, flag the merged one, move its ALIAS_OF edges."""

    async def apply(self, tx, event: EventEnvelope):
        d = event.data
        lock_query = """
        MATCH (surv:CanonicalEntity {canonical_id: $canonical_id})
        MATCH (merged:CanonicalEntity {canonical_id: $merged_canonical_id})
        SET surv.locked = true, merged.locked = true,
            merged.merged_into = $canonical_id
        RETURN count(*) AS found
        """
        result = await tx.run(
            lock_query,
            canonical_id=d["canonical_id"], merged_canonical_id=d["merged_canonical_id"],
        )
        record = await result.single()
        if not record or record["found"] == 0:
            raise ValueError(
                f"EntityMergeConfirmed {d['canonical_id']}<-{d['merged_canonical_id']}: "
                f"canonicals not yet projected"
            )
        move_query = """
        MATCH (e:Entity)-[a:ALIAS_OF {project_id: $project_id}]->
              (:CanonicalEntity {canonical_id: $merged_canonical_id})
        MATCH (surv:CanonicalEntity {canonical_id: $canonical_id})
        MERGE (e)-[na:ALIAS_OF {project_id: $project_id}]->(surv)
        SET na.method = a.method, na.confidence = a.confidence
        DELETE a
        """
        await tx.run(
            move_query,
            project_id=d["project_id"], canonical_id=d["canonical_id"],
            merged_canonical_id=d["merged_canonical_id"],
        )


class EntitySplitHandler(BaseProjectionHandler):
    """Create the split-off canonical and move the removed surfaces' edges."""

    async def apply(self, tx, event: EventEnvelope):
        d = event.data
        create_query = """
        MATCH (old:CanonicalEntity {canonical_id: $canonical_id})
        MERGE (new:CanonicalEntity {canonical_id: $new_canonical_id})
        SET new.name = $new_name, new.entity_type = old.entity_type,
            new.project_id = $project_id, new.method = 'human',
            new.confidence = 1.0, new.locked = true,
            old.locked = true
        RETURN count(new) AS created
        """
        result = await tx.run(
            create_query,
            canonical_id=d["canonical_id"], new_canonical_id=d["new_canonical_id"],
            new_name=d["new_name"], project_id=d["project_id"],
        )
        record = await result.single()
        if not record or record["created"] == 0:
            raise ValueError(
                f"EntitySplit {d['canonical_id']}: source canonical not yet projected"
            )
        move_query = """
        UNWIND $surfaces_removed AS surface
        MATCH (e:Entity {surface: surface})-[a:ALIAS_OF {project_id: $project_id}]->
              (:CanonicalEntity {canonical_id: $canonical_id})
        MATCH (new:CanonicalEntity {canonical_id: $new_canonical_id})
        MERGE (e)-[na:ALIAS_OF {project_id: $project_id}]->(new)
        SET na.method = 'human', na.confidence = 1.0
        DELETE a
        """
        await tx.run(
            move_query,
            surfaces_removed=d["surfaces_removed"], project_id=d["project_id"],
            canonical_id=d["canonical_id"], new_canonical_id=d["new_canonical_id"],
        )


class PersonIdentifiedHandler(BaseProjectionHandler):
    """Create/refresh the Person node (self-contained — no ordering guard)."""

    async def apply(self, tx, event: EventEnvelope):
        d = event.data
        query = """
        MERGE (p:Person {person_id: $person_id})
        SET p.display_name = $display_name, p.project_id = $project_id
        """
        await tx.run(
            query,
            person_id=d["person_id"], display_name=d["display_name"],
            project_id=d["project_id"],
        )


class SpeakerLinkedToPersonHandler(BaseProjectionHandler):
    """IDENTIFIED_AS edge from a Layer 1 speaker to a Person."""

    async def apply(self, tx, event: EventEnvelope):
        d = event.data
        query = """
        MATCH (sp:Speaker {speaker_id: $speaker_id})
        MATCH (p:Person {person_id: $person_id})
        MERGE (sp)-[r:IDENTIFIED_AS]->(p)
        SET r.method = $method, r.confidence = $confidence
        RETURN count(r) AS links
        """
        result = await tx.run(
            query,
            speaker_id=d["speaker_id"], person_id=d["person_id"],
            method=d["method"], confidence=d["confidence"],
        )
        record = await result.single()
        if not record or record["links"] == 0:
            raise ValueError(
                f"SpeakerLinkedToPerson {d['speaker_id']}->{d['person_id']}: "
                f"speaker or person not yet projected"
            )


class PersonLinkRemovedHandler(BaseProjectionHandler):
    """Delete the IDENTIFIED_AS edge (human correction)."""

    async def apply(self, tx, event: EventEnvelope):
        d = event.data
        query = """
        MATCH (sp:Speaker {speaker_id: $speaker_id})-[r:IDENTIFIED_AS]->
              (p:Person {person_id: $person_id})
        DELETE r
        RETURN count(r) AS removed
        """
        result = await tx.run(
            query, speaker_id=d["speaker_id"], person_id=d["person_id"],
        )
        record = await result.single()
        if not record or record["removed"] == 0:
            raise ValueError(
                f"PersonLinkRemoved {d['speaker_id']}->{d['person_id']}: "
                f"link not yet projected"
            )
```

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/projections/test_resolution_handlers.py -q --no-cov` → PASS; then `./scripts/test.sh tests/projections -q --no-cov` → no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/projections/handlers/resolution_handlers.py tests/projections/test_resolution_handlers.py
git commit -m "feat(resolution): projection handlers for CanonicalEntity/Person overlay"
```

---

### Task 5: Delivery wiring — bootstrap, subscription, lanes, pin tests

**Files:**
- Modify: `src/projections/bootstrap.py`, `src/projections/config.py`, `src/projections/lane_manager.py`
- Modify tests: `tests/projections/test_bootstrap_unit.py`, `tests/projections/test_lane_manager_unit.py` (or the existing lane-manager test file — locate with `grep -rl "_extract_interview_id\|route_event" tests/projections/`), `tests/projections/test_subscription_manager_unit.py` only if it pins real config keys.

**This is the M4.3 hard-won checklist — no Project event is delivered until ALL of these land together.**

- [ ] **Step 1: Write the failing pin tests first**

In `tests/projections/test_bootstrap_unit.py`:
- `test_create_handler_registry_registers_all_handlers`: expected count `22` → `29`; update the docstring arithmetic (`7 core + 5 speaker + 3 utterance + 4 enrichment + 3 lens + 7 resolution`).
- `test_create_handler_registry_logs_initialization_message`: `"22 handlers"` → `"29 handlers"`.
- `test_create_handler_registry_event_types_match_expected_names`: add the 7 new names to `expected_types`:
  `"EntityCanonicalized", "EntityAliasAdded", "EntityMergeConfirmed", "EntitySplit", "PersonIdentified", "SpeakerLinkedToPerson", "PersonLinkRemoved"`.
- Add a handler-class identity check mirroring the existing lens one:

```python
    @patch("src.projections.bootstrap.get_event_store_client")
    def test_resolution_handlers_registered(self, mock_get_client):
        """All 7 Project-stream events map to resolution handlers."""
        mock_get_client.return_value = MagicMock()
        registry = create_handler_registry()
        from src.projections.handlers.resolution_handlers import (
            EntityCanonicalizedHandler,
            PersonLinkRemovedHandler,
        )
        assert isinstance(registry.get_handler("EntityCanonicalized"), EntityCanonicalizedHandler)
        assert isinstance(registry.get_handler("PersonLinkRemoved"), PersonLinkRemovedHandler)
```

Add a config test (same file or `tests/projections/test_config_unit.py` if it exists):

```python
def test_project_subscription_allowlist():
    from src.projections.config import SUBSCRIPTION_CONFIG, is_event_allowed

    assert SUBSCRIPTION_CONFIG["project"]["stream"] == "$ce-Project"
    for event_type in (
        "EntityCanonicalized", "EntityAliasAdded", "EntityMergeConfirmed",
        "EntitySplit", "PersonIdentified", "SpeakerLinkedToPerson",
        "PersonLinkRemoved",
    ):
        assert is_event_allowed("project", event_type)
```

Add a lane-routing test next to the existing `_extract_interview_id` tests:

```python
def test_project_events_lane_key_is_aggregate_id():
    """Project events route by aggregate_id — one lane per project, never dropped."""
    manager = LaneManager(lane_count=4)
    event = MagicMock()
    event.aggregate_type = "Project"
    event.aggregate_id = "abc-123"
    assert manager._extract_interview_id(event) == "abc-123"
```

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/projections/test_bootstrap_unit.py -q --no-cov`
Expected: FAIL — count still 22, names missing, no "project" subscription, Project lane key returns None.

- [ ] **Step 3: Implement the wiring**

`src/projections/bootstrap.py` — import and register (after the lens block, before the `registered_types` line):

```python
from src.projections.handlers.resolution_handlers import (
    EntityAliasAddedHandler,
    EntityCanonicalizedHandler,
    EntityMergeConfirmedHandler,
    EntitySplitHandler,
    PersonIdentifiedHandler,
    PersonLinkRemovedHandler,
    SpeakerLinkedToPersonHandler,
)
```

```python
    # Register resolution handlers (Layer 4, M4.5b) — Project stream
    registry.register("EntityCanonicalized", EntityCanonicalizedHandler(parked_events_manager))
    registry.register("EntityAliasAdded", EntityAliasAddedHandler(parked_events_manager))
    registry.register("EntityMergeConfirmed", EntityMergeConfirmedHandler(parked_events_manager))
    registry.register("EntitySplit", EntitySplitHandler(parked_events_manager))
    registry.register("PersonIdentified", PersonIdentifiedHandler(parked_events_manager))
    registry.register("SpeakerLinkedToPerson", SpeakerLinkedToPersonHandler(parked_events_manager))
    registry.register("PersonLinkRemoved", PersonLinkRemovedHandler(parked_events_manager))
```

`src/projections/config.py` — add to `SUBSCRIPTION_CONFIG`:

```python
    "project": {
        "stream": "$ce-Project",  # Category stream for Project aggregate (M4.5b)
        "group": "neo4j-projection-project-v1",
        "allowlist": [
            "EntityCanonicalized",
            "EntityAliasAdded",
            "EntityMergeConfirmed",
            "EntitySplit",
            "PersonIdentified",
            "SpeakerLinkedToPerson",
            "PersonLinkRemoved",
        ],
    },
```

`src/projections/lane_manager.py` — in `_extract_interview_id`, add before the unknown-type warning:

```python
        if event.aggregate_type == "Project":
            # Project events serialize per project; lane key is the aggregate id.
            return event.aggregate_id
```

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/projections -q --no-cov` → all PASS. Then full `./scripts/test.sh` → green.

- [ ] **Step 5: Commit**

```bash
git add src/projections/bootstrap.py src/projections/config.py src/projections/lane_manager.py tests/projections/
git commit -m "feat(resolution): \$ce-Project subscription, handler registration, lane routing"
```

---

### Task 6: Resolution reader (the engine's own Neo4j input reads)

**Files:**
- Create: `src/resolution/__init__.py` (empty), `src/resolution/reader.py`
- Test: `tests/resolution/__init__.py` (empty), `tests/resolution/test_reader.py`

**Interfaces:**
- Produces: `entity_surface_rows(session, project_id) -> List[{surface, entity_type, mentions}]`, `speaker_rows(session, project_id) -> List[{interview_id, speaker_id, display_name, handle, provisional}]`.
- Note: Layer 4 owns these reads; `src/export/reader.py` stays Layer 5's consumer-Cypher home (spec).

- [ ] **Step 1: Write the failing tests**

Mirror the fake-session pattern used by `tests/export/` reader tests (a session stub whose `run` returns an async-iterable of records). Assert:
- `entity_surface_rows` sends the project-scoped MENTIONS query with `project_id` param and returns row dicts.
- `speaker_rows` filters `merged_into IS NULL` (assert the query text contains it) and returns row dicts with all 5 keys.

```python
# tests/resolution/test_reader.py
"""Resolution reader: project-scoped entity/speaker input rows (M4.5b)."""

import pytest

from src.resolution.reader import entity_surface_rows, speaker_rows


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        self._iter = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def run(self, query, **params):
        self.calls.append((query, params))
        return _FakeResult(self.rows)


@pytest.mark.asyncio
async def test_entity_surface_rows_scopes_by_project():
    session = _FakeSession([{"surface": "acme corp", "entity_type": "ORG", "mentions": 3}])
    rows = await entity_surface_rows(session, "p1")
    assert rows == [{"surface": "acme corp", "entity_type": "ORG", "mentions": 3}]
    query, params = session.calls[0]
    assert params == {"project_id": "p1"}
    assert "CONTAINS_INTERVIEW" in query and "MENTIONS" in query and ":Fragment" in query


@pytest.mark.asyncio
async def test_speaker_rows_excludes_merged():
    session = _FakeSession([{
        "interview_id": "i1", "speaker_id": "s1", "display_name": "Jane Doe",
        "handle": "S1", "provisional": False,
    }])
    rows = await speaker_rows(session, "p1")
    assert rows[0]["display_name"] == "Jane Doe"
    query, _ = session.calls[0]
    assert "merged_into IS NULL" in query and "HAS_PARTICIPANT" in query
```

If a record stub needs `dict(record)`, plain dicts already satisfy it — collect rows the same way `src/export/reader.py` does (check its idiom and copy it exactly).

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/resolution -q --no-cov` → FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/resolution/reader.py
"""Layer 4 input reads: the resolution engine's own project-scoped queries.

Layer 5's consumer Cypher stays in src/export/reader.py; these queries are
the engine's INPUT surface (spec M4.5b step 1).
"""

from typing import Any, Dict, List


async def entity_surface_rows(session, project_id: str) -> List[Dict[str, Any]]:
    """Distinct entity surfaces mentioned anywhere in the project, with counts."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(:Interview)
          -[:HAS_SENTENCE]->(:Fragment)-[m:MENTIONS]->(e:Entity)
    RETURN e.surface AS surface, e.entity_type AS entity_type, count(m) AS mentions
    ORDER BY entity_type, surface
    """
    result = await session.run(query, project_id=project_id)
    return [dict(record) async for record in result]


async def speaker_rows(session, project_id: str) -> List[Dict[str, Any]]:
    """Live (unmerged) speakers across the project's interviews."""
    query = """
    MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(i:Interview)
          -[:HAS_PARTICIPANT]->(sp:Speaker)
    WHERE sp.merged_into IS NULL
    RETURN i.interview_id AS interview_id, sp.speaker_id AS speaker_id,
           sp.display_name AS display_name, sp.handle AS handle,
           coalesce(sp.provisional, false) AS provisional
    ORDER BY interview_id, speaker_id
    """
    result = await session.run(query, project_id=project_id)
    return [dict(record) async for record in result]
```

(If the export reader's row-collection idiom differs from `[dict(record) async for record in result]`, use the export reader's idiom.)

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/resolution -q --no-cov` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/resolution/__init__.py src/resolution/reader.py tests/resolution/
git commit -m "feat(resolution): project-scoped entity/speaker input reader"
```

---

### Task 7: Candidate logic (pure functions)

**Files:**
- Create: `src/resolution/candidates.py`
- Test: `tests/resolution/test_candidates.py`

**Interfaces:**
- Produces: `normalize_surface(str) -> str`, `normalize_name(str) -> str`, `cosine(List[float], List[float]) -> float`, `exact_groups(rows) -> Dict[(norm, etype), List[row]]`, `representative(rows) -> str`, `UnionFind`, `embedding_pairs(keys, vectors, auto_thr, suggest_thr) -> (unions, suggestions)`, `person_groups(speaker_rows, participants_by_interview) -> (auto, suggestions)`.
- All pure — no IO, no async. The engine (Task 8) and on-demand suggestions (Task 10) both consume these.

- [ ] **Step 1: Write the failing tests (table-driven)**

```python
# tests/resolution/test_candidates.py
"""Deterministic candidate logic: normalization, grouping, thresholds (M4.5b)."""

import pytest

from src.resolution.candidates import (
    cosine,
    embedding_pairs,
    exact_groups,
    normalize_name,
    normalize_surface,
    person_groups,
    representative,
)


@pytest.mark.parametrize("raw,expected", [
    ("Acme Corp", "acme corp"),
    ("  The Acme   Corp ", "acme corp"),      # article + whitespace collapse
    ("acme's", "acme"),                        # possessive
    ("engineers'", "engineer"),                # trailing-apostrophe possessive + plural
    ("engineers", "engineer"),                 # naive plural fold
    ("bus", "bus"),                            # too short to deplural (len<=3)
    ("boss", "boss"),                          # 'ss' never folded
    ("An Apple", "apple"),
    ("A", "a"),                                # bare article is not stripped to empty
])
def test_normalize_surface(raw, expected):
    assert normalize_surface(raw) == expected


def test_normalize_name_casefold_and_collapse():
    assert normalize_name("  Jane   DOE ") == "jane doe"


def test_exact_groups_by_normalized_and_type():
    rows = [
        {"surface": "acme corp", "entity_type": "ORG", "mentions": 2},
        {"surface": "the acme corp", "entity_type": "ORG", "mentions": 1},
        {"surface": "acme corp", "entity_type": "PERSON", "mentions": 1},
    ]
    groups = exact_groups(rows)
    assert set(groups) == {("acme corp", "ORG"), ("acme corp", "PERSON")}
    assert len(groups[("acme corp", "ORG")]) == 2


def test_representative_prefers_mentions_then_lexicographic():
    rows = [
        {"surface": "acme", "entity_type": "ORG", "mentions": 1},
        {"surface": "acme corp", "entity_type": "ORG", "mentions": 3},
    ]
    assert representative(rows) == "acme corp"
    tied = [
        {"surface": "b", "entity_type": "ORG", "mentions": 2},
        {"surface": "a", "entity_type": "ORG", "mentions": 2},
    ]
    assert representative(tied) == "a"


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero vector guarded


def test_embedding_pairs_thresholds_and_type_isolation():
    keys = [("acme corp", "ORG"), ("acme inc", "ORG"), ("acme", "PERSON"), ("zeta", "ORG")]
    vectors = {
        ("acme corp", "ORG"): [1.0, 0.0],
        ("acme inc", "ORG"): [0.99, 0.14],   # cos ~0.990 -> auto
        ("acme", "PERSON"): [1.0, 0.0],       # different type -> never compared
        ("zeta", "ORG"): [0.62, 0.78],        # cos ~0.62 vs corp -> ignored
    }
    unions, suggestions = embedding_pairs(keys, vectors, auto_thr=0.92, suggest_thr=0.80)
    assert (("acme corp", "ORG"), ("acme inc", "ORG")) in unions
    assert all(a[1] == b[1] for a, b in unions)
    assert suggestions == []


def test_embedding_pairs_suggest_band():
    keys = [("a", "ORG"), ("b", "ORG")]
    vectors = {("a", "ORG"): [1.0, 0.0], ("b", "ORG"): [0.85, 0.53]}  # cos ~0.85
    unions, suggestions = embedding_pairs(keys, vectors, auto_thr=0.92, suggest_thr=0.80)
    assert unions == []
    assert suggestions == [{"key_a": ("a", "ORG"), "key_b": ("b", "ORG"),
                            "score": pytest.approx(0.849, abs=0.01)}]


def _speaker(iid, sid, display, handle="S1", provisional=False):
    return {"interview_id": iid, "speaker_id": sid, "display_name": display,
            "handle": handle, "provisional": provisional}


class TestPersonGroups:
    def test_exact_name_across_interviews_auto_links(self):
        auto, suggestions = person_groups(
            [_speaker("i1", "s1", "Jane Doe"), _speaker("i2", "s2", "jane doe")], {}
        )
        assert len(auto) == 1
        group = auto[0]
        assert group["person_key"] == "jane doe"
        assert group["method"] == "exact_name"
        assert sorted(group["links"]) == [("i1", "s1"), ("i2", "s2")]

    def test_front_matter_participant_single_speaker_auto_links(self):
        auto, _ = person_groups(
            [_speaker("i1", "s1", "Jane Doe")], {"i1": ["Jane Doe"]}
        )
        assert auto[0]["method"] == "front_matter"
        assert auto[0]["display_name"] == "Jane Doe"  # participant spelling wins

    def test_single_speaker_without_front_matter_not_auto(self):
        auto, _ = person_groups([_speaker("i1", "s1", "Jane Doe")], {})
        assert auto == []

    def test_provisional_and_handle_named_speakers_excluded(self):
        auto, _ = person_groups(
            [
                _speaker("i1", "s1", "S1", handle="S1"),           # unnamed
                _speaker("i2", "s2", "Jane Doe", provisional=True),
                _speaker("i3", "s3", "Jane Doe", provisional=True),
            ],
            {},
        )
        assert auto == []

    def test_first_name_only_becomes_suggestion(self):
        auto, suggestions = person_groups(
            [
                _speaker("i1", "s1", "Jane Doe"),
                _speaker("i2", "s2", "Jane Doe"),
                _speaker("i3", "s3", "Jane"),
            ],
            {},
        )
        assert len(auto) == 1
        assert len(suggestions) == 1
        s = suggestions[0]
        assert s["person_key"] == "jane doe"
        assert (s["interview_id"], s["speaker_id"]) == ("i3", "s3")
        assert s["reason"] == "first_name_match"
```

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/resolution/test_candidates.py -q --no-cov` → FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/resolution/candidates.py
"""Pure candidate logic for entity and person resolution (M4.5b).

Deterministic + review (spec): normalization and exact grouping auto-merge;
embedding cosine promotes near-duplicates at auto_merge_threshold and
surfaces a review band above suggest_threshold. No IO here — the engine and
the on-demand worklist suggestions both call these functions.
"""

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

Key = Tuple[str, str]  # (normalized_surface, entity_type)

_ARTICLES = ("the ", "a ", "an ")


def normalize_surface(surface: str) -> str:
    """casefold, collapse whitespace, strip leading article/possessive, naive plural-fold."""
    s = " ".join(surface.casefold().split())
    for article in _ARTICLES:
        if s.startswith(article):
            s = s[len(article):]
            break
    if s.endswith("'s"):
        s = s[:-2]
    elif s.endswith("s'"):
        s = s[:-1]
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s.strip()


def normalize_name(name: str) -> str:
    """casefold + whitespace collapse (person display names)."""
    return " ".join(name.casefold().split())


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def exact_groups(rows: Iterable[Dict[str, Any]]) -> Dict[Key, List[Dict[str, Any]]]:
    """Group surface rows by (normalized surface, entity_type)."""
    groups: Dict[Key, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (normalize_surface(row["surface"]), row["entity_type"])
        groups.setdefault(key, []).append(row)
    return groups


def representative(rows: List[Dict[str, Any]]) -> str:
    """The surface with the most mentions; ties break lexicographically."""
    return sorted(rows, key=lambda r: (-r.get("mentions", 0), r["surface"]))[0]["surface"]


class UnionFind:
    """Minimal union-find over hashable keys."""

    def __init__(self, keys: Iterable[Key]):
        self._parent = {k: k for k in keys}

    def find(self, key: Key) -> Key:
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def union(self, a: Key, b: Key) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    def clusters(self) -> List[List[Key]]:
        by_root: Dict[Key, List[Key]] = {}
        for key in self._parent:
            by_root.setdefault(self.find(key), []).append(key)
        return [sorted(members) for _, members in sorted(by_root.items())]


def embedding_pairs(
    keys: List[Key],
    vectors: Dict[Key, List[float]],
    auto_thr: float,
    suggest_thr: float,
) -> Tuple[List[Tuple[Key, Key]], List[Dict[str, Any]]]:
    """Pairwise cosine within an entity_type: >= auto_thr unions, [suggest, auto) suggests."""
    unions: List[Tuple[Key, Key]] = []
    suggestions: List[Dict[str, Any]] = []
    ordered = sorted(keys)
    for i, key_a in enumerate(ordered):
        for key_b in ordered[i + 1:]:
            if key_a[1] != key_b[1]:
                continue
            score = cosine(vectors[key_a], vectors[key_b])
            if score >= auto_thr:
                unions.append((key_a, key_b))
            elif score >= suggest_thr:
                suggestions.append({"key_a": key_a, "key_b": key_b, "score": score})
    return unions, suggestions


def _most_common(names: List[str]) -> str:
    counts = Counter(names)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def person_groups(
    speaker_rows: List[Dict[str, Any]],
    participants_by_interview: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Auto-link person groups + first-name-only suggestions.

    Only non-provisional speakers whose display_name differs from their handle
    participate — otherwise every unnamed "S1" across the project would
    collapse into one person. Auto when a normalized-name group spans >= 2
    speaker links OR matches a front-matter participant; the participant
    spelling wins the display name, else the most common raw display_name.
    """
    participant_pool: Dict[str, str] = {}
    for interview_id in sorted(participants_by_interview):
        for name in participants_by_interview[interview_id]:
            participant_pool.setdefault(normalize_name(name), name)

    groups: Dict[str, Dict[str, Any]] = {}
    for row in speaker_rows:
        display_name = (row.get("display_name") or "").strip()
        if not display_name or row.get("provisional") or display_name == row.get("handle"):
            continue
        key = normalize_name(display_name)
        group = groups.setdefault(key, {"links": [], "names": []})
        group["links"].append((row["interview_id"], row["speaker_id"]))
        group["names"].append(display_name)

    auto: List[Dict[str, Any]] = []
    auto_by_first_token: Dict[str, List[Dict[str, Any]]] = {}
    auto_keys = set()
    for key, group in sorted(groups.items()):
        in_pool = key in participant_pool
        if len(group["links"]) >= 2 or in_pool:
            entry = {
                "person_key": key,
                "display_name": participant_pool[key] if in_pool else _most_common(group["names"]),
                "method": "front_matter" if in_pool else "exact_name",
                "links": sorted(group["links"]),
            }
            auto.append(entry)
            auto_keys.add(key)
            auto_by_first_token.setdefault(key.split()[0], []).append(entry)

    suggestions: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        if key in auto_keys:
            continue
        for candidate in auto_by_first_token.get(key.split()[0], []):
            if candidate["person_key"] == key:
                continue
            for interview_id, speaker_id in sorted(group["links"]):
                suggestions.append({
                    "person_key": candidate["person_key"],
                    "display_name": candidate["display_name"],
                    "interview_id": interview_id,
                    "speaker_id": speaker_id,
                    "speaker_display_name": _most_common(group["names"]),
                    "reason": "first_name_match",
                })
    return auto, suggestions
```

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/resolution -q --no-cov` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/resolution/candidates.py tests/resolution/test_candidates.py
git commit -m "feat(resolution): pure candidate logic — normalization, thresholds, person groups"
```

---

### Task 8: ResolutionEngine + CLI

**Files:**
- Create: `src/resolution/engine.py`, `src/resolution/__main__.py`
- Test: `tests/resolution/test_engine.py`

**Interfaces:**
- Consumes: Tasks 1-3 (aggregate/repository/ids), Task 6 reader, Task 7 candidates, `get_embedder` (`src/enrichment/embedder.py`), `Neo4jConnectionManager` (`src/utils/neo4j_driver.py`), `get_interview_repository`.
- Produces: `ResolutionEngine(config_dict=None)` with `async apply(project_id, force=False) -> ResolutionResult`; patchable seam `_build_embedder()`; CLI `python -m src.resolution <project_id> [--force]`; actor SYSTEM `user_id="resolution"`.

**Engine algorithm (binding):**
1. Load `Project` aggregate (`project_aggregate_id(project_id)`); create fresh if absent.
2. One Neo4j session: `entity_surface_rows` + `speaker_rows`. Both empty → `ValueError` (unknown project → CLI error / 404 upstream).
3. Entity phase: `exact_groups` → embed each group's `representative` (ONE `embedder.embed` call for all reps, in-run cache is just that list) → `embedding_pairs` unions at `auto_merge_threshold` → union-find clusters → per cluster:
   - collect surfaces; find existing owners via `canonical_for_surface`;
   - **two distinct existing canonicals in one cluster → count a suggestion, touch nothing** (canonical-canonical merges are human-only `EntityMergeConfirmed`);
   - one existing owner: locked → count new surfaces as `skipped_locked`; unlocked → `add_entity_alias` for each new surface;
   - no owner: derive `cid = canonical_entity_id(project_id, normalize_surface(rep), entity_type)`; if `cid` already in state, alias onto it (locked → skip) — else `canonicalize_entity(cid, rep, etype, surfaces, "deterministic", 1.0)`.
4. Person phase: load each distinct interview aggregate for `metadata["front_matter"]["participants"]` (list of strings; missing → `[]`); `person_groups` → per auto group: `person_id_for(project_id, person_key)`; `identify_person` if new; per link: skip if `link_for_speaker` set, skip+count if pair in `blocked_links`, else `link_speaker_to_person(..., group["method"], 1.0)`.
5. Save the aggregate only if it has uncommitted events. Suggestions are counted in the result, never persisted (spec: the log stays clean).
6. `force` is accepted for CLI symmetry but currently a no-op: every run is a full recompute whose skips are state-driven; document this in the docstring and `--help`.

- [ ] **Step 1: Write the failing tests**

Patch seams: `src.resolution.engine.get_project_repository`, `get_interview_repository`, `Neo4jConnectionManager.get_session`, `entity_surface_rows`, `speaker_rows`, and `_build_embedder` (fake embedder returning fixed vectors per text). Mirror `tests/lens/test_engine.py` fixture style. Cover:

```python
# tests/resolution/test_engine.py — test list (write as real tests):
# - first run over two interviews with surfaces ["acme corp","the acme corp","zeta ltd"]
#   (exact-group merge) + fake vectors making "acme corp"~"acme inc" >= 0.92 (embedding union):
#   emits EntityCanonicalized per cluster with deterministic cids, correct surfaces/method;
#   speakers "Jane Doe" in both interviews -> PersonIdentified + 2 SpeakerLinkedToPerson
#   with method exact_name; repo.save called once; ResolutionResult counters match.
# - second run over identical inputs (aggregate state pre-loaded from run 1 events):
#   emits NOTHING (repo.save not called), counters all zero except suggestions.
# - locked canonical (state locked=True) + a new surface in its cluster:
#   no alias event, skipped_locked counted.
# - blocked pair in blocked_links: no link event, skipped counted.
# - two existing canonicals landing in one embedding cluster: no merge event,
#   entity_suggestions counted.
# - front-matter participant match: participants {"i1": ["Jane Doe"]} + single
#   speaker -> method front_matter.
# - unknown project (both reads empty) -> ValueError.
```

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/resolution/test_engine.py -q --no-cov` → FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/resolution/engine.py
"""Layer 4 resolution engine (M4.5b).

Mirrors LensEngine's shape: deterministic ids + aggregate-state checks make
re-runs idempotent; locked canonicals and blocked speaker pairs are skipped;
suggestions are computed and counted but never persisted as events.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from src.events.aggregates import Project
from src.events.envelope import Actor, ActorType, generate_correlation_id
from src.events.project_events import (
    canonical_entity_id,
    person_id_for,
    project_aggregate_id,
)
from src.events.repository import get_interview_repository, get_project_repository
from src.resolution.candidates import (
    embedding_pairs,
    exact_groups,
    normalize_surface,
    person_groups,
    representative,
    UnionFind,
)
from src.resolution.reader import entity_surface_rows, speaker_rows
from src.utils.logger import get_logger
from src.utils.neo4j_driver import Neo4jConnectionManager

logger = get_logger()


class ResolutionResult(BaseModel):
    """Summary of one resolution run."""

    project_id: str
    entities_canonicalized: int
    aliases_added: int
    entity_suggestions: int
    persons_identified: int
    speakers_linked: int
    person_suggestions: int
    skipped_locked: int
    skipped_blocked: int


class ResolutionEngine:
    """Resolves canonical entities and person identities for one project."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        from src.config import config as global_config

        self.config = config_dict if config_dict is not None else global_config

    def _build_embedder(self):
        """Construct the config-pinned embedder (patchable seam for tests)."""
        from src.enrichment.embedder import get_embedder

        return get_embedder(self.config)

    async def apply(self, project_id: str, force: bool = False) -> ResolutionResult:
        """Run entity + person resolution. force is reserved: every run is a
        full recompute whose skips are aggregate-state-driven, so there is
        nothing to supersede yet."""
        correlation_id = generate_correlation_id()
        actor = Actor(actor_type=ActorType.SYSTEM, user_id="resolution")

        repo = get_project_repository()
        aggregate_id = project_aggregate_id(project_id)
        project = await repo.load(aggregate_id)
        if project is None:
            project = Project(aggregate_id)

        async with await Neo4jConnectionManager.get_session() as session:
            entity_rows = await entity_surface_rows(session, project_id)
            speakers = await speaker_rows(session, project_id)
        if not entity_rows and not speakers:
            raise ValueError(f"Project {project_id} has nothing projected to resolve")

        (canonicalized, aliases_added, entity_suggestions,
         skipped_locked) = await self._resolve_entities(
            project, project_id, entity_rows, actor, correlation_id
        )
        (identified, linked, person_suggestions,
         skipped_blocked) = await self._resolve_persons(
            project, project_id, speakers, actor, correlation_id
        )

        if project.get_uncommitted_events():
            await repo.save(project)
        logger.info(
            f"Resolved project {project_id}: {canonicalized} canonicals, "
            f"{aliases_added} aliases, {identified} persons, {linked} links "
            f"({entity_suggestions + person_suggestions} suggestions, "
            f"{skipped_locked} locked, {skipped_blocked} blocked)"
        )
        return ResolutionResult(
            project_id=project_id,
            entities_canonicalized=canonicalized,
            aliases_added=aliases_added,
            entity_suggestions=entity_suggestions,
            persons_identified=identified,
            speakers_linked=linked,
            person_suggestions=person_suggestions,
            skipped_locked=skipped_locked,
            skipped_blocked=skipped_blocked,
        )

    async def _resolve_entities(
        self, project: Project, project_id: str, rows: List[Dict[str, Any]],
        actor: Actor, correlation_id: str,
    ) -> Tuple[int, int, int, int]:
        resolution_cfg = self.config.get("resolution", {})
        auto_thr = resolution_cfg.get("auto_merge_threshold", 0.92)
        suggest_thr = resolution_cfg.get("suggest_threshold", 0.80)

        canonicalized = aliases_added = suggestions = skipped_locked = 0
        groups = exact_groups(rows)
        keys = sorted(groups)
        unions = UnionFind(keys)
        if len(keys) > 1:
            reps = {key: representative(groups[key]) for key in keys}
            embedder = self._build_embedder()
            vectors = await embedder.embed([reps[key] for key in keys])
            by_key = dict(zip(keys, vectors))
            auto_unions, suggested_pairs = embedding_pairs(keys, by_key, auto_thr, suggest_thr)
            for key_a, key_b in auto_unions:
                unions.union(key_a, key_b)
            suggestions += len(suggested_pairs)

        for cluster in unions.clusters():
            cluster_rows = [row for key in cluster for row in groups[key]]
            entity_type = cluster[0][1]
            rep = representative(cluster_rows)
            surfaces = sorted({row["surface"] for row in cluster_rows})
            owners = {project.canonical_for_surface(s, entity_type) for s in surfaces}
            owners.discard(None)
            if len(owners) > 1:
                # two live canonicals in one cluster: merging is human-only
                suggestions += 1
                continue
            cid = owners.pop() if owners else canonical_entity_id(
                project_id, normalize_surface(rep), entity_type
            )
            entry = project.canonical_entities.get(cid)
            if entry is not None:
                new_surfaces = [s for s in surfaces if s not in entry["surfaces"]]
                if entry["locked"]:
                    skipped_locked += len(new_surfaces)
                    continue
                for surface in new_surfaces:
                    project.add_entity_alias(
                        project_id, cid, surface, "deterministic", 1.0,
                        actor=actor, correlation_id=correlation_id,
                    )
                    aliases_added += 1
                continue
            project.canonicalize_entity(
                project_id, cid, rep, entity_type, surfaces, "deterministic", 1.0,
                actor=actor, correlation_id=correlation_id,
            )
            canonicalized += 1
        return canonicalized, aliases_added, suggestions, skipped_locked

    async def _resolve_persons(
        self, project: Project, project_id: str, speakers: List[Dict[str, Any]],
        actor: Actor, correlation_id: str,
    ) -> Tuple[int, int, int, int]:
        interview_repo = get_interview_repository()
        participants: Dict[str, List[str]] = {}
        for interview_id in sorted({row["interview_id"] for row in speakers}):
            interview = await interview_repo.load(interview_id)
            front_matter = (interview.metadata.get("front_matter") or {}) if interview else {}
            raw = front_matter.get("participants") or []
            participants[interview_id] = [
                p.strip() for p in raw if isinstance(p, str) and p.strip()
            ]

        auto, suggestions = person_groups(speakers, participants)
        identified = linked = skipped_blocked = 0
        for group in auto:
            person_id = person_id_for(project_id, group["person_key"])
            if person_id not in project.persons:
                project.identify_person(
                    project_id, person_id, group["display_name"],
                    actor=actor, correlation_id=correlation_id,
                )
                identified += 1
            for interview_id, speaker_id in group["links"]:
                if project.link_for_speaker(interview_id, speaker_id) is not None:
                    continue
                if (interview_id, speaker_id) in project.blocked_links:
                    skipped_blocked += 1
                    continue
                project.link_speaker_to_person(
                    project_id, interview_id, speaker_id, person_id,
                    group["method"], 1.0, actor=actor, correlation_id=correlation_id,
                )
                linked += 1
        return identified, linked, len(suggestions), skipped_blocked
```

Note: `ResolutionResult` gains `skipped_blocked` (person pairs) distinct from `skipped_locked` (entity surfaces) — clearer than the spec's single counter; record the deviation in your report.

```python
# src/resolution/__main__.py
"""Resolve canonical entities and person identities for a project.

Usage: python -m src.resolution <project_id> [--force]
"""

import argparse
import asyncio

from src.resolution.engine import ResolutionEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Layer 4 resolution for a project")
    parser.add_argument("project_id")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reserved. Runs are full recomputes; skips are driven by aggregate "
            "state (locked canonicals, blocked speaker pairs), not run markers."
        ),
    )
    args = parser.parse_args()

    engine = ResolutionEngine()
    result = asyncio.run(engine.apply(args.project_id, force=args.force))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/resolution -q --no-cov` → PASS; full `./scripts/test.sh` → green.

- [ ] **Step 5: Commit**

```bash
git add src/resolution/engine.py src/resolution/__main__.py tests/resolution/test_engine.py
git commit -m "feat(resolution): ResolutionEngine with idempotent re-runs + python -m src.resolution CLI"
```

---

### Task 9: Corrections API

**Files:**
- Create: `src/api/routers/resolution.py`
- Modify: `src/main.py` (mount router)
- Test: `tests/api/test_resolution_router.py`

**Interfaces:**
- Consumes: `get_project_repository`, `project_aggregate_id`, `canonical_entity_id`, `normalize_surface`, Project domain methods (Task 2).
- Produces (spec paths, speakers-router pattern — 202 `{status, version}` / 404 / 409, actor HUMAN from `X-User-ID`):
  - `POST /resolution/{project_id}/entities/merge` `{surviving_canonical_id, merged_canonical_id}` → `EntityMergeConfirmed`
  - `POST /resolution/{project_id}/entities/{canonical_id}/split` `{surfaces: [...], new_name}` → `EntitySplit`
  - `POST /resolution/{project_id}/persons/{person_id}/link` `{interview_id, speaker_id, display_name?}` → `SpeakerLinkedToPerson` (creates the Person first via `PersonIdentified` when it doesn't exist AND `display_name` is provided; unknown person without `display_name` → 404)
  - `POST /resolution/{project_id}/persons/{person_id}/unlink` `{interview_id, speaker_id, note?}` → `PersonLinkRemoved`

- [ ] **Step 1: Write the failing tests**

Mirror `tests/api/test_speakers_router.py` (TestClient + patched `get_project_repository` by the ROUTER module's dotted path, i.e. `patch("src.api.routers.resolution.get_project_repository")`). Cover:

```python
# tests/api/test_resolution_router.py — test list (write as real tests):
# - merge happy path: 202 {status: "accepted", version}, repo.save called,
#   domain method received (project_id, surviving, merged) and HUMAN actor
#   with user_id from X-User-ID header
# - merge of unknown canonical -> 409 (domain ValueError)
# - merge when project stream missing (repo.load returns None) -> 404
# - split happy path: 202; new_canonical_id derived as
#   canonical_entity_id(project_id, normalize_surface(new_name), old entity_type)
# - split of unknown canonical_id in path -> 404 (not in aggregate state)
# - link happy path (person exists): 202, method "human", confidence 1.0
# - link with unknown person + display_name -> 202, identify_person called first
# - link with unknown person, no display_name -> 404
# - link blocked-pair via human -> 202 (human overrides block)
# - unlink happy path -> 202; unlink of non-linked pair -> 409
# - missing X-User-ID -> actor user_id "anonymous" (speakers-router precedent)
```

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/api/test_resolution_router.py -q --no-cov` → FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/api/routers/resolution.py
"""Human corrections for Layer 4 resolution (M4.5b).

Speakers-router pattern: load aggregate, call domain method, save; 202
{status, version} on accept, 404 unknown resource, 409 domain conflict.
"""

from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.events.envelope import Actor, ActorType
from src.events.project_events import canonical_entity_id, project_aggregate_id
from src.events.repository import get_project_repository
from src.resolution.candidates import normalize_surface

router = APIRouter(tags=["resolution"])


class MergeRequest(BaseModel):
    surviving_canonical_id: str
    merged_canonical_id: str


class SplitRequest(BaseModel):
    surfaces: List[str]
    new_name: str


class LinkRequest(BaseModel):
    interview_id: str
    speaker_id: str
    display_name: Optional[str] = None  # required only to create a new person


class UnlinkRequest(BaseModel):
    interview_id: str
    speaker_id: str
    note: Optional[str] = None


def _human_actor(x_user_id: Optional[str]) -> Actor:
    return Actor(actor_type=ActorType.HUMAN, user_id=x_user_id or "anonymous")


def _accepted(version: int) -> dict:
    return {"status": "accepted", "version": version}


async def _load_project(project_id: str):
    repo = get_project_repository()
    project = await repo.load(project_aggregate_id(project_id))
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} has no resolution state")
    return repo, project


@router.post("/resolution/{project_id}/entities/merge", status_code=202)
async def merge_entities(
    project_id: str,
    body: MergeRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    repo, project = await _load_project(project_id)
    try:
        project.confirm_entity_merge(
            project_id, body.surviving_canonical_id, body.merged_canonical_id,
            actor=_human_actor(x_user_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(project)
    return _accepted(project.version)


@router.post("/resolution/{project_id}/entities/{canonical_id}/split", status_code=202)
async def split_entity(
    project_id: str,
    canonical_id: str,
    body: SplitRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    repo, project = await _load_project(project_id)
    entry = project.canonical_entities.get(canonical_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Canonical entity {canonical_id} not found")
    new_canonical_id = canonical_entity_id(
        project_id, normalize_surface(body.new_name), entry["entity_type"]
    )
    try:
        project.split_entity(
            project_id, canonical_id, body.surfaces, new_canonical_id, body.new_name,
            actor=_human_actor(x_user_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(project)
    return _accepted(project.version)


@router.post("/resolution/{project_id}/persons/{person_id}/link", status_code=202)
async def link_speaker(
    project_id: str,
    person_id: str,
    body: LinkRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    repo, project = await _load_project(project_id)
    actor = _human_actor(x_user_id)
    try:
        if person_id not in project.persons:
            if body.display_name is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Person {person_id} not found (pass display_name to create)",
                )
            project.identify_person(project_id, person_id, body.display_name, actor=actor)
        project.link_speaker_to_person(
            project_id, body.interview_id, body.speaker_id, person_id, "human", 1.0,
            actor=actor,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(project)
    return _accepted(project.version)


@router.post("/resolution/{project_id}/persons/{person_id}/unlink", status_code=202)
async def unlink_speaker(
    project_id: str,
    person_id: str,
    body: UnlinkRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    repo, project = await _load_project(project_id)
    try:
        project.remove_person_link(
            project_id, body.interview_id, body.speaker_id, person_id, note=body.note,
            actor=_human_actor(x_user_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(project)
    return _accepted(project.version)
```

In `src/main.py`, mirror the existing includes:

```python
from src.api.routers import resolution as resolution_router
app.include_router(resolution_router.router)
```

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/api -q --no-cov` → PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add src/api/routers/resolution.py src/main.py tests/api/test_resolution_router.py
git commit -m "feat(resolution): corrections API — merge/split/link/unlink (202/404/409)"
```

---

### Task 10: Worklist suggestions + Person-grouped rollup

**Files:**
- Create: `src/resolution/suggestions.py`
- Modify: `src/api/routers/queries.py` (worklist endpoint), `src/export/reader.py` (`speaker_rollup_rows`, `_group_rollup_row`)
- Test: `tests/resolution/test_suggestions.py`, update `tests/api/test_queries_router.py` and the export reader rollup tests

**Interfaces:**
- Produces: `compute_suggestions(session, project, project_id, embedder, auto_thr=0.92, suggest_thr=0.80) -> {"entity_merge_suggestions": [...], "person_link_suggestions": [...]}` — suggestions computed on demand from the same candidate logic; NEVER persisted (spec).
- Rollup rows gain `"linked": bool` and `"person_id": Optional[str]`; linked speakers group by Person (person display_name wins), unlinked fall back to display-name grouping (spec).

- [ ] **Step 1: Write the failing tests**

`tests/resolution/test_suggestions.py` (reuse Task 6's fake session + a fake embedder):

```python
# Test list (write as real tests):
# - entity pair in the suggest band [0.80, 0.92) -> one entity_merge_suggestion row:
#   {surviving_canonical_id, merged_canonical_id, surfaces_a, surfaces_b, score}
#   with canonical ids DERIVED via canonical_entity_id (works before any engine run;
#   rows carry what POST /entities/merge needs)
# - pair where either side's canonical is locked in project state -> filtered out
# - project=None (engine never ran) -> no filtering, suggestions still computed
# - person first-name-only match -> person_link_suggestion row:
#   {person_id, display_name, interview_id, speaker_id, speaker_display_name, reason}
#   with person_id via person_id_for
# - blocked pair in project.blocked_links -> filtered out
# - already-linked pair (link_for_speaker set) -> filtered out
```

Worklist endpoint test (in `tests/api/test_queries_router.py`): with `project_id` query param present, response gains both suggestion keys (patch `compute_suggestions` at the queries-router dotted path); without `project_id`, both are `[]` and `compute_suggestions` is NOT called.

Rollup tests: linked speaker rows (fake records now carrying `person_id`/`person_name`) group under the person's display_name with `"linked": True`; unlinked keep today's behavior plus `"linked": False, "person_id": None`.

- [ ] **Step 2: Run to verify failure**

Run: `./scripts/test.sh tests/resolution/test_suggestions.py -q --no-cov` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/resolution/suggestions.py
"""On-demand resolution suggestions for the review worklist (M4.5b).

Same candidate logic as the engine, but nothing is emitted: the worklist
shows the [suggest, auto) band and first-name person matches, each row
carrying exactly what the corrections endpoints need. Canonical/person ids
are the deterministic uuid5 derivations, so rows are actionable even before
(or between) engine runs.
"""

from typing import Any, Dict, List, Optional

from src.events.aggregates import Project
from src.events.project_events import canonical_entity_id, person_id_for
from src.resolution.candidates import (
    embedding_pairs,
    exact_groups,
    person_groups,
    representative,
)
from src.resolution.reader import entity_surface_rows, speaker_rows


def _cid_for_key(project: Optional[Project], project_id: str, key, groups) -> Optional[str]:
    """Existing canonical for a group key, else the derived deterministic id.

    Returns None when the existing canonical is locked or merged away —
    the caller drops the suggestion.
    """
    normalized, entity_type = key
    surfaces = [row["surface"] for row in groups[key]]
    if project is not None:
        for surface in surfaces:
            cid = project.canonical_for_surface(surface, entity_type)
            if cid is not None:
                entry = project.canonical_entities[cid]
                if entry["locked"] or entry["merged_into"] is not None:
                    return None
                return cid
    return canonical_entity_id(project_id, normalized, entity_type)


async def compute_suggestions(
    session,
    project: Optional[Project],
    project_id: str,
    embedder,
    auto_thr: float = 0.92,
    suggest_thr: float = 0.80,
) -> Dict[str, List[Dict[str, Any]]]:
    entity_rows = await entity_surface_rows(session, project_id)
    speakers = await speaker_rows(session, project_id)

    entity_suggestions: List[Dict[str, Any]] = []
    groups = exact_groups(entity_rows)
    keys = sorted(groups)
    if len(keys) > 1:
        reps = {key: representative(groups[key]) for key in keys}
        vectors = await embedder.embed([reps[key] for key in keys])
        by_key = dict(zip(keys, vectors))
        _, band = embedding_pairs(keys, by_key, auto_thr, suggest_thr)
        for pair in band:
            cid_a = _cid_for_key(project, project_id, pair["key_a"], groups)
            cid_b = _cid_for_key(project, project_id, pair["key_b"], groups)
            if cid_a is None or cid_b is None or cid_a == cid_b:
                continue
            entity_suggestions.append({
                "surviving_canonical_id": cid_a,
                "merged_canonical_id": cid_b,
                "surfaces_a": sorted(row["surface"] for row in groups[pair["key_a"]]),
                "surfaces_b": sorted(row["surface"] for row in groups[pair["key_b"]]),
                "score": round(pair["score"], 4),
            })

    person_suggestions: List[Dict[str, Any]] = []
    _, candidates = person_groups(speakers, {})
    for suggestion in candidates:
        pair = (suggestion["interview_id"], suggestion["speaker_id"])
        if project is not None:
            if pair in project.blocked_links:
                continue
            if project.link_for_speaker(*pair) is not None:
                continue
        person_suggestions.append({
            "person_id": person_id_for(project_id, suggestion["person_key"]),
            "display_name": suggestion["display_name"],
            "interview_id": suggestion["interview_id"],
            "speaker_id": suggestion["speaker_id"],
            "speaker_display_name": suggestion["speaker_display_name"],
            "reason": suggestion["reason"],
        })

    return {
        "entity_merge_suggestions": entity_suggestions,
        "person_link_suggestions": person_suggestions,
    }
```

(Front-matter participants are skipped here deliberately — on-demand suggestions must not load every interview aggregate on a GET; the engine covers participant-based auto-links. Note this in the module docstring if the reviewer asks.)

In `src/api/routers/queries.py`, extend the worklist endpoint: when `project_id` is provided, after the existing `worklist_rows` call —

```python
    if project_id is not None:
        from src.config import config
        from src.enrichment.embedder import get_embedder
        from src.events.project_events import project_aggregate_id
        from src.events.repository import get_project_repository
        from src.resolution.suggestions import compute_suggestions

        project = await get_project_repository().load(project_aggregate_id(project_id))
        resolution_cfg = config.get("resolution", {})
        suggestions = await compute_suggestions(
            session, project, project_id, get_embedder(config),
            auto_thr=resolution_cfg.get("auto_merge_threshold", 0.92),
            suggest_thr=resolution_cfg.get("suggest_threshold", 0.80),
        )
    else:
        suggestions = {"entity_merge_suggestions": [], "person_link_suggestions": []}
    payload.update(suggestions)
```

(Adapt names to the endpoint's actual local variables; keep the session usage inside the existing `async with` block.)

In `src/export/reader.py`, rework the rollup for Person grouping — replace `_group_rollup_row` and the two queries' RETURNs:

```python
def _group_rollup_row(
    groups: Dict[str, Dict[str, Any]],
    display_name: str,
    person_id: Optional[str],
    person_name: Optional[str],
) -> Dict[str, Any]:
    key = person_id or f"name:{display_name.lower()}"
    return groups.setdefault(key, {
        "display_name": person_name or display_name,
        "linked": person_id is not None,
        "person_id": person_id,
        "items": [],
        "claims": [],
    })
```

Both rollup queries gain, before their RETURN:

```
    OPTIONAL MATCH (sp)-[:IDENTIFIED_AS]->(person:Person)
```

and add `person.person_id AS person_id, person.display_name AS person_name` to the RETURN clause. In the two row loops, pop the new fields and pass them through:

```python
        row = dict(r)
        display_name = row.pop("display_name")
        person_id = row.pop("person_id")
        person_name = row.pop("person_name")
        _group_rollup_row(groups, display_name, person_id, person_name)["items"].append(row)
```

(same for the claims loop). The final sort/pagination now sorts groups by their `display_name` value instead of the dict key:

```python
    ordered = sorted(groups.values(), key=lambda g: g["display_name"])
    if name is not None:
        needle = name.lower()
        ordered = [g for g in ordered if needle in g["display_name"].lower()]
    return ordered[offset:offset + limit]
```

- [ ] **Step 4: Run to verify pass**

Run: `./scripts/test.sh tests/resolution tests/api tests/export -q --no-cov` → PASS (including updated rollup/worklist tests).

- [ ] **Step 5: Commit**

```bash
git add src/resolution/suggestions.py src/api/routers/queries.py src/export/reader.py tests/
git commit -m "feat(resolution): on-demand worklist suggestions + Person-grouped speaker rollup"
```

---

### Task 11: OKF bundle — canonical entities + Person concept files

**Files:**
- Modify: `src/export/reader.py` (`entity_rows` gains canonical fields; new `person_rows`), `src/export/renderer.py` (`_render_entity` canonical grouping, new `_render_person`, speaker files link to persons, index gains Persons section), `src/export/bundler.py` (fetch + thread person/canonical data through `render_bundle`)
- Test: update `tests/export/test_renderer.py` / `test_reader.py` / bundler tests (whichever cover entity/speaker rendering today)

**Interfaces (spec):** `entities/<slug>.md` keys on CanonicalEntity when one exists — canonical name as title, aliases listed, mentions aggregated across its surface entities; surfaces with no canonical render per-surface exactly as today. Speaker files gain an `Identified as [<person>](/persons/<slug>.md)` reference when linked; new `persons/<slug>.md` concept file with `type: Person`.

- [ ] **Step 1: Write the failing tests**

```python
# Test list (place in the existing export test files, matching their fixtures):
# reader:
# - entity_rows now returns canonical_id / canonical_name (None when no ALIAS_OF);
#   query contains "OPTIONAL MATCH" and "ALIAS_OF"
# - person_rows(session, interview_id) returns
#   {speaker_id, person_id, display_name} via HAS_PARTICIPANT + IDENTIFIED_AS
# renderer:
# - two surface entities sharing canonical_id render ONE entities/<canonical-slug>.md
#   whose frontmatter has id: <canonical_id>, aliases: [<surface>, ...] and whose
#   mentions section aggregates both surfaces' mentions
# - a surface without canonical renders per-surface as before (regression)
# - _render_person produces persons/<slug>.md with type: Person, id, title
# - speaker file for a linked speaker contains a link to /persons/<slug>.md
# - index lists a Persons section when persons exist
```

- [ ] **Step 2: Run to verify failure** — `./scripts/test.sh tests/export -q --no-cov` → new tests FAIL.

- [ ] **Step 3: Implement**

`src/export/reader.py` — extend `entity_rows`'s query with (keep everything else identical):

```
    OPTIONAL MATCH (e)-[:ALIAS_OF]->(c:CanonicalEntity)
    WHERE c.merged_into IS NULL
```

adding `c.canonical_id AS canonical_id, c.name AS canonical_name` to the RETURN. Add:

```python
async def person_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    """Persons identified for this interview's speakers."""
    query = """
    MATCH (:Interview {interview_id: $interview_id})-[:HAS_PARTICIPANT]->
          (sp:Speaker)-[:IDENTIFIED_AS]->(p:Person)
    RETURN sp.speaker_id AS speaker_id, p.person_id AS person_id,
           p.display_name AS display_name
    ORDER BY speaker_id
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(record) async for record in result]
```

`src/export/renderer.py`:
- Group entity rows before rendering: rows sharing a non-null `canonical_id` render once via the canonical branch of `_render_entity` — title/slug from `canonical_name`, frontmatter `id: canonical_id`, an `aliases:` frontmatter list of the member surfaces, mentions concatenated in surface order. Null-canonical rows keep the existing per-surface path byte-for-byte (regression guard).
- Add `_render_person(person, references, registry)` returning `persons/<registry.slug_for(display_name)>.md` with frontmatter `{type: Person, title: display_name, id: person_id}` and a `## Speakers` section linking back to each linked speaker file.
- In `_render_speaker`, accept an optional `person` dict; when present append `Identified as [<name>](/persons/<slug>.md)` (route the label through `_link_text`, slug through the SAME registry the person file used).
- `render_bundle` / index: add persons to the file list and a `## Persons` index section mirroring the speakers section.

`src/export/bundler.py` — in `_write_bundle`'s read block add `persons = await reader.person_rows(session, interview_id)` and pass through to `render_bundle`.

Follow the existing signatures/threading exactly as `speakers` flows today; slugs must come from the single `_SlugRegistry` instance so collisions with entity/speaker slugs get suffixed.

- [ ] **Step 4: Run to verify pass** — `./scripts/test.sh tests/export tests/api -q --no-cov` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/export/ tests/export/
git commit -m "feat(export): OKF bundles key entities on canonicals; Person concept files"
```

---

### Task 12: Layer 4 integration smoke + docs + final verification

**Files:**
- Create: `tests/integration/test_layer4_resolution_smoke.py`
- Modify: `docs/architecture/database-schema.md`, `docs/ROADMAP.md`

- [ ] **Step 1: Write the smoke** (mirror `tests/integration/test_layer3_lens_smoke.py`'s infra fixtures/waits exactly — same env overrides, same projection-wait helper):

Flow, all against live docker infra via `./scripts/test-integration.sh`:
1. Unique `project_id` (uuid suffix). Ingest TWO small transcripts through `IngestionOrchestrator(project_id=...)`, each with front matter naming a shared participant (e.g. `participants: [Jane Doe]`) — Layer 1 seeds the speakers.
2. Emit `EntitiesExtracted` for a couple of fragments in each interview through the Fragment aggregate + repository (the Layer 2 smoke's canned pattern) with overlapping surfaces, e.g. `"Acme Corp"` in one and `"the Acme Corp"` in the other (exact-after-normalization merge — no embedding dependency in the smoke).
3. Wait for projection (entities + speakers present in Neo4j).
4. Run `ResolutionEngine` with `_build_embedder` patched to a fake embedder (deterministic orthogonal vectors — the smoke's auto-merges come from the exact group, keeping the assertion independent of embedding numerics).
5. Wait for Project-lane projection, then assert in Neo4j:
   - exactly one `(:CanonicalEntity {entity_type: 'ORG'})` for the acme cluster, with TWO `ALIAS_OF` edges from the two surface entities;
   - one `(:Person {display_name: 'Jane Doe'})` with `IDENTIFIED_AS` edges from both interviews' speakers (method `front_matter`);
   - `$ce-Project` delivery proven by the above (events flowed subscription → lane → handler).
6. Re-run the engine: `ResolutionResult` counters all zero (idempotence), and no duplicate nodes/edges in the graph.
7. **Dual-label invariant assertion (M4.5a backlog item — this closes it):**

```python
    query = """
    MATCH (n) WHERE (n:Sentence AND NOT n:Fragment) OR (n:Fragment AND NOT n:Sentence)
    RETURN count(n) AS mismatched
    """
    # assert mismatched == 0
```

- [ ] **Step 2: Run it live**

Infra: `docker ps` — if the test containers aren't up, `COMPOSE_PROJECT_NAME=interview_analyzer_chaining make test-infra-up`.
Run: `./scripts/test-integration.sh tests/integration/test_layer4_resolution_smoke.py -q --no-cov` → PASS.
Then all five smokes: `./scripts/test-integration.sh tests/integration/test_layer1_projection_smoke.py tests/integration/test_layer2_enrichment_smoke.py tests/integration/test_layer3_lens_smoke.py tests/integration/test_layer5_export_smoke.py tests/integration/test_layer4_resolution_smoke.py -q --no-cov` → 5 passed.

- [ ] **Step 3: Docs**

`docs/architecture/database-schema.md`: add `:CanonicalEntity` and `:Person` node sections (properties as in Task 4's docstring), `ALIAS_OF` / `IDENTIFIED_AS` relationship entries, and a Layer-4 note: canonical/person overlay never rewrites Layer 1/2 nodes; `Project-{id}` streams and the seven event names are wire format (frozen).
`docs/ROADMAP.md`: Quick Status `M4.5 | ⏳ In Progress | Layer 4: schema v2 (a ✅, b ✅, c: segments)`; add an M4.5b checklist section above M4.5a mirroring Tasks 1-12; Current Phase → "M4.5c (segments)". Mark the "dual-label invariant assertion" backlog entry (Deferred Backlog, M4.5a sub-list) as done via the Layer 4 smoke.

- [ ] **Step 4: Full gates**

`./scripts/test.sh` → green; flake8 clean on every file this plan touched.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_layer4_resolution_smoke.py docs/
git commit -m "test+docs: Layer 4 resolution smoke (incl. dual-label invariant); schema v2 docs"
```

---

## Verification (whole-plan)

1. Full unit suite green; all five integration smokes green on live infra.
2. Engine idempotence proven live (second run: zero events, zero new graph writes).
3. Wire-format audit: no existing event name / aggregate_type value / stream pattern changed; the seven NEW names + `"Project"` + `Project-{id}` are now frozen.
4. Delivery checklist audit: `grep -n "Project" src/projections/config.py src/projections/bootstrap.py src/projections/lane_manager.py` shows all three wired; bootstrap pin test at 29.
5. Corrections round-trip: merge or unlink via API → 202 → event → projection updates graph (covered by unit tests + handler tests; optional live spot-check).

## Deferred / out of scope (M4.5b)

- M4.5c (topic segments) — next plan from the same spec.
- LLM adjudication of borderline entity pairs (spec non-goal — deterministic + review only).
- Cross-project person linking (human-only, future; no endpoint yet).
- Embedding cache beyond a single run; suggestion pagination on the worklist.
- Front-matter participant matching inside on-demand suggestions (engine-only; a GET must not load every interview aggregate).
- `FragmentRepository = SentenceRepository` class alias + call-site/test patch-path flips (existing backlog, rides the alias drop).
