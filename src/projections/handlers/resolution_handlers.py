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
