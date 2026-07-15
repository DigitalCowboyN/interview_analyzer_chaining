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
