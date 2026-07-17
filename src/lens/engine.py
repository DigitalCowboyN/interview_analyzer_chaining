"""Layer 3 lens engine.

Loads a lens profile and a Layer 1 interview, runs the lens's extractors
(ordinary ExtractorSpecs, one focused call per unit of their scope) via the
M4.2 executor, resolves extracted speaker references against Layer 1 speakers,
and emits the three generic lens events on the Interview stream.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from src.enrichment.executor import EnrichmentExecutor
from src.events.aggregates import Fragment, Interview
from src.events.envelope import Actor, ActorType, generate_correlation_id
from src.events.repository import get_interview_repository, get_fragment_repository
from src.lens.models import LensExtractorDecl, LensSpec, load_lens
from src.utils.logger import get_logger

logger = get_logger()


class LensResult(BaseModel):
    """Summary of one lens application."""

    interview_id: str
    lens: str
    lens_version: int
    items_extracted: int
    items_skipped_existing: int
    items_skipped_locked: int
    units_processed: int


class LensEngine:
    """Applies a lens to an interview and emits lens events."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        from src.config import config as global_config

        self.config = config_dict if config_dict is not None else global_config

    def _build_executor(self, lens: LensSpec) -> EnrichmentExecutor:
        """Construct the executor (patchable seam for tests)."""
        from src.agents.failover_agent import get_failover_agent
        from src.utils.helpers import load_yaml

        agent = get_failover_agent(self.config)
        specs = [d.to_extractor_spec() for d in lens.extractors]
        prompts = load_yaml(lens.prompts_file)
        concurrency = self.config.get("pipeline", {}).get("num_analysis_workers", 10)
        return EnrichmentExecutor(agent, specs, prompts, domain_keywords=[], concurrency=concurrency)

    async def apply(
        self,
        interview_id: str,
        lens_name: str,
        force: bool = False,
        correlation_id: Optional[str] = None,
    ) -> LensResult:
        """Run every lens extractor over its units and record the items."""
        correlation_id = correlation_id or generate_correlation_id()
        actor = Actor(actor_type=ActorType.SYSTEM, user_id="lens")
        lens = load_lens(lens_name)

        interview_repo = get_interview_repository()
        interview = await interview_repo.load(interview_id)
        if interview is None:
            raise ValueError(f"Interview {interview_id} not found")

        ordered = await self._load_fragments(interview)
        utterance_texts = self._utterance_texts(interview, ordered)
        document_text = self._document_text(interview, ordered)

        # Same-version re-run without force skips the LensApplied emit — the
        # run is idempotent and must not supersede its own items.
        if force or interview.lens_runs.get(lens.name) != lens.version:
            interview.apply_lens(
                lens.name, lens.version, actor=actor, correlation_id=correlation_id
            )

        executor = self._build_executor(lens)
        extracted = skipped_existing = skipped_locked = units = 0

        for decl in lens.extractors:
            spec = decl.to_extractor_spec()
            for source_unit, text, unit_speaker_id, supporting_ids in self._source_units(
                decl, interview, ordered, utterance_texts, document_text
            ):
                units += 1
                outcome = await executor.run_spec_on_text(spec, text)
                if outcome.data is None:
                    logger.warning(f"{decl.name} on {source_unit}: {outcome.flags}")
                    continue
                for ordinal, item in enumerate(outcome.data.get(decl.items_field, [])):
                    item_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{interview_id}:lens:{lens.name}:{decl.node_type}"
                            f":{source_unit}:{ordinal}",
                        )
                    )
                    existing = interview.lens_items.get(item_id)
                    if existing is not None:
                        if existing["locked"]:
                            skipped_locked += 1
                        else:
                            skipped_existing += 1
                        continue
                    fields = {k: v for k, v in item.items() if k != "confidence"}
                    speaker_links = self._speaker_links(
                        lens, decl, fields, unit_speaker_id, interview
                    )
                    interview.record_lens_extraction(
                        lens=lens.name,
                        lens_version=lens.version,
                        node_type=decl.node_type,
                        item_id=item_id,
                        fields=fields,
                        supporting_fragment_ids=supporting_ids,
                        speaker_links=speaker_links,
                        confidence=item.get("confidence", 0.0),
                        model=outcome.model,
                        provider=outcome.provider,
                        actor=actor,
                        correlation_id=correlation_id,
                    )
                    extracted += 1

        await interview_repo.save(interview)
        logger.info(
            f"Applied lens {lens.name} v{lens.version} to {interview_id}: "
            f"{extracted} items ({skipped_existing} existing, {skipped_locked} locked, "
            f"{units} units)"
        )
        return LensResult(
            interview_id=interview_id,
            lens=lens.name,
            lens_version=lens.version,
            items_extracted=extracted,
            items_skipped_existing=skipped_existing,
            items_skipped_locked=skipped_locked,
            units_processed=units,
        )

    async def _load_fragments(self, interview: Interview) -> List[Fragment]:
        """Load fragments by deterministic uuid5 id, in index order."""
        fragment_repo = get_fragment_repository()
        fragment_count = interview.metadata.get("fragment_count", 0)
        ordered: List[Fragment] = []
        for index in range(fragment_count):
            sid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview.aggregate_id}:{index}"))
            sentence = await fragment_repo.load(sid)
            if sentence is not None:
                ordered.append(sentence)
        ordered.sort(key=lambda s: s.index)
        return ordered

    def _utterance_texts(
        self, interview: Interview, ordered: List[Fragment]
    ) -> Dict[str, str]:
        """Join member-fragment texts (in fragment_ids order) per utterance."""
        by_aggregate = {s.aggregate_id: s for s in ordered}
        texts: Dict[str, str] = {}
        for uid, u in interview.utterances.items():
            if u.get("removed"):
                continue
            parts = [by_aggregate[fid].text for fid in u["fragment_ids"] if fid in by_aggregate]
            texts[uid] = " ".join(parts)
        return texts

    def _document_text(self, interview: Interview, ordered: List[Fragment]) -> str:
        """Full transcript, speaker-labeled '[S1]: text' lines in index order."""
        lines = []
        for s in ordered:
            handle = interview.speakers.get(s.speaker_id, {}).get("handle", "S?")
            lines.append(f"[{handle}]: {s.text}")
        return "\n".join(lines)

    def _source_units(
        self,
        decl: LensExtractorDecl,
        interview: Interview,
        ordered: List[Fragment],
        utterance_texts: Dict[str, str],
        document_text: str,
    ) -> List[Tuple[str, str, Optional[str], List[str]]]:
        """(source_unit_key, text, unit_speaker_id, supporting_fragment_ids) per unit."""
        if decl.scope == "document":
            return [("document", document_text, None, [])]
        if decl.scope == "utterance":
            return [
                (uid, text, interview.utterances[uid]["speaker_id"],
                 list(interview.utterances[uid]["fragment_ids"]))
                for uid, text in utterance_texts.items()
            ]
        return [(s.aggregate_id, s.text, s.speaker_id, [s.aggregate_id]) for s in ordered]

    def _speaker_links(
        self,
        lens: LensSpec,
        decl: LensExtractorDecl,
        fields: Dict[str, Any],
        unit_speaker_id: Optional[str],
        interview: Interview,
    ) -> List[Dict[str, str]]:
        """Resolve the mapping's declared speaker field into a link, if possible.

        Unresolved references keep the raw string in fields plus an
        `<field>_unresolved` marker; no link is emitted.
        """
        mapping = lens.projects_to[decl.node_type]
        if not mapping.speaker_link:
            return []
        field = mapping.speaker_link["field"]
        value = fields.get(field)
        if value is None:
            return []
        speaker_id = self._resolve_speaker(value, unit_speaker_id, interview)
        if speaker_id is None:
            fields[f"{field}_unresolved"] = True
            return []
        return [{"relationship": mapping.speaker_link["relationship"], "speaker_id": speaker_id}]

    def _resolve_speaker(
        self, value: str, unit_speaker_id: Optional[str], interview: Interview
    ) -> Optional[str]:
        """Match an extracted reference against Layer 1 speakers.

        The literal 'SELF' resolves to the source unit's speaker; otherwise
        handle or display_name, case-insensitive. Merged speakers are skipped.
        """
        if value == "SELF":
            return unit_speaker_id
        needle = value.strip().lower()
        for speaker_id, info in interview.speakers.items():
            if info.get("merged_into"):
                continue
            if needle in (info["handle"].lower(), info["display_name"].lower()):
                return speaker_id
        return None
