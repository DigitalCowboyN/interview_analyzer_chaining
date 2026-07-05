"""Layer 1 ingestion orchestrator.

Flow: read -> normalize -> speakers (parse or infer) -> fragment events ->
stitch overlay -> upgraded map file. All writes go through the event-sourced
repositories; Neo4j is populated by the projection service downstream.
"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.events.aggregates import Interview, Sentence
from src.events.envelope import Actor, ActorType, generate_correlation_id
from src.events.repository import get_interview_repository, get_sentence_repository
from src.ingestion.models import NormalizedTranscript, TranscriptFormat
from src.ingestion.normalizer import normalize
from src.ingestion.speaker_inference import (
    FragmentSpeaker,
    SpeakerInferenceResult,
    SpeakerInferenceService,
)
from src.ingestion.stitcher import Stitcher, StitchResult
from src.utils.helpers import append_json_line
from src.utils.logger import get_logger

logger = get_logger()


class IngestionResult(BaseModel):
    """Summary of one ingested transcript."""

    interview_id: str
    fragment_count: int
    speaker_count: int
    utterance_count: int
    interruption_count: int
    low_confidence_count: int


class IngestionOrchestrator:
    """Runs the full Layer 1 ingestion flow for one file."""

    def __init__(
        self,
        project_id: str,
        map_dir: Path,
        confidence_review_threshold: float = 0.7,
    ):
        self.project_id = project_id
        self.map_dir = Path(map_dir)
        self.threshold = confidence_review_threshold
        self.inference = SpeakerInferenceService()
        self.stitcher = Stitcher()

    async def ingest_file(
        self, file_path: Path, correlation_id: Optional[str] = None
    ) -> IngestionResult:
        """Ingest one transcript file end to end."""
        correlation_id = correlation_id or generate_correlation_id()
        actor = Actor(actor_type=ActorType.SYSTEM, user_id="ingestion")
        file_path = Path(file_path)
        text = file_path.read_text(encoding="utf-8")
        transcript = normalize(text)
        interview_id = str(uuid.uuid4())

        interview = Interview(interview_id)
        interview.create(
            title=file_path.name,
            source=str(file_path),
            metadata={
                "content_hash": transcript.content_hash,
                "format": transcript.format.value,
                "fragment_count": len(transcript.fragments),
            },
            actor=actor,
            correlation_id=correlation_id,
            project_id=self.project_id,
        )

        assignments = await self._resolve_speakers(
            interview, transcript, actor, correlation_id
        )
        interview_repo = get_interview_repository()
        await interview_repo.save(interview)

        speaker_ids = {info["handle"]: sid for sid, info in interview.speakers.items()}
        fragment_uuids = await self._emit_fragments(
            interview_id, transcript, assignments, speaker_ids, actor, correlation_id
        )

        stitch = await self._stitch(transcript, assignments)
        utterance_ids = self._emit_stitch(
            interview, stitch, speaker_ids, fragment_uuids, actor, correlation_id
        )
        await interview_repo.save(interview)

        self._write_map(
            file_path, transcript, assignments, speaker_ids, fragment_uuids,
            stitch, utterance_ids,
        )

        low_confidence = sum(1 for a in assignments if a.confidence < self.threshold)
        logger.info(
            f"Ingested {file_path.name}: interview={interview_id}, "
            f"{len(transcript.fragments)} fragments, {len(speaker_ids)} speakers, "
            f"{len(stitch.utterances)} utterances, {low_confidence} low-confidence attributions"
        )
        return IngestionResult(
            interview_id=interview_id,
            fragment_count=len(transcript.fragments),
            speaker_count=len(speaker_ids),
            utterance_count=len(stitch.utterances),
            interruption_count=len(stitch.interruptions),
            low_confidence_count=low_confidence,
        )

    async def _resolve_speakers(
        self,
        interview: Interview,
        transcript: NormalizedTranscript,
        actor: Actor,
        correlation_id: str,
    ) -> List[FragmentSpeaker]:
        """Create speakers (parsed or inferred) and return per-fragment assignments."""
        if transcript.format == TranscriptFormat.LABELED:
            handles = list(transcript.speaker_labels)
            assignments = [
                FragmentSpeaker(
                    sequence_order=f.sequence_order,
                    handle=f.speaker_label or "S?",
                    confidence=1.0 if f.speaker_label else 0.0,
                )
                for f in transcript.fragments
            ]
            if any(f.speaker_label is None for f in transcript.fragments):
                handles.append("S?")
            provisional, method = False, "parsed"
            confidences = {h: 1.0 for h in handles}
        else:
            result: SpeakerInferenceResult = await self.inference.infer(transcript.fragments)
            handles = result.handles
            assignments = result.assignments
            confidences = {
                h: (
                    sum(a.confidence for a in assignments if a.handle == h)
                    / max(1, sum(1 for a in assignments if a.handle == h))
                )
                for h in handles
            }
            provisional, method = True, "inference"

        for handle in handles:
            speaker_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview.aggregate_id}:speaker:{handle}")
            )
            # The unknown-speaker placeholder is always a guess, regardless of
            # how the rest of the transcript's speakers were resolved.
            is_unknown = handle == "S?"
            interview.add_speaker(
                speaker_id,
                handle=handle,
                display_name=handle,
                provisional=True if is_unknown else provisional,
                confidence=0.0 if is_unknown else round(confidences[handle], 4),
                method="inference" if is_unknown else method,
                actor=actor,
                correlation_id=correlation_id,
            )
        return assignments

    async def _emit_fragments(
        self,
        interview_id: str,
        transcript: NormalizedTranscript,
        assignments: List[FragmentSpeaker],
        speaker_ids: Dict[str, str],
        actor: Actor,
        correlation_id: str,
    ) -> Dict[int, str]:
        """Create Sentence aggregates with offsets and speaker attribution."""
        sentence_repo = get_sentence_repository()
        assignment_by_seq = {a.sequence_order: a for a in assignments}
        fragment_uuids: Dict[int, str] = {}
        for frag in transcript.fragments:
            sentence_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{frag.sequence_order}")
            )
            fragment_uuids[frag.sequence_order] = sentence_id
            sentence = Sentence(sentence_id)
            sentence.create(
                interview_id=interview_id,
                index=frag.sequence_order,
                text=frag.text,
                start_char=frag.start_char,
                end_char=frag.end_char,
                actor=actor,
                correlation_id=correlation_id,
            )
            assignment = assignment_by_seq[frag.sequence_order]
            speaker_id = speaker_ids.get(assignment.handle)
            if speaker_id:
                sentence.attribute_speaker(
                    speaker_id,
                    confidence=assignment.confidence,
                    method="parsed" if transcript.format == TranscriptFormat.LABELED else "inference",
                    actor=actor,
                    correlation_id=correlation_id,
                )
            await sentence_repo.save(sentence)
        return fragment_uuids

    async def _stitch(
        self, transcript: NormalizedTranscript, assignments: List[FragmentSpeaker]
    ) -> StitchResult:
        """LABELED transcripts get the structural baseline; FLAT gets LLM refinement."""
        if transcript.format == TranscriptFormat.LABELED:
            return self.stitcher.baseline(transcript.fragments, assignments)
        return await self.stitcher.stitch(transcript.fragments, assignments)

    def _emit_stitch(
        self,
        interview: Interview,
        stitch: StitchResult,
        speaker_ids: Dict[str, str],
        fragment_uuids: Dict[int, str],
        actor: Actor,
        correlation_id: str,
    ) -> Dict[int, str]:
        """Emit UtteranceIdentified and InterruptionRecorded events."""
        utterance_ids: Dict[int, str] = {}
        for utt in stitch.utterances:
            speaker_id = speaker_ids.get(utt.handle)
            if speaker_id is None:
                logger.warning(
                    f"Skipping utterance {utt.ordinal}: no speaker registered for "
                    f"handle {utt.handle!r}"
                )
                continue
            utterance_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS, f"{interview.aggregate_id}:utterance:{utt.ordinal}"
                )
            )
            utterance_ids[utt.ordinal] = utterance_id
            interview.identify_utterance(
                utterance_id,
                speaker_id=speaker_id,
                fragment_ids=[fragment_uuids[s] for s in utt.sequence_orders],
                confidence=utt.confidence,
                actor=actor,
                correlation_id=correlation_id,
            )
        for intr in stitch.interruptions:
            if (
                intr.interrupting_ordinal not in utterance_ids
                or intr.interrupted_ordinal not in utterance_ids
            ):
                logger.warning(f"Skipping interruption referencing skipped utterance: {intr}")
                continue
            interview.record_interruption(
                interrupting_utterance_id=utterance_ids[intr.interrupting_ordinal],
                interrupted_utterance_id=utterance_ids[intr.interrupted_ordinal],
                at_fragment_id=fragment_uuids[intr.at_sequence_order],
                actor=actor,
                correlation_id=correlation_id,
            )
        return utterance_ids

    def _write_map(
        self,
        file_path: Path,
        transcript: NormalizedTranscript,
        assignments: List[FragmentSpeaker],
        speaker_ids: Dict[str, str],
        fragment_uuids: Dict[int, str],
        stitch: StitchResult,
        utterance_ids: Dict[int, str],
    ) -> None:
        """Write the upgraded map file: the pipeline's coordinate system."""
        self.map_dir.mkdir(parents=True, exist_ok=True)
        map_path = self.map_dir / f"{file_path.stem}_map.jsonl"
        if map_path.exists():
            map_path.unlink()
        assignment_by_seq = {a.sequence_order: a for a in assignments}
        utterance_by_seq = {
            seq: utterance_ids[u.ordinal]
            for u in stitch.utterances
            for seq in u.sequence_orders
        }
        for frag in transcript.fragments:
            a = assignment_by_seq[frag.sequence_order]
            append_json_line(
                {
                    "sentence_id": fragment_uuids[frag.sequence_order],
                    "sequence_order": frag.sequence_order,
                    "sentence": frag.text,
                    "start_char": frag.start_char,
                    "end_char": frag.end_char,
                    "speaker_id": speaker_ids.get(a.handle),
                    "speaker_confidence": a.confidence,
                    "utterance_id": utterance_by_seq.get(frag.sequence_order),
                },
                map_path,
            )
