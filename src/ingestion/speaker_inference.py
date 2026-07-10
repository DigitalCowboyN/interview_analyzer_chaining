"""Speaker genesis: infer provisional speakers for unlabeled transcripts.

Windowed LLM pass proposes handles per fragment; overlapping windows are
reconciled deterministically by majority vote over the overlap region.
"""

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from src.agents.agent_factory import agent
from src.ingestion.models import RawFragment
from src.models.ingestion_responses import SpeakerWindowResponse
from src.utils.helpers import load_yaml
from src.utils.logger import get_logger

logger = get_logger()

PROMPTS_PATH = "prompts/ingestion_prompts.yaml"
UNKNOWN_HANDLE = "S?"


class FragmentSpeaker(BaseModel):
    """Resolved speaker assignment for one fragment."""

    sequence_order: int = Field(..., ge=0)
    handle: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class SpeakerInferenceResult(BaseModel):
    """Inference output: global handles and one assignment per fragment."""

    handles: List[str]
    assignments: List[FragmentSpeaker]


class SpeakerInferenceService:
    """Infers provisional speaker handles for unlabeled fragments."""

    def __init__(self, window_size: int = 40, overlap: int = 10):
        if overlap >= window_size:
            raise ValueError("overlap must be smaller than window_size")
        self.window_size = window_size
        self.overlap = overlap
        self.prompts = load_yaml(PROMPTS_PATH)

    async def infer(
        self, fragments: List[RawFragment], participants: Optional[List[str]] = None
    ) -> SpeakerInferenceResult:
        """Assign a provisional speaker handle to every fragment."""
        global_assignments: Dict[int, FragmentSpeaker] = {}
        handles: List[str] = []

        step = self.window_size - self.overlap
        for window_start in range(0, len(fragments), step):
            window = fragments[window_start:window_start + self.window_size]
            if not window:
                break
            window_result = await self._infer_window(window, participants)

            mapping = self._reconcile(window, window_result, global_assignments)

            for local_index, (handle, confidence) in window_result.items():
                seq = window[local_index].sequence_order
                if seq in global_assignments:
                    continue  # overlap region: keep earlier window's assignment
                global_handle = mapping.get(handle, handle)
                if global_handle not in handles:
                    handles.append(global_handle)
                global_assignments[seq] = FragmentSpeaker(
                    sequence_order=seq, handle=global_handle, confidence=confidence
                )

            if window_start + self.window_size >= len(fragments):
                break

        # Guarantee one assignment per fragment; LLM omissions get UNKNOWN at 0.0.
        assignments = []
        for frag in fragments:
            if frag.sequence_order not in global_assignments:
                logger.warning(
                    f"No speaker assignment for fragment {frag.sequence_order}; marking unknown"
                )
                if UNKNOWN_HANDLE not in handles:
                    handles.append(UNKNOWN_HANDLE)
                global_assignments[frag.sequence_order] = FragmentSpeaker(
                    sequence_order=frag.sequence_order, handle=UNKNOWN_HANDLE, confidence=0.0
                )
            assignments.append(global_assignments[frag.sequence_order])

        return SpeakerInferenceResult(handles=handles, assignments=assignments)

    async def _infer_window(
        self, window: List[RawFragment], participants: Optional[List[str]] = None
    ) -> Dict[int, Tuple[str, float]]:
        """Run one window through the LLM; returns {local_index: (handle, confidence)}."""
        numbered = "\n".join(f"{i}: {frag.text}" for i, frag in enumerate(window))
        hint = f"Known participants: {', '.join(participants)}.\n" if participants else ""
        prompt = self.prompts["speaker_window"]["prompt"].format(
            fragments=numbered, participants_hint=hint
        )
        raw = await agent.call_model(prompt)
        try:
            response = SpeakerWindowResponse.model_validate(raw)
        except ValidationError as e:
            logger.warning(f"Invalid speaker window response, skipping window: {e}")
            return {}
        return {
            a.index: (a.speaker, a.confidence)
            for a in response.assignments
            if 0 <= a.index < len(window)
        }

    def _reconcile(
        self,
        window: List[RawFragment],
        window_result: Dict[int, Tuple[str, float]],
        global_assignments: Dict[int, FragmentSpeaker],
    ) -> Dict[str, str]:
        """Map window-local handles to global handles by overlap majority vote."""
        votes: Dict[str, Dict[str, int]] = {}
        for local_index, (handle, _confidence) in window_result.items():
            seq = window[local_index].sequence_order
            existing = global_assignments.get(seq)
            if existing is None:
                continue
            votes.setdefault(handle, {}).setdefault(existing.handle, 0)
            votes[handle][existing.handle] += 1

        return {
            local_handle: max(counts, key=counts.get)
            for local_handle, counts in votes.items()
        }
