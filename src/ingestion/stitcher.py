"""Stitching pass: overlay utterance structure on the immutable fragment sequence.

Baseline: consecutive same-speaker fragments form utterances. An LLM pass may
merge non-adjacent same-speaker utterances (continuation across interruption)
and report interruptions. The fragment sequence itself is never modified.
"""

from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

from src.agents.agent_factory import agent
from src.ingestion.models import RawFragment
from src.ingestion.speaker_inference import FragmentSpeaker
from src.models.ingestion_responses import StitchWindowResponse
from src.utils.helpers import load_yaml
from src.utils.logger import get_logger

logger = get_logger()

PROMPTS_PATH = "prompts/ingestion_prompts.yaml"


class StitchedUtterance(BaseModel):
    """A speaker's continuous thought spanning one or more fragments."""

    ordinal: int = Field(..., ge=0, description="Stable ordinal within the interview")
    handle: str
    sequence_orders: List[int] = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class Interruption(BaseModel):
    """One utterance breaking into another."""

    interrupting_ordinal: int
    interrupted_ordinal: int
    at_sequence_order: int


class StitchResult(BaseModel):
    """Utterance overlay for an interview."""

    utterances: List[StitchedUtterance]
    interruptions: List[Interruption]


class Stitcher:
    """Builds the utterance overlay: baseline grouping refined by an LLM pass."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.prompts = load_yaml(PROMPTS_PATH)

    def baseline(
        self, fragments: List[RawFragment], assignments: List[FragmentSpeaker]
    ) -> StitchResult:
        """Baseline grouping only (no LLM): consecutive same-speaker fragments."""
        handle_by_seq = {a.sequence_order: a.handle for a in assignments}
        groups = self._baseline_groups(fragments, handle_by_seq)
        utterances = [
            StitchedUtterance(ordinal=i, handle=h, sequence_orders=seqs, confidence=c)
            for i, (h, seqs, c) in enumerate(groups)
        ]
        return StitchResult(utterances=utterances, interruptions=[])

    async def stitch(
        self, fragments: List[RawFragment], assignments: List[FragmentSpeaker]
    ) -> StitchResult:
        """Produce the utterance overlay for the given attributed fragments."""
        handle_by_seq = {a.sequence_order: a.handle for a in assignments}
        groups = self._baseline_groups(fragments, handle_by_seq)

        merged_groups, interruptions = await self._refine_with_llm(
            fragments, handle_by_seq, groups
        )

        utterances = [
            StitchedUtterance(
                ordinal=i, handle=handle, sequence_orders=seqs, confidence=confidence
            )
            for i, (handle, seqs, confidence) in enumerate(merged_groups)
        ]
        return StitchResult(utterances=utterances, interruptions=interruptions)

    def _baseline_groups(
        self, fragments: List[RawFragment], handle_by_seq: Dict[int, str]
    ) -> List[Tuple[str, List[int], float]]:
        """Group consecutive same-speaker fragments: [(handle, [seq, ...], confidence)].

        Fragments are sorted by sequence_order first, so callers may pass them
        in any order.
        """
        groups: List[Tuple[str, List[int], float]] = []
        for frag in sorted(fragments, key=lambda f: f.sequence_order):
            handle = handle_by_seq.get(frag.sequence_order, "S?")
            if groups and groups[-1][0] == handle:
                groups[-1][1].append(frag.sequence_order)
            else:
                groups.append((handle, [frag.sequence_order], 1.0))
        return groups

    async def _refine_with_llm(
        self,
        fragments: List[RawFragment],
        handle_by_seq: Dict[int, str],
        groups: List[Tuple[str, List[int], float]],
    ) -> Tuple[List[Tuple[str, List[int], float]], List[Interruption]]:
        """Ask the LLM for cross-interruption merges and interruption edges."""
        if len(fragments) > self.window_size * 10:
            logger.warning(
                f"Stitch refinement prompt spans {len(fragments)} fragments "
                f"(> {self.window_size * 10}); prompt size is unbounded"
            )
        numbered = "\n".join(
            f"{f.sequence_order}: [{handle_by_seq.get(f.sequence_order, 'S?')}] {f.text}"
            for f in fragments
        )
        prompt = self.prompts["stitch_window"]["prompt"].format(fragments=numbered)

        try:
            raw = await agent.call_model(prompt)
            response = StitchWindowResponse.model_validate(raw)
        except Exception as e:  # LLM failure or invalid shape -> baseline only
            logger.warning(f"Stitch pass failed or invalid; using baseline grouping: {e}")
            return groups, []

        valid_seqs = set(handle_by_seq) & {f.sequence_order for f in fragments}
        merged: List[Tuple[str, List[int], float]] = []
        # Track which merged entry each LLM proposal produced, so interruption
        # ordinals (which index the proposal list) can be remapped after the
        # merged list is re-sorted by first sequence order.
        entry_by_proposal: Dict[int, Tuple[str, List[int], float]] = {}
        claimed: set = set()
        for prop_index, proposal in enumerate(response.utterances):
            seqs = proposal.fragment_indices
            if (
                seqs != sorted(seqs)
                or not set(seqs).issubset(valid_seqs)
                or len({handle_by_seq.get(s) for s in seqs}) != 1
                or handle_by_seq.get(seqs[0]) != proposal.speaker
            ):
                logger.warning(f"Dropping invalid utterance proposal: {proposal}")
                continue
            if claimed & set(seqs):
                # Proposals must be disjoint: a fragment in two utterances would
                # project conflicting PART_OF_UTTERANCE memberships.
                logger.warning(f"Dropping overlapping utterance proposal: {proposal}")
                continue
            claimed.update(seqs)
            entry = (proposal.speaker, seqs, proposal.confidence)
            entry_by_proposal[prop_index] = entry
            merged.append(entry)

        # Fragments not covered by valid proposals keep their baseline groups.
        covered = {s for _, seqs, _ in merged for s in seqs}
        for handle, seqs, confidence in groups:
            remaining = [s for s in seqs if s not in covered]
            if remaining:
                merged.append((handle, remaining, confidence))
        merged.sort(key=lambda g: g[1][0])
        position_of = {id(entry): pos for pos, entry in enumerate(merged)}

        interruptions = []
        for prop in response.interruptions:
            interrupting = entry_by_proposal.get(prop.interrupting)
            interrupted = entry_by_proposal.get(prop.interrupted)
            if interrupting is None or interrupted is None:
                logger.warning(f"Dropping interruption referencing dropped/unknown proposal: {prop}")
                continue
            # The interruption point must lie within the interrupted utterance's
            # span, else the proposal is incoherent.
            interrupted_seqs = interrupted[1]
            if not (interrupted_seqs[0] <= prop.at_index <= interrupted_seqs[-1]):
                logger.warning(f"Dropping interruption outside interrupted span: {prop}")
                continue
            interruptions.append(
                Interruption(
                    interrupting_ordinal=position_of[id(interrupting)],
                    interrupted_ordinal=position_of[id(interrupted)],
                    at_sequence_order=prop.at_index,
                )
            )
        return merged, interruptions
