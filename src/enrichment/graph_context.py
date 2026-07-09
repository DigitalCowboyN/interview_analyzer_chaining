"""Speaker- and utterance-aware context building for enrichment extractors.

Replaces the legacy ContextBuilder's bare-sentence windows with lines rendered
as `[S1]: text` from the Layer 1 graph, plus the stitched utterance a fragment
belongs to.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FragmentView(BaseModel):
    """Enrichment-side view of one fragment."""

    index: int = Field(..., ge=0)
    text: str
    speaker_handle: str
    utterance_id: Optional[str] = None


class GraphContextBuilder:
    """Builds context strings with speaker labels and utterance awareness."""

    def __init__(self, context_windows: Dict[str, int]):
        self.context_windows = context_windows

    def build_all(
        self, fragments: List[FragmentView], utterance_texts: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """One context dict per fragment: configured windows + utterance_context."""
        contexts: List[Dict[str, str]] = []
        for frag in fragments:
            ctx: Dict[str, str] = {}
            for name, window in self.context_windows.items():
                ctx[name] = self._window(fragments, frag.index, window)
            if frag.utterance_id and frag.utterance_id in utterance_texts:
                ctx["utterance_context"] = utterance_texts[frag.utterance_id]
            else:
                ctx["utterance_context"] = frag.text
            contexts.append(ctx)
        return contexts

    @staticmethod
    def _window(fragments: List[FragmentView], idx: int, window: int) -> str:
        start = max(0, idx - window)
        end = min(len(fragments), idx + window + 1)
        lines = []
        for frag in fragments[start:end]:
            line = f"[{frag.speaker_handle}]: {frag.text}"
            if frag.index == idx:
                line = f">>> {line} <<<"
            lines.append(line)
        return "\n".join(lines)
