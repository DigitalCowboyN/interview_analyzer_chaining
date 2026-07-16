"""Context-block assembly and synthesis-prompt rendering (pure functions)."""

from typing import Any, Dict, List


def build_blocks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One grounded block per retrieved fragment, in reader (transcript) order."""
    blocks = []
    for row in rows:
        pieces = [{"text": row["text"], "order": row["sequence_order"]}] + [
            s for s in row["siblings"] if s.get("text") is not None
        ]
        utterance_text = " ".join(
            p["text"] for p in sorted(pieces, key=lambda p: p["order"])
        )
        speaker_line = row.get("speaker") or "Unknown speaker"
        if row.get("person"):
            speaker_line = f"{speaker_line} ({row['person']})"
        blocks.append(
            {
                "fragment_id": row["fragment_id"],
                "interview_id": row["interview_id"],
                "quote": row["text"],
                "speaker_line": speaker_line,
                "utterance_text": utterance_text,
                "segment_topics": [t for t in row["segment_topics"] if t],
                "entities": [e for e in row["entities"] if e],
            }
        )
    return blocks


def _block_lines(block: Dict[str, Any]) -> str:
    topics = ", ".join(block["segment_topics"]) or "-"
    entities = ", ".join(block["entities"]) or "-"
    return (
        f"[{block['fragment_id']}] {block['speaker_line']} "
        f"(interview {block['interview_id']}; topics: {topics}; entities: {entities})\n"
        f"{block['utterance_text']}"
    )


def render_prompt(template: str, question: str, blocks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(_block_lines(b) for b in blocks)
    return template.format(question=question, context=context)


def quotes_for(
    citation_ids: List[str], blocks: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Attach verbatim quotes by code — the LLM supplies ids only."""
    cited = set(citation_ids)
    return [
        {"fragment_id": b["fragment_id"], "interview_id": b["interview_id"],
         "quote": b["quote"]}
        for b in blocks
        if b["fragment_id"] in cited
    ]
