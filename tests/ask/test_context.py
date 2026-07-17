"""Context blocks, prompt rendering, code-attached citations."""

from src.ask.context import build_blocks, quotes_for, render_prompt

ROW = {
    "fragment_id": "f1", "text": "We chose Acme.", "sequence_order": 2,
    "interview_id": "i1", "title": "Kickoff", "speaker": "S1 Alice",
    "person": "Alice Smith", "segment_topics": ["Vendor choice"],
    "entities": ["acme"], "siblings": [
        {"text": "After review,", "order": 1}, {"text": "for the pilot.", "order": 3},
    ],
}


def test_build_blocks_merges_utterance_in_order_and_keeps_verbatim_quote():
    block = build_blocks([ROW])[0]
    assert block["quote"] == "We chose Acme."           # verbatim fragment text
    assert block["utterance_text"] == "After review, We chose Acme. for the pilot."
    assert block["fragment_id"] == "f1"
    assert block["speaker_line"] == "S1 Alice (Alice Smith)"
    assert block["segment_topics"] == ["Vendor choice"]
    assert block["title"] == "Kickoff"


def test_build_blocks_speaker_line_without_person():
    row = dict(ROW, person=None)
    assert build_blocks([row])[0]["speaker_line"] == "S1 Alice"


def test_build_blocks_carries_title_none_when_absent():
    row = dict(ROW, title=None)
    assert build_blocks([row])[0]["title"] is None


def test_render_prompt_carries_question_and_fragment_ids():
    rendered = render_prompt(
        "Q: {question}\nCTX:\n{context}", "why acme?", build_blocks([ROW])
    )
    assert "why acme?" in rendered
    assert "[f1]" in rendered and "We chose Acme." in rendered
    assert "— Kickoff" in rendered


def test_render_prompt_omits_title_dash_when_title_is_none():
    row = dict(ROW, title=None)
    rendered = render_prompt(
        "Q: {question}\nCTX:\n{context}", "why acme?", build_blocks([row])
    )
    assert "—" not in rendered


def test_quotes_for_drops_unknown_ids_and_preserves_block_order():
    blocks = build_blocks([ROW])
    quotes = quotes_for(["made-up", "f1"], blocks)
    assert quotes == [{"fragment_id": "f1", "interview_id": "i1",
                       "quote": "We chose Acme."}]


def test_real_prompt_asset_renders_through_render_prompt():
    from src.utils.helpers import load_yaml

    template = load_yaml("prompts/ask_prompts.yaml")["ask_synthesis"]["prompt"]
    rendered = render_prompt(template, "why acme?", build_blocks([ROW]))
    assert "why acme?" in rendered
    assert '{"answer":' in rendered  # JSON literal survived formatting
    assert "[f1]" in rendered
