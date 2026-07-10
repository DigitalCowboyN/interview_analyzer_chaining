import re

import pytest
import yaml

from src.export.renderer import item_dir, item_filename, render_bundle, slugify
from src.lens.models import load_lens

HEADER = {
    "interview_id": "iid-1", "title": "Q3 Vendor Selection", "source": "m.txt",
    "started_at": "2026-07-01T00:00:00", "project_id": "telemetry",
    "metadata": {"front_matter": {"participants": ["Alice Johnson"]}},
    "participants": [{"handle": "Alice", "display_name": "Alice Johnson", "provisional": False}],
    "fragment_count": 2, "utterance_count": 1,
    "lens": "meeting_minutes", "lens_version": 1,
}
TRANSCRIPT = [
    {"sentence_id": "f1", "sequence_order": 0, "text": "Let's go with X.",
     "speaker_id": "sp1", "speaker": "Alice Johnson", "utterance_id": "u-abc"},
    {"sentence_id": "f2", "sequence_order": 1, "text": "I'll draft the doc.",
     "speaker_id": "sp1", "speaker": "Alice Johnson", "utterance_id": "u-abc"},
]
SPEAKERS = [{"speaker_id": "sp1", "handle": "Alice", "display_name": "Alice Johnson", "provisional": False}]
ITEMS = [{
    "item_id": "8888888812345678", "node_type": "Decision", "lens_version": 1,
    "confidence": 0.9, "model": "haiku", "provider": "anthropic", "locked": False,
    "props": {"item_id": "8888888812345678", "lens": "meeting_minutes", "node_type": "Decision",
              "text": "Go with X", "made_by": "Alice", "confidence": 0.9},
    "speaker_links": [{"relationship": "DECIDED_BY", "speaker_id": "sp1", "display_name": "Alice Johnson"}],
    "supporting_fragment_ids": ["f1"],
}]
CLAIMS = [{"claim_id": "c1", "text": "We will ship", "kind": "commitment", "confidence": 0.8,
           "model": "haiku", "provider": "anthropic", "speaker_id": "sp1",
           "speaker": "Alice Johnson", "supporting_fragment_ids": ["f2"]}]
ENTITIES = [{"surface": "ecu", "entity_type": "product",
             "mentions": [{"sentence_id": "f1", "start": 0, "end": 3, "text": "ECU", "confidence": 0.9}]}]
ANALYSIS = [{"sequence_order": 0, "text": "Let's go with X.", "speaker": "Alice Johnson",
             "function": "declarative", "structure": "simple", "purpose": "Statement",
             "topics": ["vendors"], "keywords": ["telemetry"], "confidence": 0.9, "flags": None}]


def render():
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    return files


def test_naming_helpers():
    assert slugify("Alice Johnson") == "alice-johnson"
    assert item_dir("ActionItem") == "action-items"
    assert item_filename("ActionItem", "8888888812345678") == "action-item-88888888.md"


def test_every_nonreserved_file_is_okf_conformant():
    files = render()
    assert "index.md" in files and "interview.md" in files
    for path, content in files.items():
        if path == "index.md":
            assert not content.startswith("---"), "index.md is reserved: no frontmatter"
            continue
        assert content.startswith("---\n"), f"{path} missing frontmatter"
        fm = yaml.safe_load(content.split("---\n")[1])
        assert fm.get("type"), f"{path} missing required type"


def test_lens_item_file_content():
    files = render()
    content = files["decisions/decision-88888888.md"]
    fm = yaml.safe_load(content.split("---\n")[1])
    assert fm["type"] == "Decision" and fm["lens"] == "meeting_minutes"
    assert fm["made_by"] == "Alice"                      # extracted field passes through
    assert "item_id: 8888888812345678" in content        # id preserved (as `id`)
    assert "DECIDED_BY: [Alice Johnson](/speakers/alice-johnson.md)" in content
    assert "> Let's go with X." in content               # verbatim grounding quote
    assert "(/transcript.md#u-1)" in content             # bundle-absolute anchor link


def test_transcript_anchors_and_index_links():
    files = render()
    assert '<a id="u-1"></a>' in files["transcript.md"]
    assert "](decisions/decision-88888888.md)" in files["index.md"]  # relative link
    assert "claims/claim-c1.md" in files  # claim file named by id prefix rule
    assert "entities/ecu.md" in files


def test_index_covers_interview_transcript_and_analysis():
    files = render()
    index = files["index.md"]
    assert "](interview.md)" in index
    assert "](transcript.md)" in index
    assert "](analysis.md)" in index


def test_frontmatter_survives_embedded_delimiter_line():
    lens = load_lens("meeting_minutes")
    tricky_text = "line one\n---\nline two"
    items = [{
        "item_id": "8888888812345678", "node_type": "Decision", "lens_version": 1,
        "confidence": 0.9, "model": "haiku", "provider": "anthropic", "locked": False,
        "props": {"item_id": "8888888812345678", "lens": "meeting_minutes", "node_type": "Decision",
                  "text": tricky_text, "made_by": "Alice", "confidence": 0.9},
        "speaker_links": [{"relationship": "DECIDED_BY", "speaker_id": "sp1", "display_name": "Alice Johnson"}],
        "supporting_fragment_ids": ["f1"],
    }]
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, items, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    content = files["decisions/decision-88888888.md"]
    fm = yaml.safe_load(content.split("---\n")[1])
    assert fm["description"] == tricky_text


@pytest.mark.parametrize(
    "tricky_text",
    [
        "---\nleading dash line",     # bare --- as the first line
        "starts with ---\nrest",      # --- at start of first line's content
        "Some decision text\n---",    # bare --- as the last line
    ],
    ids=["leading-line", "start-of-line", "trailing-line"],
)
def test_frontmatter_survives_delimiter_at_scalar_boundary(tricky_text):
    """Line-anchored detection missed cases where '---' shares a physical line
    with a folded/quoted scalar's opening or closing quote. These are the
    reproduced bypass variants; the substring-based check must catch all."""
    lens = load_lens("meeting_minutes")
    items = [{
        "item_id": "8888888812345678", "node_type": "Decision", "lens_version": 1,
        "confidence": 0.9, "model": "haiku", "provider": "anthropic", "locked": False,
        "props": {"item_id": "8888888812345678", "lens": "meeting_minutes", "node_type": "Decision",
                  "text": tricky_text, "made_by": "Alice", "confidence": 0.9},
        "speaker_links": [{"relationship": "DECIDED_BY", "speaker_id": "sp1", "display_name": "Alice Johnson"}],
        "supporting_fragment_ids": ["f1"],
    }]
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, items, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    content = files["decisions/decision-88888888.md"]
    fm = yaml.safe_load(content.split("---\n")[1])
    assert fm["description"] == tricky_text


def test_coerce_scalar_only_applies_to_id_field():
    lens = load_lens("meeting_minutes")
    items = [{
        "item_id": "8888888812345678", "node_type": "Decision", "lens_version": 1,
        "confidence": 0.9, "model": "haiku", "provider": "anthropic", "locked": False,
        "props": {"item_id": "8888888812345678", "lens": "meeting_minutes", "node_type": "Decision",
                  "text": "Go with X", "ref_code": "12345", "confidence": 0.9},
        "speaker_links": [{"relationship": "DECIDED_BY", "speaker_id": "sp1", "display_name": "Alice Johnson"}],
        "supporting_fragment_ids": ["f1"],
    }]
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, items, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    content = files["decisions/decision-88888888.md"]
    fm = yaml.safe_load(content.split("---\n")[1])
    assert fm["ref_code"] == "12345"
    assert isinstance(fm["ref_code"], str)
    assert 'ref_code: "12345"' in content or "ref_code: '12345'" in content


def test_extracted_field_colliding_with_okf_key_does_not_shadow_it():
    """An extracted lens field named e.g. `type` must not silently overwrite
    the OKF-reserved frontmatter key; it survives as `field_type` instead."""
    lens = load_lens("meeting_minutes")
    items = [{
        "item_id": "8888888812345678", "node_type": "Decision", "lens_version": 1,
        "confidence": 0.9, "model": "haiku", "provider": "anthropic", "locked": False,
        "props": {"item_id": "8888888812345678", "lens": "meeting_minutes", "node_type": "Decision",
                  "text": "Go with X", "type": "bogus", "confidence": 0.9},
        "speaker_links": [{"relationship": "DECIDED_BY", "speaker_id": "sp1", "display_name": "Alice Johnson"}],
        "supporting_fragment_ids": ["f1"],
    }]
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, items, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    content = files["decisions/decision-88888888.md"]
    fm = yaml.safe_load(content.split("---\n")[1])
    assert fm["type"] == "Decision"        # OKF key untouched
    assert fm["field_type"] == "bogus"     # extracted value preserved under prefixed key


def test_relationship_line_skips_falsy_relationship_or_display():
    lens = load_lens("meeting_minutes")
    items = [{
        "item_id": "8888888812345678", "node_type": "Decision", "lens_version": 1,
        "confidence": 0.9, "model": "haiku", "provider": "anthropic", "locked": False,
        "props": {"item_id": "8888888812345678", "lens": "meeting_minutes", "node_type": "Decision",
                  "text": "Go with X", "confidence": 0.9},
        "speaker_links": [
            {"relationship": None, "speaker_id": "sp1", "display_name": "Alice Johnson"},
            {"relationship": "DECIDED_BY", "speaker_id": "sp2", "display_name": None},
        ],
        "supporting_fragment_ids": ["f1"],
    }]
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, items, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    content = files["decisions/decision-88888888.md"]
    assert "None:" not in content
    assert "[None]" not in content


def test_link_text_escapes_and_collapses():
    from src.export.renderer import _link_text

    assert _link_text("line one\nline two") == "line one line two"
    assert _link_text("a [bracketed] thing") == "a \\[bracketed\\] thing"
    assert len(_link_text("x" * 200)) == 80


def test_table_cells_escape_pipes_and_newlines():
    import copy

    analysis = copy.deepcopy(ANALYSIS)
    analysis[0]["text"] = "cell | with pipe\nand newline"
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               ENTITIES, analysis, lens, exported_at="2026-07-10T12:00:00+00:00"))
    table_lines = [ln for ln in files["analysis.md"].splitlines() if "cell" in ln]
    assert table_lines, "analysis row missing"
    assert "\\|" in table_lines[0] and "\n" not in table_lines[0]


def test_index_link_labels_survive_hostile_item_text():
    import copy

    items = copy.deepcopy(ITEMS)
    items[0]["props"]["text"] = "Decide [now]\nor never | maybe"
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, items, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    index = files["index.md"]
    # the link label must not contain raw newlines or unescaped brackets
    label_line = next(ln for ln in index.splitlines() if "decision-" in ln)
    assert "\n" not in label_line and "[now]" not in label_line


def test_link_text_truncates_before_escaping_so_escapes_never_sever():
    """Escape-then-truncate can cut a `\\[`/`\\]` pair in half, leaving a lone
    trailing backslash that escapes the link's closing `]` and breaks the
    markdown link. Truncating the raw text first (then escaping) guarantees
    the escaped result never ends mid-escape, even if it slightly exceeds
    `limit` afterward."""
    from src.export.renderer import _link_text

    value = "x" * 79 + "[" + "y" * 20
    label = _link_text(value)

    assert not label.endswith("\\")
    # every bracket in the label must be escaped (no unescaped `[` or `]`)
    unescaped = re.sub(r"\\[\[\]]", "", label)
    assert "[" not in unescaped and "]" not in unescaped

    # a real markdown link built from this label must stay well-formed:
    # the label's own escaping must not consume the link's closing `]`.
    link = f"[{label}](/f.md)"
    assert link.endswith("(/f.md)")
    before_paren = link[: link.rindex("](/f.md)") + 1]
    assert not before_paren.endswith("\\]")


def test_render_interview_escapes_participant_display_name():
    import copy

    header = copy.deepcopy(HEADER)
    header["participants"] = [
        {"handle": "Bob", "display_name": "Bob [Legal]", "provisional": False}
    ]
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(header, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    interview = files["interview.md"]
    label_line = next(ln for ln in interview.splitlines() if "Bob" in ln)
    assert "\\[Legal\\]" in label_line
    assert "[Bob [Legal]]" not in label_line


def test_render_entity_escapes_mention_text_pipe():
    import copy

    entities = copy.deepcopy(ENTITIES)
    entities[0]["mentions"][0]["text"] = "ECU | ABS"
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               entities, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    content = files["entities/ecu.md"]
    row_line = next(ln for ln in content.splitlines() if "ECU" in ln)
    assert "\\|" in row_line
    # exactly the escaped pipe plus the two table-delimiter pipes on either
    # side of the cell should remain -- no raw pipe from the mention text
    assert row_line.count(" | ") == 2
