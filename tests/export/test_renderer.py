import re

import pytest
import yaml

from src.export.renderer import item_dir, item_filename, render_bundle, slugify
from src.lens.models import load_lens

HEADER = {
    "interview_id": "iid-1", "title": "Q3 Vendor Selection", "source": "m.txt",
    "started_at": "2026-07-01T00:00:00", "project_id": "telemetry",
    "metadata": {"front_matter": {"participants": ["Alice Johnson"]}},
    "participants": [{"speaker_id": "sp1", "handle": "Alice", "display_name": "Alice Johnson", "provisional": False}],
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
PERSONS = [{"speaker_id": "sp1", "person_id": "person-1", "display_name": "Jane Doe"}]


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


def test_link_text_escapes_backslashes():
    r"""A raw trailing backslash must not be able to escape the link's closing
    `]`: backslashes are escaped first, so `_link_text("bad\\")` ends with two
    backslashes (the escaped literal), and a link built from a value ending in
    `\` never yields a *lone* trailing backslash immediately before the
    label's closing bracket -- i.e. the label's own escaping can never
    consume the link's closing `]`."""
    from src.export.renderer import _link_text

    label = _link_text("bad\\")
    assert label.endswith("\\\\")

    link = f"[{_link_text('trailing' + chr(92))}](x)"
    assert link.endswith("](x)")
    # the character immediately before the label's closing `]` must not be a
    # lone (odd count of) backslash -- an even count is fully escaped and inert.
    label_end = link.rindex("]")
    trailing_backslashes = len(link[:label_end]) - len(link[:label_end].rstrip("\\"))
    assert trailing_backslashes % 2 == 0


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


def test_slug_registry_uniquifies_and_hashes():
    from src.export.renderer import _SlugRegistry

    reg = _SlugRegistry()
    assert reg.slug_for("ECU") == "ecu"
    assert reg.slug_for("ecu") == "ecu-2"          # collision
    assert reg.slug_for("ECU!") == "ecu-3"          # normalizes then collides
    hashed = reg.slug_for("!!!")
    assert hashed.startswith("x-") and len(hashed) == 10


def test_bundle_entity_slug_collision_yields_two_files():
    import copy

    entities = copy.deepcopy(ENTITIES) + [
        {"surface": "ECU", "entity_type": "product", "mentions": []}
    ]  # ENTITIES already has surface "ecu"
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               entities, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    entity_files = [p for p in files if p.startswith("entities/")]
    assert len(entity_files) == 2 and len(set(entity_files)) == 2


def test_speaker_slug_keyed_by_id_not_identity_string():
    """Two distinct speakers who both lack a display_name/handle (both reduce
    to identity string "unknown") must not collapse onto one cache entry: each
    speaker_id gets its own registry slug, its own file, and claims by either
    speaker must link to that speaker's own file -- not overwrite it."""
    header = {
        "interview_id": "iid-2", "title": "Anon Session", "source": "m.txt",
        "started_at": "2026-07-01T00:00:00", "project_id": "telemetry",
        "metadata": {"front_matter": {}},
        "participants": [],
        "fragment_count": 2, "utterance_count": 1,
        "lens": "meeting_minutes", "lens_version": 1,
    }
    transcript = [
        {"sentence_id": "f1", "sequence_order": 0, "text": "First speaker line.",
         "speaker_id": "spA", "speaker": None, "utterance_id": "u-a"},
        {"sentence_id": "f2", "sequence_order": 1, "text": "Second speaker line.",
         "speaker_id": "spB", "speaker": None, "utterance_id": "u-b"},
    ]
    speakers = [
        {"speaker_id": "spA", "handle": None, "display_name": None, "provisional": False},
        {"speaker_id": "spB", "handle": None, "display_name": None, "provisional": False},
    ]
    claims = [
        {"claim_id": "cB", "text": "Second speaker's claim", "kind": "commitment",
         "confidence": 0.8, "model": "haiku", "provider": "anthropic", "speaker_id": "spB",
         "speaker": None, "supporting_fragment_ids": ["f2"]},
    ]
    lens = load_lens("meeting_minutes")
    files_list = render_bundle(
        header, transcript, speakers, [], claims, [], [], lens, exported_at="2026-07-10T12:00:00+00:00"
    )
    files = dict(files_list)

    # No path collisions anywhere in the bundle.
    assert len(files_list) == len(files)

    # Both anonymous speakers get their own file.
    assert "speakers/unknown.md" in files
    assert "speakers/unknown-2.md" in files

    # spA's file must actually belong to spA (id in frontmatter), likewise spB.
    fm_a = yaml.safe_load(files["speakers/unknown.md"].split("---\n")[1])
    fm_b = yaml.safe_load(files["speakers/unknown-2.md"].split("---\n")[1])
    assert {fm_a["id"], fm_b["id"]} == {"spA", "spB"}

    # The claim (made by spB) links to spB's own file, whichever slug that is.
    claim_content = files["claims/claim-cB.md"]
    slug_for_b = "unknown" if fm_a["id"] == "spB" else "unknown-2"
    assert f"MADE_BY: [Unknown](/speakers/{slug_for_b}.md)" in claim_content


def test_two_surfaces_sharing_canonical_render_one_entity_file():
    entities = [
        {"surface": "ecu", "entity_type": "product", "canonical_id": "canon-1", "canonical_name": "ECU",
         "mentions": [{"sentence_id": "f1", "start": 0, "end": 3, "text": "ECU", "confidence": 0.9}]},
        {"surface": "Engine Control Unit", "entity_type": "product", "canonical_id": "canon-1",
         "canonical_name": "ECU",
         "mentions": [{"sentence_id": "f2", "start": 0, "end": 19, "text": "Engine Control Unit",
                       "confidence": 0.8}]},
    ]
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               entities, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    entity_files = [p for p in files if p.startswith("entities/")]
    assert len(entity_files) == 1
    path = entity_files[0]
    content = files[path]
    fm = yaml.safe_load(content.split("---\n")[1])
    assert fm["id"] == "canon-1"
    assert fm["title"] == "ECU"
    assert set(fm["aliases"]) == {"ecu", "Engine Control Unit"}
    assert "ECU" in content and "Engine Control Unit" in content


def test_surface_without_canonical_renders_per_surface_regression():
    files = render()
    content = files["entities/ecu.md"]
    fm = yaml.safe_load(content.split("---\n")[1])
    assert fm["type"] == "Entity"
    assert fm["title"] == "ecu"
    assert "aliases" not in fm


def test_render_person_produces_persons_file():
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00",
                               persons=PERSONS))
    person_files = [p for p in files if p.startswith("persons/")]
    assert len(person_files) == 1
    path = person_files[0]
    content = files[path]
    fm = yaml.safe_load(content.split("---\n")[1])
    assert fm["type"] == "Person"
    assert fm["id"] == "person-1"
    assert fm["title"] == "Jane Doe"


def test_speaker_file_links_to_identified_person():
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00",
                               persons=PERSONS))
    speaker_content = files["speakers/alice-johnson.md"]
    person_path = next(p for p in files if p.startswith("persons/"))
    person_slug = person_path.split("/", 1)[1].removesuffix(".md")
    assert f"Identified as [Jane Doe](/persons/{person_slug}.md)" in speaker_content


def test_index_lists_persons_section_when_persons_exist():
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00",
                               persons=PERSONS))
    index = files["index.md"]
    assert "## Persons" in index
    person_path = next(p for p in files if p.startswith("persons/"))
    assert f"]({person_path})" in index


def test_index_omits_persons_section_when_no_persons():
    files = render()
    assert "## Persons" not in files["index.md"]


_SEGMENT_TRANSCRIPT = [
    {"sentence_id": "f1", "sequence_order": 0, "text": "Let's talk roadmap.",
     "speaker_id": "sp1", "speaker": "Alice Johnson", "utterance_id": "u-abc"},
    {"sentence_id": "f2", "sequence_order": 1, "text": "Ship in Q3.",
     "speaker_id": "sp1", "speaker": "Alice Johnson", "utterance_id": "u-abc"},
    {"sentence_id": "f3", "sequence_order": 2, "text": "Now, budget.",
     "speaker_id": "sp1", "speaker": "Alice Johnson", "utterance_id": "u-abc"},
]
_SEGMENTS = [
    {"segment_id": "s1", "topic": "Roadmap", "confidence": 0.9, "start_index": 0, "end_index": 1},
    {"segment_id": "s2", "topic": "Budget", "confidence": 0.8, "start_index": 2, "end_index": 2},
]


def test_transcript_renders_segment_headings_at_start_indices():
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(
        HEADER, _SEGMENT_TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS, ENTITIES, ANALYSIS, lens,
        exported_at="2026-07-10T12:00:00+00:00", segments=_SEGMENTS,
    ))
    content = files["transcript.md"]
    expected = (
        "---\ntype: Transcript\n---\n"
        "\n"
        "# Transcript\n"
        "\n"
        "## Roadmap\n"
        "\n"
        '<a id="u-1"></a>\n'
        "**Speaker:** Alice Johnson\n"
        "\n"
        "Let's talk roadmap.\n"
        "Ship in Q3.\n"
        "\n"
        "## Budget\n"
        "\n"
        '<a id="u-1"></a>\n'
        "**Speaker:** Alice Johnson\n"
        "\n"
        "Now, budget.\n"
    )
    assert content == expected
    # Order check: "## Roadmap" appears before the first fragment's text, and
    # "## Budget" appears before the third fragment's text, in that order.
    assert content.index("## Roadmap") < content.index("Let's talk roadmap.")
    assert content.index("Let's talk roadmap.") < content.index("## Budget")
    assert content.index("## Budget") < content.index("Now, budget.")


def test_transcript_without_segments_is_byte_identical_to_before():
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(
        HEADER, _SEGMENT_TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS, ENTITIES, ANALYSIS, lens,
        exported_at="2026-07-10T12:00:00+00:00",
    ))
    content = files["transcript.md"]
    expected = (
        "---\ntype: Transcript\n---\n"
        "\n"
        "# Transcript\n"
        "\n"
        '<a id="u-1"></a>\n'
        "**Speaker:** Alice Johnson\n"
        "\n"
        "Let's talk roadmap.\n"
        "Ship in Q3.\n"
        "Now, budget.\n"
    )
    assert content == expected
