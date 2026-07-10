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
