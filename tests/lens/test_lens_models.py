import pytest

from src.lens.models import load_lens


def test_load_meeting_minutes_lens():
    lens = load_lens("meeting_minutes")
    assert lens.name == "meeting_minutes"
    assert lens.version == 1
    assert set(lens.projects_to) == {"Objective", "Decision", "ActionItem", "FollowUp"}
    assert lens.projects_to["ActionItem"].speaker_link == {
        "field": "owner",
        "relationship": "OWNED_BY",
    }
    scopes = {e.name: e.scope for e in lens.extractors}
    assert scopes["objectives"] == "document"
    assert scopes["decisions"] == "utterance"


def test_extractor_decls_convert_to_specs_with_lens_module():
    lens = load_lens("meeting_minutes")
    spec = next(e for e in lens.extractors if e.name == "decisions").to_extractor_spec()
    from src.models.lens_responses import DecisionsResult

    assert spec.resolve_model() is DecisionsResult


def test_unknown_lens_rejected():
    with pytest.raises(ValueError, match="Unknown lens"):
        load_lens("no_such_lens")


def test_extractor_node_type_must_be_declared(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nversion: 1\nprompts_file: p.yaml\n"
        "projects_to:\n  Decision: {}\n"
        "extractors:\n  - {name: x, prompt_key: k, response_model: DecisionsResult, "
        "scope: utterance, node_type: NotDeclared, items_field: decisions}\n"
    )
    with pytest.raises(ValueError, match="node_type"):
        load_lens("bad", lenses_dir=str(tmp_path))


def test_invalid_label_rejected(tmp_path):
    bad = tmp_path / "bad2.yaml"
    bad.write_text(
        "name: bad2\nversion: 1\nprompts_file: p.yaml\n"
        "projects_to:\n  'DROP DATABASE': {}\nextractors: []\n"
    )
    with pytest.raises(ValueError, match="label"):
        load_lens("bad2", lenses_dir=str(tmp_path))


def test_lens_prompts_format_cleanly():
    from src.utils.helpers import load_yaml

    lens = load_lens("meeting_minutes")
    prompts = load_yaml(lens.prompts_file)
    for decl in lens.extractors:
        formatted = prompts[decl.prompt_key]["prompt"].format(sentence="Test text.")
        assert "{sentence}" not in formatted


def test_lens_response_models_are_openai_strict_compliant():
    from tests.enrichment.test_final_review_fixes import _assert_strict

    lens = load_lens("meeting_minutes")
    for decl in lens.extractors:
        _assert_strict(
            decl.to_extractor_spec().resolve_model().model_json_schema(), decl.name
        )


def test_load_persona_lens():
    lens = load_lens("persona")
    assert lens.name == "persona"
    assert lens.version == 1
    assert set(lens.projects_to) == {"Trait", "Goal", "PainPoint", "NotableQuote"}
    assert lens.projects_to["Goal"].speaker_link == {
        "field": "speaker",
        "relationship": "HELD_BY",
    }
    scopes = {e.name: e.scope for e in lens.extractors}
    assert scopes["traits"] == "document"
    assert scopes["goals"] == "utterance"
    assert scopes["pain_points"] == "utterance"
    assert scopes["notable_quotes"] == "utterance"


def test_persona_extractor_decls_convert_to_specs_with_lens_module():
    lens = load_lens("persona")
    spec = next(e for e in lens.extractors if e.name == "goals").to_extractor_spec()
    from src.models.lens_responses import GoalsResult

    assert spec.resolve_model() is GoalsResult


def test_persona_prompts_format_cleanly():
    from src.utils.helpers import load_yaml

    lens = load_lens("persona")
    prompts = load_yaml(lens.prompts_file)
    for decl in lens.extractors:
        formatted = prompts[decl.prompt_key]["prompt"].format(sentence="Test text.")
        assert "{sentence}" not in formatted


def test_persona_response_models_are_openai_strict_compliant():
    from tests.enrichment.test_final_review_fixes import _assert_strict

    lens = load_lens("persona")
    for decl in lens.extractors:
        _assert_strict(
            decl.to_extractor_spec().resolve_model().model_json_schema(), decl.name
        )
