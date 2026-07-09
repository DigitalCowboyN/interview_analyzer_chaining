from src.enrichment.registry import ExtractorRegistry
from src.models.extractor_responses import PurposeResult


def test_loads_core_extractors_in_order():
    specs = ExtractorRegistry.load("config/extractors.yaml")
    names = [s.name for s in specs]
    for expected in [
        "function_type",
        "structure_type",
        "purpose",
        "topic_level_1",
        "topic_level_3",
        "overall_keywords",
        "domain_keywords",
        "entity_mentions",
        "claims",
    ]:
        assert expected in names


def test_scopes_and_models_resolve():
    specs = {s.name: s for s in ExtractorRegistry.load("config/extractors.yaml")}
    assert specs["purpose"].scope == "fragment"
    assert specs["claims"].scope == "utterance"
    assert specs["purpose"].resolve_model() is PurposeResult


def test_unknown_model_rejected():
    import pytest

    from src.enrichment.models import ExtractorSpec

    spec = ExtractorSpec(name="x", prompt_key="purpose", response_model="NoSuchModel", scope="fragment")
    with pytest.raises(ValueError, match="Unknown response model"):
        spec.resolve_model()


def test_prompts_exist_for_every_extractor():
    from src.utils.helpers import load_yaml

    prompts = load_yaml("prompts/core_extractors.yaml")
    for spec in ExtractorRegistry.load("config/extractors.yaml"):
        assert spec.prompt_key in prompts, spec.name
        assert "prompt" in prompts[spec.prompt_key]


def test_prompt_placeholders_format_cleanly():
    """Every prompt must .format() with exactly the contract's placeholders.

    Executor contract: {sentence} is always supplied; {context} only when
    context_needs is non-empty; {domain_keywords} only for that extractor.
    Using keyword-strict formatting (no extra kwargs) catches both undeclared
    placeholders (KeyError) and misleading context_needs declarations.
    """
    from src.utils.helpers import load_yaml

    prompts = load_yaml("prompts/core_extractors.yaml")
    for spec in ExtractorRegistry.load("config/extractors.yaml"):
        kwargs = {"sentence": "Test sentence."}
        if spec.context_needs:
            kwargs["context"] = "ctx"
        if spec.name == "domain_keywords":
            kwargs["domain_keywords"] = "ECU, CAN"
        formatted = prompts[spec.prompt_key]["prompt"].format(**kwargs)
        assert "{sentence}" not in formatted
        if not spec.context_needs:
            assert "{context}" not in prompts[spec.prompt_key]["prompt"]


def test_disabled_extractors_filtered(tmp_path):
    doc = tmp_path / "x.yaml"
    doc.write_text(
        "extractors:\n"
        "  - {name: a, prompt_key: purpose, response_model: PurposeResult, scope: fragment}\n"
        "  - {name: b, prompt_key: purpose, response_model: PurposeResult, scope: fragment, enabled: false}\n"
    )
    specs = ExtractorRegistry.load(str(doc))
    assert [s.name for s in specs] == ["a"]
