from tools.prompts.reader import extract_values, derive_consumers, load_prompt_entries, PromptEntry

def test_extract_values_both_shapes():
    fmt = 'Report {"entity_type": "person|organization|tool"} for each.'
    assert extract_values(fmt) == ["person", "organization", "tool"]
    bullets = "Choose one.\nOptions:\n  - declarative\n  - interrogative\n  - imperative\n"
    assert extract_values(bullets) == ["declarative", "interrogative", "imperative"]
    assert extract_values("free-form prompt, no enum") == []

def test_derive_consumers_and_lens_convention(tmp_path):
    (tmp_path / "src" / "enrichment").mkdir(parents=True)
    (tmp_path / "src" / "enrichment" / "o.py").write_text('load_yaml("prompts/core_extractors.yaml")', encoding="utf-8")
    assert derive_consumers("core_extractors.yaml", root=str(tmp_path)) == ["enrichment"]
    assert derive_consumers("lens_persona.yaml", root=str(tmp_path)) == ["lens"]   # convention
    assert derive_consumers("task_prompts.yaml", root=str(tmp_path)) == []          # orphan

def test_load_prompt_entries_reads_metadata(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "core_extractors.yaml").write_text(
        "function_type:\n  used_for: [classification]\n  audience: [enrichment]\n"
        "  prompt: |\n    Options:\n      - declarative\n      - interrogative\n", encoding="utf-8")
    (tmp_path / "src" / "enrichment").mkdir(parents=True)
    (tmp_path / "src" / "enrichment" / "o.py").write_text('"prompts/core_extractors.yaml"', encoding="utf-8")
    entries = load_prompt_entries(str(tmp_path))
    e = next(x for x in entries if x.key == "function_type")
    assert e.used_for == ["classification"] and e.audience == ["enrichment"]
    assert e.values == ["declarative", "interrogative"] and e.consumers == ["enrichment"]

def test_extract_values_ignores_instructional_bullets_without_options():
    instructional = ("Group the fragments into utterances:\n"
                     "  - Consecutive fragments from the same speaker\n"
                     "  - If a speaker is interrupted\n"
                     "  - A genuine topic change\n")
    assert extract_values(instructional) == []          # no 'Options:' marker -> not values
    opts = "Classify.\nOptions:\n  - declarative\n  - interrogative\n"
    assert extract_values(opts) == ["declarative", "interrogative"]   # 'Options:' present -> extracted
