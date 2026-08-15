from tools.knowledge.surfaces import changed_domains


def test_changed_domains_maps_capability_edit():
    assert changed_domains(["docs/capabilities/x.md"]) == ["capability"]


def test_changed_domains_src_touches_code_family():
    got = set(changed_domains(["src/api/routers/foo.py"]))
    assert {"api", "code", "capability", "glossary", "prompts"} <= got
    assert "usecase" not in got and "testmap" not in got


def test_changed_domains_unmatched_path_is_empty():
    assert changed_domains(["README.md"]) == []


def test_changed_domains_dedupes_and_sorts():
    out = changed_domains(["tests/a.py", "tests/b.py"])
    assert out == ["testmap"]
