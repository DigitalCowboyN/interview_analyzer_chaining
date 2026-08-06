from tools.testmap.reader import TEST_TYPES, load_tests, verifies_edges


def _seed(tmp_path):
    t = tmp_path / "tests"
    (t / "capability").mkdir(parents=True)
    (t / "capability" / "test_check.py").write_text(
        "def test_a():\n    pass\n\ndef test_b():\n    pass\n", encoding="utf-8"
    )
    (t / "integration").mkdir(parents=True)
    (t / "integration" / "test_e2e_user_edits.py").write_text(
        '"""e2e."""\n# verifies: use-cases:correct-what-the-system-got-wrong\n'
        "def test_flow():\n    pass\n",
        encoding="utf-8",
    )
    (t / "integration" / "test_api_calls.py").write_text(
        "def test_call():\n    pass\n", encoding="utf-8"
    )
    # a real tools package DIR so real_code_units()/packages() resolves the target
    # (resolution scans src/ + tools/ dirs, NOT docs/code nodes):
    (tmp_path / "tools" / "capability").mkdir(parents=True)


def test_type_and_target_derivation(tmp_path):
    _seed(tmp_path)
    tests = {t.slug: t for t in load_tests(str(tmp_path))}
    assert tests["capability.test_check"].test_type == "unit"
    assert tests["capability.test_check"].target == "tools.capability"
    assert tests["capability.test_check"].n_tests == 2
    assert tests["integration.test_e2e_user_edits"].test_type == "e2e"
    assert tests["integration.test_api_calls"].test_type == "integration"
    assert tests["integration.test_e2e_user_edits"].verifies == [
        "use-cases:correct-what-the-system-got-wrong"
    ]
    assert "unit" in TEST_TYPES


def test_verifies_edges(tmp_path):
    _seed(tmp_path)
    edges = set(verifies_edges(str(tmp_path)))
    assert ("tests:capability.test_check", "code:tools.capability", "unit") in edges
    assert (
        "tests:integration.test_e2e_user_edits",
        "use-cases:correct-what-the-system-got-wrong",
        "e2e",
    ) in edges
    # unresolved target (no code unit for 'integration') emits no derived→code edge
    assert not any(
        s == "tests:integration.test_api_calls" and d.startswith("code:")
        for s, d, _ in edges
    )
