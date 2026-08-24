from tools.graph.check import check_flow_registrations


def test_real_repo_flow_registrations_clean():
    # every register("Type", Handler) resolves, and every writing handler maps a glossary label
    assert check_flow_registrations(".") == []


def test_unmatched_registration_is_flagged(tmp_path):
    import os
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "src/projections"), exist_ok=True)
    os.makedirs(os.path.join(root, "src/events"), exist_ok=True)
    open(os.path.join(root, "src/events/__init__.py"), "w").close()
    open(os.path.join(root, "src/projections/__init__.py"), "w").close()
    # a register for an event with no FooData class and no handler class
    with open(os.path.join(root, "src/projections/bootstrap.py"), "w") as f:
        f.write('def build(r):\n    r.register("Foo", FooHandler())\n')
    msgs = [f.message for f in check_flow_registrations(root)]
    assert any("Foo" in m for m in msgs)
