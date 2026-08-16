import os

from tools.corpus.check import check_unregistered_types


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def test_unregistered_type_is_flagged(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "docs/policies/p.md"), "---\ntype: Policy\n---\nbody\n")
    _write(os.path.join(root, "docs/capabilities/c.md"), "---\ntype: Capability\n---\nok\n")
    msgs = [f.message for f in check_unregistered_types(root)]
    assert any("Policy" in m for m in msgs)
    assert not any("Capability" in m for m in msgs)   # registered → not flagged


def test_only_registered_types_is_clean(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "docs/adr/a.md"), "---\ntype: ADR\n---\nok\n")
    assert check_unregistered_types(root) == []


def test_body_fenced_type_is_not_flagged(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "docs/plans/p.md"), "# A plan\n\n```\ntype: Policy\n```\n")
    assert check_unregistered_types(root) == []   # top frontmatter only, not body
