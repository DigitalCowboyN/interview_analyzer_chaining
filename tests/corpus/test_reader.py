import os

from tools.corpus.reader import okf_records


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def test_discovers_by_frontmatter_not_body(tmp_path):
    root = str(tmp_path)
    # a real capability record (its OWN frontmatter is type: Capability)
    _write(os.path.join(root, "docs/capabilities/import.md"),
           "---\ntype: Capability\n---\nImport transcripts.\n")
    # a PLAN that merely EMBEDS `type: Capability` in a fenced example — NOT a record
    _write(os.path.join(root, "docs/superpowers/plans/p.md"),
           "# A plan\n\nExample frontmatter:\n\n```\ntype: Capability\n```\n")
    recs = okf_records(root)
    assert [(r.type, r.id) for r in recs] == [("Capability", "import")]


def test_finds_misfiled_record_anywhere(tmp_path):
    root = str(tmp_path)
    # a Capability record sitting in the ADR folder — must still be discovered (type-primary)
    _write(os.path.join(root, "docs/adr/stray.md"),
           "---\ntype: Capability\n---\nStray.\n")
    recs = okf_records(root)
    assert [(r.type, r.path.replace(os.sep, "/")) for r in recs] == [
        ("Capability", "docs/adr/stray.md")]


def test_ignore_dirs_and_no_frontmatter(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "node_modules/pkg/readme.md"),
           "---\ntype: Capability\n---\nvendored, must be ignored.\n")
    _write(os.path.join(root, "docs/notes.md"), "# just a note, no frontmatter\n")
    assert okf_records(root) == []
