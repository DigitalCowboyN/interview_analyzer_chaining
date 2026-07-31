from tools.glossary.model import parse_term, load_glossary, Term

TERM = ("---\ntype: Term\nterm: ActorType\nkind: enum\n"
        "source: src/events/envelope.py\nvalues: [HUMAN, SYSTEM, AI]\n---\nWho caused an event.\n")


def test_parse_term(tmp_path):
    t = parse_term(TERM, path="docs/glossary/actortype.md")
    assert t.term == "ActorType" and t.kind == "enum"
    assert t.values == ["HUMAN", "SYSTEM", "AI"]
    assert "Who caused" in t.definition


def test_load_glossary_skips_index(tmp_path):
    (tmp_path / "index.md").write_text("# generated\n", encoding="utf-8")
    (tmp_path / "actortype.md").write_text(TERM, encoding="utf-8")
    terms = load_glossary(str(tmp_path))
    assert [t.term for t in terms] == ["ActorType"]
