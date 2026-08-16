from tools.corpus.model import OKF_HOMES, Record


def test_okf_homes_cover_the_document_types():
    assert OKF_HOMES == {
        "ADR": "docs/adr",
        "Capability": "docs/capabilities",
        "UseCase": "docs/use-cases",
        "Term": "docs/glossary",
    }


def test_record_is_a_plain_value():
    r = Record(type="Capability", id="import-transcripts",
               path="docs/capabilities/import-transcripts.md",
               frontmatter={"type": "Capability"}, body="…")
    assert r.type == "Capability" and r.id == "import-transcripts"
    assert r.path.endswith("import-transcripts.md")
