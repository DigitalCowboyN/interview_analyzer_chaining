from tools.prompts.reader import PromptEntry
from tools.glossary.model import Term
from tools.prompts.check import (
    check_values_vs_glossary, check_audience_vs_consumers, check_orphan, Finding,
)

def test_values_vs_glossary_names_glossary_as_fix():
    entry = PromptEntry("core_extractors.yaml", "purpose", ["classification"], ["enrichment"],
                        ["Statement", "Query"], ["enrichment"])
    terms = [Term("purpose", "dimension", "prompts/core_extractors.yaml", ["Statement"], "", "p")]  # missing Query
    msgs = " ".join(f.message for f in check_values_vs_glossary([entry], terms))
    assert "purpose" in msgs and "glossary" in msgs.lower() and "Query" in msgs

def test_audience_vs_consumers():
    # declared internal role with no consumer -> flagged
    e1 = PromptEntry("x.yaml", "k", ["extraction"], ["api"], [], [])
    m1 = " ".join(f.message for f in check_audience_vs_consumers([e1]))
    assert "api" in m1 and "no code consumes" in m1
    # consumed by a role not declared -> flagged
    e2 = PromptEntry("x.yaml", "k", ["extraction"], [], [], ["enrichment"])
    m2 = " ".join(f.message for f in check_audience_vs_consumers([e2]))
    assert "enrichment" in m2 and "not" in m2.lower()
    # external role declared -> not reconciled (no finding for cli alone)
    e3 = PromptEntry("x.yaml", "k", ["extraction"], ["cli"], [], ["enrichment"])
    m3 = " ".join(f.message for f in check_audience_vs_consumers([e3]))
    assert "cli" not in m3

def test_orphan_flags_no_consumer():
    e = PromptEntry("task_prompts.yaml", "sentence_purpose", [], [], ["Statement"], [])
    msgs = " ".join(f.message for f in check_orphan([e]))
    assert "task_prompts.yaml" in msgs and "unused" in msgs
