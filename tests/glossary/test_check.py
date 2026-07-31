from tools.glossary.reader import CodeTerm
from tools.glossary.model import Term
from tools.glossary.check import (
    check_coverage, check_enum_values, check_stale_source, Finding,
)


def _code(**kw):
    return {kw["name"]: CodeTerm(kw["name"], kw["kind"], kw.get("source", "src/x.py"), kw.get("values", []))}


def test_coverage_flags_uncovered_code_term():
    code = _code(name="ActorType", kind="enum", values=["HUMAN"])
    msgs = " ".join(f.message for f in check_coverage(code, []))
    assert "ActorType" in msgs and "no glossary term" in msgs


def test_enum_values_reconciled_only_for_enums():
    code = {**_code(name="ActorType", kind="enum", values=["HUMAN", "AI"])}
    term = Term("ActorType", "enum", "src/x.py", ["HUMAN"], "", "p")   # missing AI
    msgs = " ".join(f.message for f in check_enum_values(code, [term]))
    assert "ActorType" in msgs and "AI" in msgs
    # dimension values are NOT reconciled
    dcode = {**_code(name="purpose", kind="dimension", values=[])}
    dterm = Term("purpose", "dimension", "src/m.py", ["statement", "question"], "", "p")
    assert check_enum_values(dcode, [dterm]) == []


def test_stale_source_flags_term_not_in_code():
    term = Term("GoneEnum", "enum", "src/x.py", [], "", "p")
    msgs = " ".join(f.message for f in check_stale_source({}, [term]))
    assert "GoneEnum" in msgs
