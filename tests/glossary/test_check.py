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


def test_enum_values_matches_code_symbol_literal():
    from tools.glossary.reader import CodeTerm
    from tools.glossary.model import Term
    from tools.glossary.check import check_enum_values
    code = {"Claim.kind": CodeTerm("Claim.kind", "literal", "src/m.py", ["assertion", "commitment", "request"])}
    term = Term("claim-kind", "claim-kind", "src/m.py", ["assertion", "commitment"], "", "p")  # missing request
    term.code_symbol = "Claim.kind"
    msgs = " ".join(f.message for f in check_enum_values(code, [term]))
    assert "claim-kind" in msgs and "request" in msgs


def test_coverage_ignores_literals():
    from tools.glossary.reader import CodeTerm
    from tools.glossary.check import check_coverage
    # a plain Literal (ExtractorSpec.scope-like) is NOT passed to coverage -> no finding
    code = {"ActorType": CodeTerm("ActorType", "enum", "src/e.py", ["HUMAN"])}
    term = Term("ActorType", "enum", "src/e.py", ["HUMAN"], "", "p")
    assert check_coverage(code, [term]) == []


def test_stale_source_skips_registry_pinned_and_honors_code_symbol():
    from tools.glossary.reader import CodeTerm
    from tools.glossary.check import check_stale_source
    lits = {"ClaimItem.kind": CodeTerm("ClaimItem.kind", "literal", "src/m.py", ["assertion"])}
    ck = Term("claim-kind", "claim-kind", "src/m.py", ["assertion"], "", "p"); ck.code_symbol = "ClaimItem.kind"
    et = Term("entity-type", "entity-type", "prompts/core_extractors.yaml", ["person"], "", "p")
    # claim-kind's code_symbol resolves -> not stale; entity-type registry-pinned -> not stale
    assert check_stale_source(lits, [ck, et]) == []
    # a code_symbol that vanished -> stale
    ck2 = Term("claim-kind", "claim-kind", "src/m.py", ["assertion"], "", "p"); ck2.code_symbol = "Gone.kind"
    assert check_stale_source({}, [ck2])
