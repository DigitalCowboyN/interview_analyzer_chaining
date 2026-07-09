"""Deterministic spaCy cross-check for function/structure classifications.

Does NOT replace the LLM calls (spec: expansion, not reduction); disagreement
becomes a review flag carried on the AnalysisGenerated event.
"""

from typing import Dict

from src.utils.text_processing import nlp


def _spacy_function_type(doc, text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("?"):
        return "interrogative"
    if stripped.endswith("!"):
        return "exclamatory"
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is not None and root.pos_ == "VERB" and root.tag_ == "VB":
        has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in root.children)
        if not has_subject:
            return "imperative"
    return "declarative"


def _spacy_structure_type(doc) -> str:
    has_coord = any(t.dep_ == "conj" and t.head.dep_ == "ROOT" for t in doc)
    has_sub = any(t.dep_ in ("advcl", "ccomp", "relcl", "acl", "csubj") for t in doc)
    if has_coord and has_sub:
        return "compound-complex"
    if has_coord:
        return "compound"
    if has_sub:
        return "complex"
    return "simple"


def syntax_flags(text: str, function_type: str, structure_type: str) -> Dict[str, str]:
    """Return review flags where spaCy's parse disagrees with the LLM labels.

    Empty labels are never flagged (the dimension may have been omitted after
    a validation failure). Returns {} when spaCy is unavailable.
    """
    if nlp is None or not text.strip():
        return {}
    doc = nlp(text)
    flags: Dict[str, str] = {}
    if function_type:
        spacy_fn = _spacy_function_type(doc, text)
        if spacy_fn != function_type.lower():
            flags["function_type_disagreement"] = f"spacy={spacy_fn} llm={function_type}"
    if structure_type:
        spacy_st = _spacy_structure_type(doc)
        if spacy_st != structure_type.lower():
            flags["structure_type_disagreement"] = f"spacy={spacy_st} llm={structure_type}"
    return flags
