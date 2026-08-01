from tools.glossary.reader import code_enums, code_dimensions, CodeTerm


def test_code_enums_extracts_members(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "e.py").write_text(
        "from enum import Enum\n"
        "class Color(str, Enum):\n    RED = 'red'\n    BLUE = 'blue'\n\n"
        "class Plain:\n    x = 1\n", encoding="utf-8")
    enums = code_enums(str(tmp_path))
    assert "Color" in enums and enums["Color"].kind == "enum"
    assert enums["Color"].values == ["RED", "BLUE"]
    assert enums["Color"].source == "src/e.py"
    assert "Plain" not in enums          # non-enum ignored


def test_code_dimensions_reads_annotated_fields(tmp_path):
    p = tmp_path / "src" / "models"; p.mkdir(parents=True)
    (p / "analysis_result.py").write_text(
        "from pydantic import BaseModel\n"
        "class AnalysisResult(BaseModel):\n"
        "    '''Attributes: not a field.'''\n"
        "    function_type: str\n    purpose: str\n", encoding="utf-8")
    dims = code_dimensions(str(tmp_path))
    assert set(dims) == {"function_type", "purpose"}
    assert dims["function_type"].kind == "dimension"
    assert "Attributes" not in dims       # docstring not a field


def test_code_literals_extracts_field_literals(tmp_path):
    from tools.glossary.reader import code_literals
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text(
        "from typing import Literal\n"
        "class Claim:\n    kind: Literal['assertion', 'commitment', 'request']\n", encoding="utf-8")
    lits = code_literals(str(tmp_path))
    assert "Claim.kind" in lits
    assert lits["Claim.kind"].values == ["assertion", "commitment", "request"]
    assert lits["Claim.kind"].source == "src/m.py"


def test_graph_vocabulary_extracts_labels_rels_props(tmp_path):
    from tools.glossary.reader import graph_vocabulary
    p = tmp_path / "src" / "projections"; p.mkdir(parents=True)
    (p / "h.py").write_text(
        'q = ("MERGE (c:Claim {claim_id: $id}) SET c.confidence = 0.9 "\n'
        '    "MERGE (s:Speaker)-[:MADE_BY]->(c)")\n', encoding="utf-8")
    gv = graph_vocabulary(str(tmp_path))
    assert gv["Claim"].kind == "graph-label" and gv["Speaker"].kind == "graph-label"
    assert gv["MADE_BY"].kind == "rel-type"
    assert "claim_id" in gv and gv["claim_id"].kind == "graph-property"
    assert "confidence" in gv


def test_graph_vocabulary_ignores_python_attributes(tmp_path):
    from tools.glossary.reader import graph_vocabulary
    p = tmp_path / "src" / "projections"; p.mkdir(parents=True)
    (p / "h.py").write_text(
        "class H:\n"
        "    def __init__(self):\n"
        "        self._buffer = []\n"          # python attr, NOT a graph prop
        "        self.handler_registry = {}\n"
        "    def run(self, session):\n"
        "        return session.run('MERGE (c:Claim {claim_id: $id}) SET c.confidence = 0.9')\n",
        encoding="utf-8")
    gv = graph_vocabulary(str(tmp_path))
    assert "claim_id" in gv and "confidence" in gv          # from the Cypher string
    assert "_buffer" not in gv and "handler_registry" not in gv   # python attrs excluded
    assert gv["Claim"].kind == "graph-label"
