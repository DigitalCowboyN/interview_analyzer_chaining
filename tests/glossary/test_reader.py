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
