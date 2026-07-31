import subprocess
import sys


def test_cli_check_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.glossary", "check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "glossary-check" in proc.stdout


def test_scaffold_enum_prefills_values(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "e.py").write_text(
        "from enum import Enum\nclass Color(str, Enum):\n    RED = 'r'\n    BLUE = 'b'\n", encoding="utf-8")
    from tools.glossary.scaffold import new_term
    path = new_term("Color", "enum", root=str(tmp_path))
    body = open(path, encoding="utf-8").read()
    assert "term: Color" in body and "values: [RED, BLUE]" in body
    assert "source: src/e.py" in body
