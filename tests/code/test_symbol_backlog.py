import os

from tools.code.check import check_missing_symbol_docstring, run_all


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def test_flags_undocumented_symbols_only(tmp_path):
    _w(str(tmp_path / "src/svc/__init__.py"), "")
    _w(str(tmp_path / "src/svc/m.py"),
       'def documented():\n    """Has one."""\n    return 1\n\n'
       'def bare():\n    return 2\n')
    msgs = " ".join(f.message for f in check_missing_symbol_docstring(str(tmp_path)))
    assert "svc.m.bare" in msgs
    assert "svc.m.documented" not in msgs      # documented -> not flagged


def test_symbol_backlog_is_not_in_default_run_all(tmp_path):
    # run_all is the module-grain default; it must NOT include symbol findings (opt-in only)
    _w(str(tmp_path / "src/svc/__init__.py"), "")
    _w(str(tmp_path / "src/svc/m.py"), "def bare():\n    return 2\n")
    msgs = " ".join(f.message for f in run_all(str(tmp_path)))
    assert "symbol" not in msgs
