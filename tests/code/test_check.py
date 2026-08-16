from tools.code.reader import CodeUnit
from tools.code.check import check_map_in_sync, check_missing_docstring, run_all
from tools.code.render import render_index, render_pipeline


def test_missing_docstring_flags_modules_only():
    units = [
        CodeUnit("api", level="package", description=""),            # package: not flagged
        CodeUnit("api.main", level="module", description=""),        # module, no docstring: flagged
        CodeUnit("api.ok", level="module", description="Has one."),  # module with docstring: not
    ]
    msgs = " ".join(f.message for f in check_missing_docstring(units))
    assert "api.main" in msgs and "api.ok" not in msgs and "code: module api " not in msgs


def test_map_in_sync_clean_then_drift(tmp_path):
    units = [CodeUnit("api", level="package"),
             CodeUnit("api.main", level="module", depends_on=["events"])]
    idx, pipe = tmp_path / "index.md", tmp_path / "pipeline.md"
    idx.write_text(render_index(units), encoding="utf-8")
    pipe.write_text(render_pipeline(units), encoding="utf-8")
    assert check_map_in_sync(str(idx), str(pipe), units) == []
    drifted = units + [CodeUnit("api.new", level="module", depends_on=["events"])]
    msgs = " ".join(f.message for f in check_map_in_sync(str(idx), str(pipe), drifted))
    assert "index.md" in msgs and "pipeline.md" in msgs


def test_run_all_never_raises_on_empty_root(tmp_path):
    findings = run_all(str(tmp_path))
    assert isinstance(findings, list)
