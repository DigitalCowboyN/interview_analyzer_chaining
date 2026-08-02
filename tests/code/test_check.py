from tools.code.reader import CodeUnit
from tools.code.check import (
    check_coverage,
    check_classification,
    check_map_in_sync,
    check_stale,
    run_all,
    Finding,
)
from tools.code.render import render_index, render_pipeline


def test_coverage_flags_undocumented_package():
    msgs = " ".join(f.message for f in check_coverage(["resolution", "events"], [CodeUnit("events", "infrastructure")]))
    assert "resolution" in msgs


def test_classification_flags_missing_role():
    msgs = " ".join(f.message for f in check_classification([CodeUnit("x", "")]))
    assert "x" in msgs


def test_stale_flags_unit_not_in_code():
    msgs = " ".join(f.message for f in check_stale([CodeUnit("gone", "surface")], ["events", "api"]))
    assert "gone" in msgs


def test_map_in_sync_clean_then_drift(tmp_path):
    units = [
        CodeUnit("ingestion", "pipeline-layer", [], ["events"], ["ESDB"], "x.", "p"),
        CodeUnit("events", "infrastructure", [], [], ["ESDB"], "y.", "p"),
    ]
    idx, pipe = tmp_path / "index.md", tmp_path / "pipeline.md"
    idx.write_text(render_index(units), encoding="utf-8")
    pipe.write_text(render_pipeline(units), encoding="utf-8")
    # committed artifacts match a fresh render → no findings
    assert check_map_in_sync(str(idx), str(pipe), units) == []
    # a new cross-package dependency appears → both artifacts flagged
    drifted = units + [CodeUnit("api", "surface", [], ["events"], [], "z.", "p")]
    msgs = " ".join(f.message for f in check_map_in_sync(str(idx), str(pipe), drifted))
    assert "index.md" in msgs and "pipeline.md" in msgs


def test_run_all_never_raises_on_empty_root(tmp_path):
    # no src/, no docs/code/ — every check must degrade to a finding, never raise
    findings = run_all(str(tmp_path))
    assert isinstance(findings, list)
