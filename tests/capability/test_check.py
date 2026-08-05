# tests/capability/test_check.py
from types import SimpleNamespace as NS
from tools.capability.reader import Capability
from tools.capability.check import (
    CATEGORIES, Finding, check_links, check_coverage, check_classification, check_index_sync,
    run_all,
)
from tools.capability.render import render_index


def _cap(slug, kind="primary", tier="core", parent="", impl=None, category="operations"):
    return Capability(slug, kind, tier, parent, impl or [], f"{slug} does a thing.", "p", category)


def test_links_flag_unknown_unit():
    caps = [_cap("x", impl=["enrichment", "not_a_unit"])]
    msgs = " ".join(f.message for f in check_links(caps, {"enrichment"}))
    assert "not_a_unit" in msgs


def test_coverage_flags_unclaimed_pipeline_unit_but_not_infra():
    nodes = [NS(unit="lens", role="pipeline-layer"), NS(unit="utils", role="infrastructure")]
    caps = [_cap("x", impl=["ingestion"])]  # claims neither
    msgs = " ".join(f.message for f in check_coverage(caps, nodes))
    assert "lens" in msgs and "utils" not in msgs  # infra advisory, not flagged


def test_coverage_parent_package_covers_key_module():
    nodes = [NS(unit="lens.engine", role="pipeline-layer")]
    caps = [_cap("x", impl=["lens"])]  # claims the package → covers the module
    assert check_coverage(caps, nodes) == []


def test_classification_flags_missing_kind_tier_parent():
    caps = [
        _cap("noprimarytier", tier=""),                     # primary w/o tier
        _cap("orphan", kind="child", tier="", parent=""),   # child w/o parent
    ]
    msgs = " ".join(f.message for f in check_classification(caps))
    assert "noprimarytier" in msgs and "orphan" in msgs


def test_index_sync_flags_stale(tmp_path):
    caps = [_cap("x", impl=["enrichment"])]
    idx = tmp_path / "index.md"
    idx.write_text("stale", encoding="utf-8")
    assert check_index_sync(str(idx), caps)  # non-empty
    idx.write_text(render_index(caps), encoding="utf-8")
    assert check_index_sync(str(idx), caps) == []


def test_run_all_returns_list_never_raises(tmp_path):
    assert isinstance(run_all(str(tmp_path)), list)


def test_classification_flags_primary_missing_category():
    caps = [Capability("p", "primary", "core", "", [], "x", "path", "")]  # category ""
    msgs = " ".join(f.message for f in check_classification(caps))
    assert "category" in msgs


def test_coverage_now_flags_unclaimed_tooling():
    nodes = [NS(unit="tools.adr", role="tooling"), NS(unit="utils", role="infrastructure")]
    caps = [_cap("x", impl=["tools.code"])]
    msgs = " ".join(f.message for f in check_coverage(caps, nodes))
    assert "tools.adr" in msgs and "utils" not in msgs  # tooling mandatory; infra still advisory
