from tools.adr.code_links import extract_refs, adr_ids_from_refs, scan_markers


def test_extract_refs_and_ids():
    text = '"""Module.\n\ngoverned-by: ADR-0003, ADR-0001\n"""\n'
    refs = extract_refs(text)
    assert refs == ["ADR-0003", "ADR-0001"]
    assert adr_ids_from_refs(refs) == [3, 1]


def test_scan_markers_keys_files_and_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pkg").mkdir()
    # file marker -> keyed by file path
    (tmp_path / "src" / "svc.py").write_text('"""Svc.\ngoverned-by: ADR-0003\n"""\n', encoding="utf-8")
    # __init__.py marker -> keyed by directory path
    (tmp_path / "src" / "pkg" / "__init__.py").write_text('"""Pkg.\ngoverned-by: ADR-0005\n"""\n', encoding="utf-8")
    # README marker -> keyed by directory path
    (tmp_path / "src" / "pkg" / "README.md").write_text("governed-by: ADR-0009\n", encoding="utf-8")
    # unmarked file -> absent
    (tmp_path / "src" / "plain.py").write_text('"""nothing here"""\n', encoding="utf-8")
    markers = scan_markers(str(tmp_path))
    assert markers["src/svc.py"] == ["ADR-0003"]
    assert sorted(markers["src/pkg/"]) == ["ADR-0005", "ADR-0009"]
    assert "src/plain.py" not in markers
