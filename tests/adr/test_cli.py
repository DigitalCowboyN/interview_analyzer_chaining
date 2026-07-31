import subprocess, sys

def test_cli_check_exits_zero_even_with_findings(tmp_path):
    # a bundle with a broken supersede edge would produce findings, but must still exit 0
    (tmp_path / "0001-a.md").write_text(
        "---\ntype: ADR\nid: 1\ntitle: A\nstatus: accepted\ndate: 2026-07-04\n"
        "supersedes: [2]\nsuperseded_by: []\ntags: []\nsource: docs/x.md\n---\nbody\n",
        encoding="utf-8")
    specs = tmp_path / "specs"; specs.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "tools.adr", "check", "--adr-dir", str(tmp_path), "--specs-dir", str(specs)],
        capture_output=True, text=True)
    assert proc.returncode == 0            # non-blocking guarantee
    assert "0001 supersedes unknown 0002" in (proc.stdout + proc.stderr)

def test_cli_where_reports_governing_adr(tmp_path):
    (tmp_path / "0003-p.md").write_text(
        "---\ntype: ADR\nid: 3\ntitle: Projections\nstatus: accepted\ndate: 2026-07-04\n"
        "governs:\n  - src/projections/\n---\nbody\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "tools.adr", "where", "src/projections/svc.py", "--adr-dir", str(tmp_path)],
        capture_output=True, text=True)
    assert proc.returncode == 0
    assert "ADR-0003" in proc.stdout
