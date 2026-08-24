import os

from tools.infra.check import check_infra


def test_real_repo_infra_clean():
    # every code-service command resolves to a code node; every talks-to marker names a real service
    assert check_infra(".") == []


def test_string_literal_marker_is_not_flagged(tmp_path):
    # a `# talks-to:` sequence inside a STRING (not a real comment) must be ignored — no false finding
    os.makedirs(tmp_path / "src" / "svc", exist_ok=True)
    open(tmp_path / "docker-compose.yml", "w").write("services:\n  neo4j:\n    image: neo4j:5\n")
    open(tmp_path / "src" / "svc" / "__init__.py", "w").close()
    open(tmp_path / "src" / "svc" / "m.py", "w").write(
        'DESC = "see # talks-to: notaservice for the syntax"\n')   # inside a string, not a comment
    assert check_infra(str(tmp_path)) == []


def test_unresolvable_command_is_flagged(tmp_path):
    open(tmp_path / "docker-compose.yml", "w").write(
        "services:\n"
        "  app:\n"
        "    build: .\n"
        "    command: [\"bash\", \"start.sh\"]\n")   # code service, no resolvable src.* module
    msgs = [f.message for f in check_infra(str(tmp_path))]
    assert any("app" in m and "command" in m for m in msgs)
