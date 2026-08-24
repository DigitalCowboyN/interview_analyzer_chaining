# tests/infra/test_reader.py
from tools.infra.reader import load_services, load_env_vars


def test_services_have_kind_axis():
    by = {s.id: s for s in load_services(".")}
    # every real compose service is present
    assert {"app", "worker", "projection-service", "redis", "neo4j", "eventstore"} <= set(by)
    # code services build our image + carry a command; backing services are image-only
    assert by["app"].kind == "code" and by["app"].command
    assert by["projection-service"].kind == "code"
    assert by["neo4j"].kind == "backing" and by["neo4j"].image.startswith("neo4j")
    assert by["eventstore"].kind == "backing"


def test_env_vars_are_inline_only_never_dotenv():
    names = {e.name for e in load_env_vars(".")}
    # inline `environment:` vars are modeled
    assert {"PROJECTION_LANE_COUNT", "ESDB_CONNECTION_STRING", "ENABLE_PROJECTION_SERVICE"} <= names
    # a .env-only secret must NOT appear (env_file contents are never read)
    assert not any(n in names for n in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEO4J_PASSWORD"))
    # services record that they load .env as an opaque boolean, not its contents
    assert {s.id: s.loads_env_file for s in load_services(".")}["app"] is True
