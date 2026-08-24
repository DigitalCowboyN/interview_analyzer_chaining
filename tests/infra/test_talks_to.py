import os
from tools.infra.reader import talks_to_pairs
from tools.graph.traverse import walk


def test_client_lib_imports_derive_talks_to():
    pairs = set(talks_to_pairs("."))
    assert ("utils.neo4j_driver", "neo4j") in pairs       # imports neo4j driver
    assert ("events.store", "eventstore") in pairs         # imports esdbclient
    assert ("celery_app", "redis") in pairs                # imports celery


def test_marker_adds_talks_to(tmp_path):
    # a synthetic module with only a `# talks-to:` marker (no client-lib import) still links
    os.makedirs(tmp_path / "src" / "svc", exist_ok=True)
    open(tmp_path / "docker-compose.yml", "w").write(
        "services:\n  neo4j:\n    image: neo4j:5\n")
    open(tmp_path / "src" / "svc" / "__init__.py", "w").close()
    open(tmp_path / "src" / "svc" / "m.py", "w").write("# talks-to: neo4j\ndef f():\n    pass\n")
    assert ("svc.m", "neo4j") in set(talks_to_pairs(str(tmp_path)))


def test_schema_topology_from_a_backing_service():
    # from a backing service, walk inbound: who runs toward it (requires) AND who talks to it
    sg = walk("service:neo4j", direction="in", depth=1, level="module")
    talkers = {e.src for e in sg.edges if e.type == "talks_to"}
    assert "code:utils.neo4j_driver" in talkers
