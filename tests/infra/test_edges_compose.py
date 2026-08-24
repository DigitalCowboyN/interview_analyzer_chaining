from tools.graph.traverse import walk


def test_app_requires_backing_services():
    sg = walk("service:app", direction="out", depth=1, level="module")
    reqs = {e.dst for e in sg.edges if e.type == "requires"}
    assert {"service:neo4j", "service:eventstore", "service:redis"} <= reqs


def test_projection_service_configured_by_lane_count():
    sg = walk("service:projection-service", direction="out", depth=1, level="module")
    cfg = {e.dst for e in sg.edges if e.type == "configured_by"}
    assert "env:PROJECTION_LANE_COUNT" in cfg
    assert "env:ESDB_CONNECTION_STRING" in cfg


def test_backing_service_reached_inbound_from_requirer():
    # required_by is discoverable inbound: who needs neo4j?
    sg = walk("service:neo4j", direction="in", depth=1, level="module")
    requirers = {e.src for e in sg.edges if e.type == "requires"}
    assert "service:app" in requirers and "service:projection-service" in requirers
