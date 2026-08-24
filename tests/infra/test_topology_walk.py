from tools.graph.traverse import walk


def test_backing_services_for_api_reads():
    # "which backing services must be up for the API?" — app runs code:main and requires the stores
    app = walk("service:app", direction="both", depth=2, level="module")
    kinds = {e.dst for e in app.edges if e.type == "requires"}
    assert {"service:neo4j", "service:eventstore"} <= kinds
    assert any(e.type == "runs" and e.dst == "code:main" for e in app.edges)


def test_projection_service_needs():
    # "what does projection-service need?" — deps + entrypoint + config all reachable in one walk
    sg = walk("service:projection-service", direction="out", depth=2, level="module")
    assert any(e.type == "requires" and e.dst == "service:eventstore" for e in sg.edges)
    assert any(e.type == "runs" and e.dst == "code:run_projection_service" for e in sg.edges)
    assert any(e.type == "configured_by" and e.dst == "env:PROJECTION_LANE_COUNT" for e in sg.edges)


def test_schema_blast_from_backing_service_reaches_gold_code():
    # deploy-projection-service gold: subscription_manager is reachable inbound from eventstore
    sg = walk("service:eventstore", direction="in", depth=1, level="module")
    talkers = {e.src for e in sg.edges if e.type == "talks_to"}
    assert "code:projections.subscription_manager" in talkers
    assert "code:events.store" in talkers
