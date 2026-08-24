from tools.infra.reader import runs_pairs, _entrypoint_module
from tools.graph.traverse import walk


def test_entrypoint_resolver_extracts_src_module():
    code_ids = {"main", "run_projection_service", "celery_app"}
    assert _entrypoint_module(["uvicorn", "src.main:app", "--reload"], code_ids) == "main"
    assert _entrypoint_module(["python", "-m", "src.run_projection_service"], code_ids) == "run_projection_service"
    assert _entrypoint_module(["celery", "-A", "src.celery_app", "worker"], code_ids) == "celery_app"
    # a command that names no known src.* module resolves to None (no edge, flagged elsewhere)
    assert _entrypoint_module(["bash", "start.sh"], code_ids) is None


def test_services_run_their_entrypoint_modules():
    pairs = dict(runs_pairs("."))
    assert pairs["app"] == "main"
    assert pairs["projection-service"] == "run_projection_service"
    assert pairs["worker"] == "celery_app"


def test_runs_edge_reaches_code_and_is_inbound_discoverable():
    sg = walk("service:projection-service", direction="out", depth=1, level="module")
    assert any(e.type == "runs" and e.dst == "code:run_projection_service" for e in sg.edges)
    # run_by is inbound-discoverable: which service runs this code?
    back = walk("code:main", direction="in", depth=1, level="module")
    assert any(e.type == "runs" and e.src == "service:app" for e in back.edges)
