from tools.adr.scaffold import next_id, new_adr
from tools.adr.model import parse_adr

def test_next_id_starts_at_one_and_increments(tmp_path):
    assert next_id(str(tmp_path)) == 1
    new_adr(str(tmp_path), "First decision")
    assert next_id(str(tmp_path)) == 2

def test_new_adr_writes_parseable_stub(tmp_path):
    path = new_adr(str(tmp_path), "Use Redis for the queue")
    adr = parse_adr(open(path, encoding="utf-8").read(), path=path)
    assert adr.id == 1
    assert adr.status == "proposed"
    assert "use-redis-for-the-queue" in path
    assert "## Decision" in adr.body
