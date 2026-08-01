# tests/graphq/test_reader.py
from tools.graphq.reader import parse_cypher, parse_graphq_marker, load_queries, QueryEntry

def test_parse_cypher():
    q = "MATCH (i:Interview)-[:HAS_SENTENCE]->(s:Sentence) RETURN s.text AS text, s.sentence_id AS sid"
    labels, rels, props, returns = parse_cypher(q)
    assert set(labels) == {"Interview", "Sentence"}
    assert rels == ["HAS_SENTENCE"]
    assert returns == ["text", "sid"]

def test_parse_graphq_marker():
    doc = "Low-conf queue.\n\ngraphq: purpose=export scope=domain-broad audience=[export, api]\n"
    m = parse_graphq_marker(doc)
    assert m["purpose"] == "export" and m["scope"] == "domain-broad" and m["audience"] == ["export", "api"]

def test_load_queries_finds_query_function(tmp_path):
    d = tmp_path / "src" / "export"; d.mkdir(parents=True)
    (d / "reader.py").write_text(
        'def worklist_rows(session):\n'
        '    """Queue.\n\n    graphq: purpose=export scope=domain-broad audience=[export]\n    """\n'
        '    return session.run("MATCH (i:Interview) RETURN i.interview_id AS interview_id")\n',
        encoding="utf-8")
    entries = load_queries(str(tmp_path))
    e = next(x for x in entries if x.name == "worklist_rows")
    assert e.bundle == "src/export/reader.py" and e.purpose == "export"
    assert e.labels == ["Interview"] and e.returns == ["interview_id"]
