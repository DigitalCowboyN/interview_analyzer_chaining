from tools.glossary.model import Term
from tools.glossary.render import render_index


def test_render_index():
    out = render_index([Term("ActorType", "enum", "src/events/envelope.py", ["HUMAN"], "", "p")])
    assert "| ActorType | enum | src/events/envelope.py |" in out
