import pytest
from tools.adr.intent import is_architectural

@pytest.mark.parametrize("prompt", [
    "Let's brainstorm the design for the new exporter",
    "Should we switch the queue to Redis? What are the trade-offs?",
    "write a spec for the ingestion refactor",
])
def test_matches_architectural_intent(prompt):
    assert is_architectural(prompt) is True

@pytest.mark.parametrize("prompt", [
    "fix the failing test in test_reader.py",
    "what does line 42 do?",
    "bump the black version",
])
def test_ignores_non_architectural(prompt):
    assert is_architectural(prompt) is False
