import inspect
import evals.graph.agentic as ag


def test_harness_uses_no_api_path():
    src = inspect.getsource(ag)
    assert "ANTHROPIC_API_KEY" not in src        # never set an API key
    assert "--bare" not in src                   # never the API-forcing flag


def test_agent_prompt_is_generic_and_tool_scoped():
    scn = {"id": "x", "task": "Trace the obligation of derive_axes.",
           "entry": ["code:tools.graph.classify.derive_axes"], "gold_context": [], "category": "meta",
           "expected": "solvable"}
    p = ag.build_agent_prompt(scn)
    assert "derive_axes" in p and "graph" in p.lower()
    # generic: it teaches the CLI syntax but must NOT prescribe a concrete recipe — it hands the
    # agent the strategy. (The flag names --dir/--depth appear as tool docs; a *fixed* depth number
    # like "--depth 3" would be a prescribed walk.)
    assert "decide your own strategy" in p.lower()
    import re
    assert not re.search(r"--depth\s+\d", p)      # no hardcoded depth recipe


def test_judge_prompt_carries_gold_and_trajectory():
    scn = {"id": "x", "task": "t", "gold_context": ["adr:27"], "gold_answer": "A",
           "category": "meta", "expected": "solvable"}
    p = ag.build_judge_prompt(scn, answer="my answer", trajectory=["walk ..."])
    assert "adr:27" in p and "my answer" in p and "verdict" in p.lower()
