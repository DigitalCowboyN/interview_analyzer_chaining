# evals/graph/agentic.py
"""Layer-2 agentic eval harness, driven by the Claude Code SUBSCRIPTION (never an API key).

Mode A: headless `claude -p` subprocess per scenario (agent-under-test + judge), subscription auth.
Mode B: when headless is unavailable here, the pre-commit routine in AGENTIC.md drives the same
scenarios via subagents inside a Claude Code session; this module's prompt builders + aggregate() are
reused, and answers/trajectories are read from evals/graph/.runs/<id>.json written by that routine.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List

from evals.graph.run import load_scenarios

RUNS_DIR = os.path.join(os.path.dirname(__file__), ".runs")
_AGENT_SYSTEM = (
    "You are an eval agent. Investigate ONLY via the graph CLI you are given; do NOT read source "
    "files or repo docs. Start coarse and expand progressively — walk up to governing intent, then "
    "outward. If the graph cannot answer, say so explicitly; never infer relevance from proximity. "
    "End with a clear final answer.")
_ALLOWED = ("Bash(python -m tools.graph walk:*),Bash(python -m tools.graph context:*)")


def build_agent_prompt(scenario: dict) -> str:
    return (f"{scenario['task']}\n\n"
            "Nodes are addressed <domain>:<id> (code:, capabilities:, use-cases:, adr:, tests:). "
            "Your only tools are:\n"
            "  python -m tools.graph walk <entry> --dir out|in|both --depth N|full --level module|symbol\n"
            "  python -m tools.graph context <entry>   (minimal task-context: walk up to intent + local)\n"
            f"A sensible starting node: {scenario['entry'][0]}. Decide your own strategy.")


def build_judge_prompt(scenario: dict, answer: str, trajectory: List[str]) -> str:
    rubric = open(os.path.join(os.path.dirname(__file__), "RUBRIC.md"), encoding="utf-8").read()
    return (f"{rubric}\n\n---\nTASK: {scenario['task']}\n"
            f"EXPECTED: {scenario['expected']} (a 'gap' scenario PASSES if the agent correctly "
            f"reports the graph cannot answer).\n"
            f"GOLD CONTEXT: {scenario.get('gold_context')}\n"
            f"GOLD ANSWER: {scenario.get('gold_answer', '(none — judge by context+honesty)')}\n"
            f"AGENT ANSWER:\n{answer}\n\nAGENT TRAJECTORY (its tool calls, in order):\n"
            + "\n".join(trajectory) + "\n\nReturn ONLY the JSON verdict object.")


def _claude(prompt: str, extra: List[str]) -> subprocess.CompletedProcess:
    # subscription auth only — never the API-key-forcing flag, never an API-key env var.
    # (kept token-free so the no-API guard test can string-match the source.)
    return subprocess.run(["claude", "-p", prompt, *extra],
                          capture_output=True, text=True, timeout=300)


def probe() -> bool:
    """True if headless subscription `claude -p` runs here (Mode A available)."""
    try:
        p = _claude("Reply with the single word OK.", ["--max-turns", "1"])
        return p.returncode == 0 and "OK" in p.stdout
    except Exception:
        return False


def run_agent(scenario: dict) -> dict:
    """Mode A: run the agent-under-test headless; parse answer + trajectory from stream-json."""
    p = _claude(build_agent_prompt(scenario),
                ["--max-turns", "8", "--allowedTools", _ALLOWED,
                 "--append-system-prompt", _AGENT_SYSTEM,
                 "--output-format", "stream-json", "--verbose"])
    answer, trajectory = "", []
    for line in p.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "assistant":
            for c in ev.get("message", {}).get("content", []):
                if c.get("type") == "tool_use":
                    trajectory.append(json.dumps(c.get("input", {}))[:200])
        if ev.get("type") == "result":
            answer = ev.get("result", "")
    return {"id": scenario["id"], "answer": answer, "trajectory": trajectory}


def run_judge(scenario: dict, answer: str, trajectory: List[str]) -> dict:
    p = _claude(build_judge_prompt(scenario, answer, trajectory), ["--output-format", "json"])
    try:
        outer = json.loads(p.stdout)
        return json.loads(outer["result"]) if isinstance(outer, dict) else json.loads(p.stdout)
    except Exception:
        return {"verdict": "fail", "rationale": "judge output unparseable", "raw": p.stdout[:300]}


def aggregate(records: List[dict]) -> dict:
    verdicts = [r.get("verdict", {}).get("verdict", "fail") for r in records]
    return {"n": len(records),
            "pass": verdicts.count("pass"),
            "partial": verdicts.count("partial"),
            "fail": verdicts.count("fail")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.graph.agentic")
    ap.add_argument("--probe", action="store_true", help="check if headless subscription works here")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--mode-b", action="store_true",
                    help="aggregate from evals/graph/.runs/<id>.json (session-driven Mode B)")
    args = ap.parse_args(argv)

    if args.probe:
        print("headless subscription available:", probe())
        return 0

    scenarios = [s for s in load_scenarios() if args.scenario in (None, s["id"])]
    records = []
    if args.mode_b:                                   # read session-produced answers
        for s in scenarios:
            path = os.path.join(RUNS_DIR, f"{s['id']}.json")
            if not os.path.exists(path):
                continue
            rec = json.load(open(path, encoding="utf-8"))
            # Prefer a verdict the session-driven judge already wrote into the record; only fall back
            # to a headless judge (Mode A) when the record has none.
            if "verdict" not in rec:
                rec["verdict"] = (run_judge(s, rec["answer"], rec["trajectory"]) if probe()
                                  else {"verdict": "fail", "rationale": "no judge available"})
            records.append(rec)
    else:                                             # Mode A: fully headless
        if not probe():
            print("headless subscription unavailable — use Mode B (see evals/graph/AGENTIC.md)")
            return 0
        for s in scenarios:
            rec = run_agent(s)
            rec["verdict"] = run_judge(s, rec["answer"], rec["trajectory"])
            records.append(rec)
    print(json.dumps(aggregate(records), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
