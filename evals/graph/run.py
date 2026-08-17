# evals/graph/run.py
"""Lean agentic-fitness eval runner for the knowledge graph.

Each scenario (evals/graph/scenarios/*.json) is a real agentic task with a hand-verified gold
context set. `walk` materializes the subgraph an agent would see; `score` measures whether the
graph surfaces the right, small, well-described context. Run before/after a change to see the lift.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

from tools.graph.traverse import Subgraph, walk

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")


def substantive(ctx: str, min_words: int = 4) -> bool:
    """A node's context is usable only if it is present and more than a couple of words."""
    return bool(ctx) and len(ctx.split()) >= min_words


def load_scenarios() -> List[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(SCENARIO_DIR, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def score(subgraph: Subgraph, scenario: dict) -> Dict:
    reached = set(subgraph.nodes)
    gold = list(scenario.get("gold_context", []))
    gold_set = set(gold)
    gold_code = [a for a in gold if a.startswith("code:")]
    covered = [a for a in gold_code
               if a in reached and substantive(subgraph.nodes[a].context)]
    reached_gold = gold_set & reached
    return {
        "coverage": (len(covered) / len(gold_code)) if gold_code else None,
        "recall": (len(reached_gold) / len(gold_set)) if gold_set else None,
        "precision": (len(reached_gold) / len(reached)) if reached else None,
        "overfetch": len(reached - gold_set),
        "reached": len(reached),
        "gold": len(gold_set),
        "missing": sorted(gold_set - reached),
    }


def run_scenario(scenario: dict, root: str = ".") -> Dict:
    sg = walk(list(scenario["entry"]), direction=scenario.get("direction", "both"),
              depth=scenario.get("depth"), root=root)
    return score(sg, scenario)


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.graph.run")
    ap.add_argument("--root", default=".", help="repo checkout to evaluate (for before/after)")
    ap.add_argument("--scenario", default=None, help="run only this scenario id")
    args = ap.parse_args(argv)

    scenarios = [s for s in load_scenarios()
                 if args.scenario in (None, s["id"])]
    print(f"{'scenario':28} {'cat':11} {'cover':>5} {'recall':>6} {'prec':>5} {'over':>4} {'missing':>7}")
    for s in scenarios:
        r = run_scenario(s, args.root)
        print(f"{s['id']:28} {s['category']:11} {_fmt(r['coverage']):>5} "
              f"{_fmt(r['recall']):>6} {_fmt(r['precision']):>5} {_fmt(r['overfetch']):>4} "
              f"{len(r['missing']):>7}")
        if r["missing"]:
            print(f"    missing: {', '.join(r['missing'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
