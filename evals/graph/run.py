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

from tools.graph.traverse import Subgraph, walk, gather_context

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
    level = scenario.get("level_grain", "module")   # graph grain, not the descriptive 'level'
    # Back-compat: a scenario with an explicit direction/depth is a walk scenario (the original 3);
    # everything else uses gather_context (minimal-context retrieval) from the first entry.
    if "direction" in scenario or "depth" in scenario:
        sg = walk(list(scenario["entry"]), direction=scenario.get("direction", "both"),
                  depth=scenario.get("depth"), root=root, level=level)
    else:
        sg = gather_context(list(scenario["entry"])[0], root=root, level=level)
    r = score(sg, scenario)
    r["id"] = scenario["id"]
    r["category"] = scenario["category"]
    r["expected"] = scenario["expected"]
    return r


def scorecard(results: List[Dict]) -> Dict:
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None
    by_cat, by_exp = {}, {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
        by_exp.setdefault(r["expected"], []).append(r)
    return {
        "by_category": {c: {"recall": _mean([r["recall"] for r in rs]),
                            "coverage": _mean([r["coverage"] for r in rs]),
                            "n": len(rs)} for c, rs in sorted(by_cat.items())},
        "by_expected": {e: {"recall": _mean([r["recall"] for r in rs]), "n": len(rs)}
                        for e, rs in sorted(by_exp.items())},
    }


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def _render(results: List[Dict]) -> str:
    lines = [f"{'scenario':30} {'cat':12} {'exp':8} {'recall':>6} {'cover':>5} {'over':>4}"]
    lines.append("-" * 70)
    for r in sorted(results, key=lambda r: (r["category"], r["id"])):
        flag = " (expected low)" if r["expected"] in ("gap", "partial") else ""
        lines.append(f"{r['id']:30} {r['category']:12} {r['expected']:8} "
                     f"{_fmt(r['recall']):>6} {_fmt(r['coverage']):>5} {_fmt(r['overfetch']):>4}{flag}")
    sc = scorecard(results)
    lines.append("\nBy category (mean recall / coverage):")
    for c, v in sc["by_category"].items():
        lines.append(f"  {c:14} recall={_fmt(v['recall'])}  coverage={_fmt(v['coverage'])}  n={v['n']}")
    lines.append("\nBy expected (mean recall) — gap/partial are meant to be low (the roadmap):")
    for e, v in sc["by_expected"].items():
        lines.append(f"  {e:10} recall={_fmt(v['recall'])}  n={v['n']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.graph.run")
    ap.add_argument("--root", default=".", help="repo checkout to evaluate (for before/after)")
    ap.add_argument("--scenario", default=None, help="run only this scenario id")
    ap.add_argument("--results", action="store_true", help="write evals/graph/RESULTS.md")
    args = ap.parse_args(argv)

    scenarios = [s for s in load_scenarios() if args.scenario in (None, s["id"])]
    results = [run_scenario(s, args.root) for s in scenarios]
    table = _render(results)
    print(table)
    if args.results:
        path = os.path.join(os.path.dirname(__file__), "RESULTS.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# Graph agentic-fitness eval — Layer 1 (deterministic) scorecard\n\n"
                     "Run `make eval-graph` (or `python -m evals.graph.run --results`) to refresh.\n"
                     "`gap`/`partial` scenarios are *expected* to score low — they document the "
                     "roadmap (flow/architecture nodes, infra modeling).\n\n```\n" + table + "\n```\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
