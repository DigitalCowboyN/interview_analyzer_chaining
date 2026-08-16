# Docstring backlog + graph agentic evals — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Burn down the 61-module docstring backlog (so the graph's node context is real), and stand up a lean `evals/graph/` harness — 3 scenarios mined from real artifacts, a runner over `walk`, a deterministic lift metric, and one agentic proof — then report the before/after lift.

**Architecture:** `evals/graph/run.py` reuses `tools.graph.traverse.walk` to materialize each scenario's subgraph and scores it (context-coverage / recall / precision-overfetch) against a hand-verified gold set in `evals/graph/scenarios/*.json`. Docstrings are added to the 61 flagged modules. Everything is in git, so the before/after lift is computed by running the runner against the pre-backlog and post-backlog checkouts.

**Tech Stack:** Python 3 (stdlib: `json`, `argparse`), pytest. Reuses `tools.graph`. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-16-docstring-backlog-graph-evals-design.md`. **ADRs:** none new (evals are a measurement harness, not a decision; ADR-0016/0023 govern non-blocking tooling).

## Global Constraints

- **Reuse, don't rebuild:** the runner calls `tools.graph.traverse.walk` — it does not re-implement traversal or harvesting.
- **Pure, testable metric:** the scoring is a pure function `score(subgraph, scenario) -> dict` that takes a `Subgraph` + a scenario dict; `run_scenario` wraps `walk` + `score`. Unit-test `score` directly on a hand-built `Subgraph`.
- **No dangling gold:** every address in a scenario's `entry` / `gold_context` must resolve to a real node on the current graph. A gold address that does not resolve is a scenario bug (a test enforces this).
- **`evals/` is not code:** `discover_units` only walks `src/`/`tools/`, so `evals/` never becomes graph nodes and needs no docstring coverage itself. Evals are a measurement harness, not a guarded knowledge domain.
- **Docstring quality bar:** 1–3 sentences stating a module's **responsibility and role in its package** — purpose-first, not restating the code — matching existing `src/` docstring style. No boilerplate, no restating the filename.
- **Backlog done when:** `docs/code/docstring-backlog.md` lists **0 modules**, `check_missing_docstring` is silent, `make regen-derived && git diff` is CLEAN, and the full unit suite is green.
- **Names verbatim:** `score(subgraph, scenario)`, `run_scenario(scenario, root=".")`, `substantive(ctx, min_words=4)`, runner CLI `python -m evals.graph.run [--root PATH] [--scenario ID]`.

---

### Task 1: The eval harness (`evals/graph/`)

Build the runner, the metric, and the 3 mined scenario files. This must land before the backlog so the pre-backlog commit is the "before" baseline.

**Files:**
- Create: `evals/__init__.py`, `evals/graph/__init__.py` (empty — make `evals.graph` importable)
- Create: `evals/graph/run.py`
- Create: `evals/graph/scenarios/explore-tools-graph.json`
- Create: `evals/graph/scenarios/spec-code-intake.json`
- Create: `evals/graph/scenarios/trace-classify-obligation.json`
- Test: `tests/evals/__init__.py` (empty), `tests/evals/test_run.py`

**Interfaces:**
- Consumes: `tools.graph.traverse.walk`, `tools.graph.traverse.Subgraph`/`Node`.
- Produces: `score(subgraph, scenario) -> dict`, `run_scenario(scenario, root=".") -> dict`, `substantive(ctx, min_words=4) -> bool`, `load_scenarios() -> list[dict]` (reads the module-relative `scenarios/` dir — the same fixed set is scored against every `--root`, so before/after compare like for like).

- [ ] **Step 1: Write the scenario files.**

`evals/graph/scenarios/explore-tools-graph.json`:

```json
{
  "id": "explore-tools-graph",
  "category": "exploration",
  "task": "What does the tools.graph package and each of its modules do?",
  "entry": ["code:tools.graph"],
  "direction": "out",
  "depth": 1,
  "gold_context": [
    "code:tools.graph.__main__",
    "code:tools.graph.check",
    "code:tools.graph.classify",
    "code:tools.graph.reader",
    "code:tools.graph.registry",
    "code:tools.graph.render",
    "code:tools.graph.traverse"
  ],
  "gold_answer": "tools.graph builds and serves the cross-domain knowledge graph: registry (node/edge types), reader (harvest nodes+edges), traverse (walk), render (catalogs/mermaid), check (non-blocking guards), classify (derived category/determinism axes), __main__ (CLI).",
  "source": "manual — package module listing"
}
```

`evals/graph/scenarios/spec-code-intake.json`:

```json
{
  "id": "spec-code-intake",
  "category": "spec",
  "task": "Gather the minimal necessary context to spec the hierarchical-code-intake change (derive the code map from source, retire the overlay).",
  "entry": ["code:tools.code.reader", "code:tools.graph.reader"],
  "direction": "both",
  "depth": 2,
  "gold_context": [
    "code:tools.code.reader",
    "code:tools.graph.reader",
    "code:tools.graph.registry",
    "code:tools.graph.traverse",
    "code:tools.graph.classify",
    "capabilities:link-the-domains",
    "capabilities:map-the-code",
    "adr:19",
    "adr:20",
    "adr:24"
  ],
  "source": "PR #43 spec — docs/superpowers/specs/2026-08-16-hierarchical-code-intake-design.md. NOTE: adr:19/20/24 are in gold deliberately to expose that no `governs` edge links the KG-tooling ADRs to this code — recall < 1.0 is a real finding, not a scenario bug."
}
```

`evals/graph/scenarios/trace-classify-obligation.json`:

```json
{
  "id": "trace-classify-obligation",
  "category": "implement",
  "task": "What obligation does tools/graph/classify.py fulfill (which capability does it serve)?",
  "entry": ["code:tools.graph.classify"],
  "direction": "in",
  "depth": 2,
  "gold_context": ["capabilities:link-the-domains"],
  "gold_answer": "tools.graph.classify is contained by the tools.graph package, which is implemented_by the capability 'link-the-domains' — so classify fulfills that capability. (Walk up contained_by, then implemented_by.)",
  "source": "PR #43 — walk-up via contained_by then implemented_by"
}
```

- [ ] **Step 2: Write the failing test** — `tests/evals/__init__.py` (empty), then `tests/evals/test_run.py`:

```python
import json
import os

from tools.graph.traverse import Node, Subgraph
from evals.graph.run import load_scenarios, score, substantive


def _sg(nodes):
    # nodes: {address: context}
    return Subgraph(nodes={a: Node(address=a, type="", context=c) for a, c in nodes.items()}, edges=[])


def test_substantive_requires_real_words():
    assert not substantive("")
    assert not substantive("x y")               # under the word floor
    assert substantive("This module walks the graph.")


def test_score_context_coverage_and_recall():
    scenario = {"gold_context": ["code:a", "code:b", "capabilities:c"]}
    # a is reached WITH context, b reached WITHOUT, c not reached at all
    sg = _sg({"code:a": "Does a real thing here.", "code:b": "", "code:x": "extra reached node."})
    s = score(sg, scenario)
    assert s["coverage"] == 0.5                  # of {a,b} code gold, only a has substantive context
    assert round(s["recall"], 3) == round(2 / 3, 3)  # a,b reached of {a,b,c}
    assert s["overfetch"] == 1                   # code:x reached but not gold
    assert s["missing"] == ["capabilities:c"]


def test_load_scenarios_reads_all():
    # load_scenarios just parses the scenario dir; validity-on-real-graph is checked separately
    ids = {s["id"] for s in load_scenarios()}
    assert {"explore-tools-graph", "spec-code-intake", "trace-classify-obligation"} <= ids


def test_gold_addresses_resolve_on_the_real_graph():
    # every entry + gold_context address must be a real node — no dangling gold
    from tools.graph.reader import nodes
    from tools.graph.registry import NODE_DOMAINS
    slug_type = {v: k for k, v in NODE_DOMAINS.items()}
    ns = nodes(".")
    real = {f"{NODE_DOMAINS[t]}:{i}" for t, ids in ns.items() for i in ids}
    for s in load_scenarios():
        for addr in list(s["entry"]) + list(s["gold_context"]):
            assert addr in real, f"{s['id']}: gold address {addr} does not resolve"
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/evals/test_run.py -q --no-cov`
Expected: FAIL — `No module named 'evals.graph.run'`.

- [ ] **Step 4: Implement** — create the empty `evals/__init__.py`, `evals/graph/__init__.py`, `tests/evals/__init__.py`, then `evals/graph/run.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest tests/evals/test_run.py -q --no-cov`
Expected: PASS (4 passed). If `test_gold_addresses_resolve_on_the_real_graph` fails, a gold address is wrong — fix the scenario file to match the live graph (do NOT weaken the test).

- [ ] **Step 6: Smoke-run the runner on the real repo (pre-backlog baseline)**

Run: `python -m evals.graph.run`
Expected: a 3-row table. `explore-tools-graph` coverage ≈ `0.00` (docstrings not written yet); `spec-code-intake` recall < 1.0 with `adr:19/20/24` in missing (the govern-edge gap) and a large `overfetch`; `trace-classify-obligation` recall `1.00`. Note these numbers — they are the "before".

- [ ] **Step 7: Lint + commit**

```bash
python -m flake8 evals/graph/run.py tests/evals/test_run.py
git add evals/ tests/evals/
git commit -m "feat(evals): lean graph agentic-fitness eval harness (3 mined scenarios + runner)"
```

Record this commit SHA — it is the **pre-backlog "before"** reference for Task 3.

---

### Task 2: Burn down the docstring backlog (61 modules)

Add purpose-first module docstrings to every module in `docs/code/docstring-backlog.md`, package by package, until the backlog is empty.

**Files:** the 61 modules listed in `docs/code/docstring-backlog.md` (nearly all under `tools/*/`, plus `src/utils/text_processing.py`). No test files change; the regenerated `docs/code/docstring-backlog.md` + `docs/code/index.md` are the evidence.

**Approach (controller):** work one package at a time (the backlog's grouping). For each package, dispatch a subagent that reads each listed module and adds a module docstring meeting the quality bar (it only edits files — no pyenv needed). After each package (or a small batch), the controller runs `make code-index` (the backlog shrinks) and `python -m pytest tests/code -q --no-cov` (stays green), then commits. The packages:

`tools.adr, tools.api, tools.capability, tools.cli, tools.code, tools.corpus, tools.glossary, tools.graph, tools.graphq, tools.knowledge, tools.prompts, tools.testmap, tools.usecase`, plus the top-level `src/utils/text_processing.py`.

- [ ] **Step 1: Confirm the worklist**

Run: `cat docs/code/docstring-backlog.md`
Expected: ~61 modules grouped by package. This is the authoritative list; work top to bottom.

- [ ] **Step 2: Document each package (repeat per package)**

For each package `P` in the list, dispatch an implementer subagent with this contract:
- Read every module file under `P` that appears in `docs/code/docstring-backlog.md`.
- Add a module docstring (triple-quoted, first statement in the file) of 1–3 sentences stating the module's **responsibility and role in its package** — purpose-first, matching `src/` docstring style. Do not restate the code or the filename. Do not touch anything else.
- The subagent edits files only; it must not run tests, make, or git.

Example (the quality bar), `tools/graph/traverse.py`:

```python
"""Ephemeral graph traversal. walk(entry, direction, depth) rebuilds the reachable subgraph
from source on each call — the LLM working-context substrate (ADR-0025) — resolving each node's
claim/context via the per-type _CONTEXT table. Separate from the transcript Neo4j graph."""
```

- [ ] **Step 3: After each package — regenerate, verify, commit**

```bash
make code-index                                   # backlog shrinks; index.md context columns fill
python -m pytest tests/code -q --no-cov           # stays green (docstrings don't change logic)
python -m flake8 tools/<pkg>                       # docstrings are lint-clean
git add tools/<pkg> docs/code/index.md docs/code/pipeline.md docs/code/docstring-backlog.md
git commit -m "docs(code): module docstrings for <pkg> (docstring backlog)"
```

(Committing per package keeps each change reviewable and the ledger honest. `docs/code/pipeline.md` only changes if a module's presence changed — usually just index.md + backlog.)

- [ ] **Step 4: Confirm the backlog is empty**

Run: `make code-index && cat docs/code/docstring-backlog.md`
Expected: **0 module(s) remaining** (only the header remains).
Run: `python -m tools.code check`
Expected: exit 0, `code-check: clean` (no `has no docstring` findings).
Run: `make regen-derived && git diff --exit-code`
Expected: CLEAN.

- [ ] **Step 5: Full suite green**

Run: `make test-unit`
Expected: green (docstrings are inert to logic; only the generated catalogs changed).

Record the final backlog-complete commit SHA — it is the **post-backlog "after"** reference for Task 3.

---

### Task 3: Before/after lift + agentic proof + results write-up

Run the runner against the before and after checkouts, run the one agentic proof, and write up the numbers.

**Files:**
- Create: `evals/graph/RESULTS.md`

- [ ] **Step 1: Run the deterministic before/after.** Use a throwaway worktree for the "before" so the main checkout is untouched:

```bash
BEFORE=<Task 1 commit SHA>
git worktree add /tmp/eval-before "$BEFORE"
echo "=== BEFORE ==="; python -m evals.graph.run --root /tmp/eval-before
echo "=== AFTER  ==="; python -m evals.graph.run --root .
git worktree remove /tmp/eval-before
```

Expected: `explore-tools-graph` coverage rises from ≈0.00 → ≈1.00; `spec-code-intake` coverage rises (its reached code nodes are now documented) while recall/overfetch are unchanged (structure, not context); `trace-classify-obligation` unchanged (docstring-neutral). Capture both tables.

- [ ] **Step 2: Run the agentic proof (scenario 1, graph-context-only).** The main session materializes the graph's view and hands it — and nothing else — to a subagent, before and after:

For each of BEFORE (`--root /tmp/eval-before`, re-add the worktree) and AFTER (`--root .`), produce the walk view:

```bash
python -c "from tools.graph.traverse import walk; sg=walk('code:tools.graph','out',1,root='<root>'); [print(a,'::',sg.nodes[a].context or '<empty>') for a in sorted(sg.nodes)]"
```

Dispatch a subagent with ONLY that text as context and the task: *"From this graph context alone — you may not read any source file — summarize what the tools.graph package and each of its modules does. If the context is insufficient, say so explicitly."* Do this for the BEFORE view and the AFTER view. Expected: BEFORE → the subagent reports the context is empty/insufficient (it cannot summarize the modules); AFTER → it produces an accurate per-module summary from the graph alone. Capture both answers.

- [ ] **Step 3: Write `evals/graph/RESULTS.md`** — the before/after metric table (all 3 scenarios), the two agentic-proof answers (quoted), and a short findings section noting: (a) the docstring lift (S1 0→100% coverage), (b) the govern-edge gap surfaced by S2 (`adr:19/20/24` unreachable — the KG-tooling ADRs lack `governs` edges to their code), and (c) the S2 over-fetch (depth-2 both-direction walk pulls ~53 nodes — "minimal context" wants a tighter walk). Frame (b) and (c) as inputs to the future full-framework milestone, not defects here.

- [ ] **Step 4: Commit**

```bash
git add evals/graph/RESULTS.md
git commit -m "docs(evals): before/after lift results + agentic proof for the docstring backlog"
```

---

## After all tasks

- `docs/code/docstring-backlog.md` is empty; `check_missing_docstring` silent; `make regen-derived && git diff` clean; `make test-unit` green.
- `python -m evals.graph.run` reports the 3 scenarios; `tests/evals/` green.
- `evals/graph/RESULTS.md` shows S1 context coverage 0→100% and records the S2 structural findings.
- Run the final whole-branch review on the most capable model, then use **superpowers:finishing-a-development-branch**.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-16.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| code | yes | 61 module docstrings added (node context); `docstring-backlog.md` → 0 | the backlog subject |
| graph | yes (read-only) | the runner consumes `traverse.walk`; no graph-logic change | evals read the graph |
| evals (new) | yes | new `evals/graph/` measurement harness (scenarios + runner + results) | not a guarded domain; not scanned as code |
| capabilities / adr / tests | no (logic) | referenced as scenario gold only | gold references only |

**Verdict:** reconciled — code gains real node context (docstrings) and the graph gains its first agentic-fitness evals (a git-reproducible before/after harness). No new ADR; evals are a measurement harness distinct from the deterministic guards. The full framework, a `make evals` target/CI gate, more scenarios, transcript-graph evals, and symbols are deferred.
