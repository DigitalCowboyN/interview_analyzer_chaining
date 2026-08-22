# Durable agentic-fitness eval suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the lean 3-scenario proof into a durable, re-runnable agentic-fitness suite: a broad scenario corpus (bug-fix → refactor, governance, pipeline, deployment) with a solvable/partial/gap tracker, a deterministic Layer-1 scorecard, and a subscription-driven Layer-2 agentic harness (no API / usage-based billing).

**Architecture:** Extend `evals/graph/run.py` (Layer 1) to score by category + `expected` and emit a scorecard. Add ~13 new scenario JSONs (to the 3 existing) with hand-verified gold. Add `evals/graph/agentic.py` (Layer 2) whose agent-under-test / judge run on the **Claude Code subscription** — Mode A headless `claude -p` if it runs cleanly here, else Mode B (subagent-driven pre-commit routine). Add `evals/graph/RUBRIC.md`, regenerate `RESULTS.md`, wire a make target + pre-commit hook.

**Tech Stack:** Python 3 (stdlib: `json`, `subprocess`, `argparse`), pytest. The Claude Code subscription CLI (`claude -p`) for Mode A. No new Python deps. **No `ANTHROPIC_API_KEY`, no `--bare`, no API path — ever.**

**Spec:** `docs/superpowers/specs/2026-08-21-durable-eval-suite-design.md`. **ADRs:** none new.

## Global Constraints

- **Subscription only.** The harness MUST NOT set `ANTHROPIC_API_KEY`, pass `--bare`, or call any API/usage-metered endpoint. A test asserts the harness source contains no `--bare` and no `ANTHROPIC_API_KEY`.
- **Gold is hand-verified against the live graph.** Every scenario's `entry` and `gold_context` address must resolve to a real node (a test enforces no dangling gold). Gold is derived by *running* `gather_context`/`walk` during the build and trimming to necessary+sufficient — never guessed.
- **`gap`/`partial` scenarios scoring low is expected**, not a failure — the runner reports them as such; they document a missing capability and are the roadmap.
- **`evals/` is not code the graph ingests** — `discover_units` walks only `src/`/`tools/`, so scenarios/harness never become graph nodes.
- **Names verbatim:** `run_scenario(scenario, root=".")`, `scorecard(results)`, `agentic.py` with `run_agent(scenario)` / `run_judge(scenario, answer, trajectory)` / `--probe`.

---

### Task 1: Scenario corpus (schema + ~13 new scenarios + validity test)

**Files:**
- Create: `evals/graph/scenarios/*.json` (13 new — see the table)
- Modify: the 3 existing scenarios to add the `expected` + `level` fields (default `solvable`)
- Test: `tests/evals/test_scenarios.py`

**The corpus (ids + category/level/expected):** author one JSON per row. For each, set `task` (a generic goal, NO prescribed walk), `entry` (verified node address[es]), `gold_context` (verified), optional `gold_answer`, `expected`, and for partial/gap a `gap_note`.

| id | category | level | expected |
| --- | --- | --- | --- |
| fix-calls-resolution | bug-fix | symbol | solvable |
| fix-speaker-inference | bug-fix | module | solvable |
| add-enrichment-extractor | new-component | module | solvable |
| add-projection-handler | new-component | module | solvable |
| refactor-resolution-engine | refactor | subsystem | solvable |
| split-export-bundler | refactor | subsystem | solvable |
| govern-event-envelope | governance | module | solvable |
| govern-projection-service | governance | subsystem | solvable |
| govern-superseded-near-ingestion | governance | subsystem | partial |
| pipeline-ingestion-flow | pipeline | subsystem | gap |
| pipeline-write-path | pipeline | subsystem | gap |
| deploy-projection-service | deployment | subsystem | gap |
| deploy-service-topology | deployment | subsystem | gap |
| deploy-neo4j-schema-blast | deployment | subsystem | partial |

(The 3 existing — `explore-tools-graph`, `spec-code-intake`, `trace-classify-obligation` — get `expected: solvable` and a `level`, bringing the suite to 17.)

- [ ] **Step 1: Derive + verify gold for each scenario against the live graph.** For each `entry`, run the real graph to find the true neighborhood, then hand-trim to necessary+sufficient gold. Use this helper pattern (run per scenario during authoring, not committed):

```bash
python -c "
from tools.graph.traverse import gather_context, walk
sg = gather_context('code:resolution.engine', level='module')   # or walk(...) for broad ones
print('reached:', sorted(sg.nodes))
"
```
Record the capability/ADR/test/code addresses that genuinely belong in gold. For `gap` scenarios, set `gold_context` to what a *complete* answer would need (e.g. an ordered pipeline flow) and `gap_note` to what the graph can't reach — those are expected-missing.

- [ ] **Step 2: Write the scenario JSONs.** Example (`refactor-resolution-engine.json`) — fill the others the same way with verified addresses:

```json
{
  "id": "refactor-resolution-engine",
  "category": "refactor",
  "level": "subsystem",
  "task": "I need to refactor the entity-resolution engine. Gather the context I need: the code it spans, the decision that governs it, and the tests that pin its behavior.",
  "entry": ["code:resolution.engine"],
  "gold_context": ["code:resolution.engine", "code:resolution.candidates", "code:resolution.suggestions", "adr:11", "capabilities:resolve-entities-and-people"],
  "expected": "solvable",
  "source": "hand-authored, gold verified against live graph"
}
```

Example gap scenario (`pipeline-write-path.json`):

```json
{
  "id": "pipeline-write-path",
  "category": "pipeline",
  "level": "subsystem",
  "task": "A user submits a correction command. Trace what happens to it end to end — through the event store, the projection service, into the read model the API serves.",
  "entry": ["code:commands"],
  "gold_context": ["code:commands", "code:events", "code:projections.projection_service", "code:api"],
  "expected": "gap",
  "gap_note": "The command->event->projection->read-model flow is a RUNTIME/async path; the graph has no ordered flow edge across it (only static depends_on). Closes when the flow/architecture-nodes milestone lands.",
  "source": "hand-authored"
}
```

- [ ] **Step 3: Write the validity test** — `tests/evals/test_scenarios.py`:

```python
from evals.graph.run import load_scenarios
from tools.graph.reader import nodes
from tools.graph.registry import NODE_DOMAINS

_VALID_CATEGORY = {"bug-fix", "new-component", "refactor", "governance",
                   "pipeline", "deployment", "exploration", "meta", "spec", "implement"}
_VALID_EXPECTED = {"solvable", "partial", "gap"}


def _real_addresses():
    ns = nodes(".")
    return {f"{NODE_DOMAINS[t]}:{i}" for t, ids in ns.items() for i in ids}


def test_every_scenario_has_required_fields_and_valid_enums():
    for s in load_scenarios():
        assert s["id"] and s["task"] and s["entry"] and s["gold_context"]
        assert s["category"] in _VALID_CATEGORY, s["id"]
        assert s["expected"] in _VALID_EXPECTED, s["id"]
        if s["expected"] in ("partial", "gap"):
            assert s.get("gap_note"), f"{s['id']} missing gap_note"


def test_no_dangling_gold_addresses():
    real = _real_addresses()
    for s in load_scenarios():
        for addr in list(s["entry"]) + list(s["gold_context"]):
            assert addr in real, f"{s['id']}: gold address {addr} does not resolve"


def test_corpus_is_broad():
    cats = {s["category"] for s in load_scenarios()}
    assert {"bug-fix", "refactor", "governance", "pipeline", "deployment"} <= cats
```

- [ ] **Step 4: Run to verify** — `python -m pytest tests/evals/test_scenarios.py -q --no-cov`. If `test_no_dangling_gold_addresses` fails, fix the offending scenario's address to a real node (do not weaken the test). Expected: PASS once gold is correct.

- [ ] **Step 5: Commit**

```bash
git add evals/graph/scenarios tests/evals/test_scenarios.py
git commit -m "feat(evals): broad scenario corpus (17 scenarios, solvable/partial/gap tracker) + validity test"
```

---

### Task 2: Layer 1 runner — score by category/expected + scorecard + RESULTS

**Files:**
- Modify: `evals/graph/run.py` (use `gather_context`; aggregate by category + expected; `scorecard`)
- Test: `tests/evals/test_run.py` (extend)

- [ ] **Step 1: Extend `run_scenario`** to prefer `gather_context` (the minimal-context retrieval) but honor an explicit `direction`/`depth` when a scenario sets them:

```python
from tools.graph.traverse import Subgraph, walk, gather_context

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
```

(Note: scenarios use `level` for the *descriptive* grain in the table; the graph grain for
`walk`/`gather_context` is `module` by default. A scenario needing symbol-grain retrieval sets
`"level_grain": "symbol"` — kept distinct from the descriptive `level` to avoid a collision. The
original 3 scenarios keep their `direction`/`depth` and thus their exact prior behavior.)

- [ ] **Step 2: Add `scorecard`** — aggregate mean recall/coverage by category and by expected:

```python
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
```

- [ ] **Step 3: Extend `main`** to print the per-scenario table (existing), then the scorecard, and (with `--results`) write `evals/graph/RESULTS.md`. A `gap`/`partial` row is annotated `(expected low)` so a low score doesn't read as a regression.

- [ ] **Step 4: Extend the test** — `tests/evals/test_run.py`: `scorecard` on a fixture list computes correct per-category means; a `gap` scenario's low recall is still surfaced (not dropped). Keep the existing `score`/`substantive`/`load_scenarios` tests.

- [ ] **Step 5: Run + generate baseline** — `python -m pytest tests/evals -q --no-cov`; then `python -m evals.graph.run --results` writes the Layer-1 baseline into `RESULTS.md`. Sanity: a `solvable` control (`explore-tools-graph`) scores high; a `gap` (`pipeline-write-path`) scores low.

- [ ] **Step 6: Commit**

```bash
git add evals/graph/run.py tests/evals/test_run.py evals/graph/RESULTS.md
git commit -m "feat(evals): Layer-1 scorecard by category + expected (deterministic regression core)"
```

---

### Task 3: The judge rubric

**Files:**
- Create: `evals/graph/RUBRIC.md`

- [ ] **Step 1: Write `RUBRIC.md`** — the versioned, fixed-wording judge rubric per the spec: four dimensions (answer correctness / context sufficiency / trajectory quality / honesty), each scored + justified, with an **escape hatch** (a `gap` scenario passes when the agent correctly reports the graph can't answer) and **grade-the-goal-not-the-path** (trajectory graded on *shape* — coarse→walk-up→horizontal, efficient, progressive — not an exact path). Overall verdict `pass | partial | fail` + one-line rationale. Include the exact JSON shape the judge must emit:

```json
{"answer_correctness": 0-2, "context_sufficiency": 0-2, "trajectory_quality": 0-2,
 "honesty": 0-2, "verdict": "pass|partial|fail", "rationale": "one line"}
```

- [ ] **Step 2: Commit**

```bash
git add evals/graph/RUBRIC.md
git commit -m "docs(evals): judge rubric (outcome + trajectory + honesty, escape hatch, grade-goal-not-path)"
```

---

### Task 4: Layer 2 harness — subscription-driven (Mode A probe → Mode A or Mode B)

**Files:**
- Create: `evals/graph/agentic.py`
- Create: `evals/graph/AGENTIC.md` (the Mode B pre-commit routine procedure)
- Test: `tests/evals/test_agentic.py`

**Interfaces:**
- Produces: `build_agent_prompt(scenario) -> str`, `build_judge_prompt(scenario, answer, trajectory) -> str`, `run_agent(scenario) -> dict` (Mode A: headless subprocess), `run_judge(...) -> dict`, `probe() -> bool` (does headless subscription work here?), `aggregate(records) -> dict`.

- [ ] **Step 1: Write the no-API guard + prompt builders test first** — `tests/evals/test_agentic.py`:

```python
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
    # generic: teaches CLI syntax but must not prescribe a concrete recipe (no hardcoded depth)
    assert "decide your own strategy" in p.lower()
    import re
    assert not re.search(r"--depth\s+\d", p)


def test_judge_prompt_carries_gold_and_trajectory():
    scn = {"id": "x", "task": "t", "gold_context": ["adr:27"], "gold_answer": "A",
           "category": "meta", "expected": "solvable"}
    p = ag.build_judge_prompt(scn, answer="my answer", trajectory=["walk ..."])
    assert "adr:27" in p and "my answer" in p and "verdict" in p.lower()
```

- [ ] **Step 2: Implement `evals/graph/agentic.py`.** Prompt builders (pure, tested), a headless-subscription runner, a probe, and aggregation. **No `--bare`, no API key** anywhere:

```python
# evals/graph/agentic.py
"""Layer-2 agentic eval harness, driven by the Claude Code SUBSCRIPTION (never an API key).

Mode A: headless `claude -p` subprocess per scenario (agent-under-test + judge), subscription auth.
Mode B: when headless is unavailable here, the pre-commit routine in AGENTIC.md drives the same
scenarios via subagents inside a Claude Code session; this module's prompt builders + aggregate() are
reused, and answers/trajectories are read from evals/graph/.runs/<id>.json written by that routine.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List

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
    # subscription auth: NO --bare, NO ANTHROPIC_API_KEY. Fails cleanly if headless is unavailable.
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
            rec["verdict"] = run_judge(s, rec["answer"], rec["trajectory"]) \
                if probe() else rec.get("verdict", {"verdict": "fail", "rationale": "no judge"})
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
```

- [ ] **Step 3: Write `evals/graph/AGENTIC.md`** — the Mode B pre-commit routine: for each scenario, a Claude Code session dispatches a subagent with `build_agent_prompt(scenario)` and the validated WALK/CONTEXT tool-loop (session executes the graph CLI, returns results, loops to conclusion), writes `{id, answer, trajectory}` to `evals/graph/.runs/<id>.json`, then runs `python -m evals.graph.agentic --mode-b` to judge + aggregate. Document that `.runs/` is git-ignored scratch.

- [ ] **Step 4: Run tests + probe** — `python -m pytest tests/evals/test_agentic.py -q --no-cov` (PASS, incl. the no-API guard). Then `python -m evals.graph.agentic --probe` — record whether headless subscription is available here (decides Mode A vs Mode B for the baseline run in Task 5). Add `.runs/` to `.gitignore`.

- [ ] **Step 5: Commit**

```bash
git add evals/graph/agentic.py evals/graph/AGENTIC.md tests/evals/test_agentic.py .gitignore
git commit -m "feat(evals): Layer-2 subscription-driven agentic harness (Mode A headless / Mode B routine) + no-API guard"
```

---

### Task 5: Baseline agentic run + make target + pre-commit hook + gate

- [ ] **Step 1: Run the agentic baseline** on 2-3 representative scenarios (one `solvable` control, one `governance`, one `gap`). If `--probe` was true, run Mode A (`python -m evals.graph.agentic --scenario <id>`). If false, run Mode B: this session dispatches the subagent tool-loop per the AGENTIC.md procedure, writes `.runs/<id>.json`, then `--mode-b`. Record the verdicts.

- [ ] **Step 2: Write the results** into `evals/graph/RESULTS.md`: the Layer-1 scorecard (all 17, by category + expected) and the Layer-2 agentic baseline (the 2-3 judged scenarios, with the mode used). Note which `gap` scenarios scored low as expected and which milestone would lift them.

- [ ] **Step 3: Make target + pre-commit hook.** Add to the `Makefile`:

```make
.PHONY: eval-graph
eval-graph: ## Layer-1 deterministic eval scorecard (fast, deterministic)
	@$(PYTHON) -m evals.graph.run --results

.PHONY: eval-graph-agentic
eval-graph-agentic: ## Layer-2 agentic eval (subscription; Mode A headless, else see AGENTIC.md)
	@$(PYTHON) -m evals.graph.agentic --probe
```

Add a **pre-commit** entry (non-blocking, local) that runs `make eval-graph` (Layer 1 only — deterministic, fast) and prints the scorecard, so regressions are visible before commit. Layer 2 stays manual (`make eval-graph-agentic` / the AGENTIC.md routine).

- [ ] **Step 4: Full gate** — `python -m pytest tests/evals -q --no-cov` green; `make regen-derived && git diff --exit-code` clean (adding `evals/` + test files only drifts the testmap/graph indexes by the new test nodes — regenerate and include them); `make test-unit` green.

- [ ] **Step 5: Commit + final review**

```bash
git add evals/graph/RESULTS.md Makefile .pre-commit-config.yaml docs/tests/index.md docs/graph/index.md docs/graph/graph.md
git commit -m "feat(evals): agentic baseline + eval-graph make target + pre-commit scorecard"
```

Then dispatch the final whole-branch review (most capable model) with a review package (`scripts/review-package "$(git merge-base main HEAD)" HEAD`), and use **superpowers:finishing-a-development-branch**.

## After all tasks

- 17 scenarios across bug-fix/new-component/refactor/governance/pipeline/deployment, gold verified (no dangling), with a solvable/partial/gap tracker.
- `make eval-graph` prints the deterministic scorecard by category + expected; `RESULTS.md` holds the baseline; the `gap` scenarios score low as designed (the roadmap).
- The Layer-2 harness runs on the subscription (Mode A headless if available, else the Mode B routine) with a test guaranteeing no API/`--bare` path; an agentic baseline is recorded.
- Full suite green; freshness clean.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-21.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| evals (new) | yes | corpus + Layer-1 scorecard + Layer-2 subscription harness + rubric + results | the suite; measurement harness, not a guarded domain |
| graph / code | yes (read-only) | Layer 1 consumes `gather_context`/`walk`; Layer 2 drives the `walk`/`context` CLI | consumed, not changed |
| capabilities / adr / tests | no (logic) | referenced as scenario gold only | gold references |
| cli | no | reuses existing `walk`/`context` commands; no new graph command | — |

**Verdict:** reconciled — a durable, subscription-only agentic-fitness suite (no API/usage-based): deterministic scorecard + local agentic harness over a broad, gap-tracking scenario corpus. No graph-model change, no new ADR; CI and the gap-closing milestones deferred.
