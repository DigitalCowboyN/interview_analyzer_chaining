# Durable agentic-fitness eval suite (design)

**Status:** proposed (brainstorm dialogue with owner, 2026-08-21).
**Program:** the first-class knowledge graph
(`docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`). Turns the lean
3-scenario proof (`evals/graph/`, from PR #44) and the validated agent-driven eval *method* (the
2-eval redo before PR #46) into a **durable, re-runnable suite** that tracks the graph's fitness for
real agentic work over time.

## Purpose

Measure — repeatably — whether an agent can use the knowledge graph to gather the correct, minimal
context for real development tasks, across a broad spread (bug-fix → new component → large refactor,
governance, pipeline, deployment). The suite is a **regression tracker** (catch when a change
degrades usability) *and* a **roadmap** (scenarios the graph can't answer yet are marked and their
scores rise when a future milestone closes the gap).

## Two layers

- **Layer 1 — deterministic context eval (the fast regression core).** For each scenario, run
  `gather_context`/`walk` and score whether the graph *surfaces the gold context*: **recall** (gold
  nodes reached), **minimality** (over-fetch vs gold), and **answer-reach** (the answer node is on
  the path). Pure, fast, deterministic — run it every change. Extends today's `evals/graph/run.py`.
- **Layer 2 — full agentic eval, purpose-built on Claude Code (owner decision, 2026-08-21).** For
  each scenario, a **headless `claude` agent** (the agent-under-test) is given the generic goal and
  **only the graph CLI** as a tool; it drives its own exploration; a second headless **`claude`
  judge** scores its answer + trajectory against a rubric + the reference solution. **Runs locally /
  pre-commit, not CI** (it spawns real agents; non-deterministic, costs tokens).

## Scenario schema

`evals/graph/scenarios/<id>.json`:

```json
{
  "id": "refactor-resolution-engine",
  "category": "bug-fix | new-component | refactor | governance | pipeline | deployment | exploration | meta",
  "level": "symbol | module | subsystem",
  "task": "generic goal — what a human/agent wants, NO prescribed walk steps",
  "entry": ["code:resolution.engine"],
  "gold_context": ["code:resolution.*", "adr:11", "tests:...", "capabilities:resolve-entities-and-people"],
  "gold_answer": "optional — for answer-checkable tasks",
  "expected": "solvable | partial | gap",
  "gap_note": "for partial/gap: what the graph can't reach yet and which milestone would fix it",
  "source": "mined from PR #.. / hand-authored"
}
```

`expected` is the **gap tracker**: a `gap` scenario scoring low is expected (it documents a missing
capability); when a future milestone lands the score rises and the scenario is re-marked `solvable`.
Per eval best-practice, the suite deliberately includes can't-do cases — an all-pass suite is too easy.

## The scenario corpus (v1 — ~16, across the owner's categories)

Reference solutions (`gold_context`/`gold_answer`) are **hand-verified against the live graph** during
the build, and each `entry`/`gold` address must resolve (a test enforces no dangling gold).

| id | category / level | task | expected |
| --- | --- | --- | --- |
| fix-calls-resolution | bug-fix / symbol | fix a bug in how `calls_of` resolves imported calls | solvable |
| fix-speaker-inference | bug-fix / module | fix speaker inference for a messy transcript | solvable |
| add-enrichment-extractor | new-component / module | add a new enrichment extractor (lens dimension) | solvable |
| add-projection-handler | new-component / module | add a projection handler for a new event type | solvable |
| refactor-resolution-engine | refactor / subsystem | refactor the entity-resolution engine | solvable-broad |
| split-export-bundler | refactor / subsystem | split the export bundler into smaller modules | solvable |
| govern-event-envelope | governance | am I violating a decision by changing `events.envelope`? | solvable |
| govern-projection-service | governance | which ADR governs the projection service; still current? | solvable |
| govern-superseded-near-ingestion | governance | is a superseded decision in play near ingestion? | partial |
| pipeline-ingestion-flow | pipeline | trace the ingestion→analysis flow for a transcript | **gap** (runtime order not modeled) |
| pipeline-write-path | pipeline | what happens to a command after submit (event→projection→read)? | **gap** (async seam) |
| deploy-projection-service | deployment | what does the projection-service container need to run? | **gap** (infra not in graph) |
| deploy-service-topology | deployment | which services must be up for the API to serve reads? | **gap** |
| deploy-neo4j-schema-blast | deployment | what breaks downstream if the Neo4j schema changes? | partial |
| explore-ask-subsystem | exploration | what does the ask/retrieval subsystem do, and what serves it? | solvable (control) |
| meta-unverified-capabilities | meta | how many capabilities have no verifying test? | solvable (control) |

The `gap` scenarios (pipeline runtime flow, deployment topology/infra) map exactly onto the deferred
**flow/architecture-nodes** and **infra-modeling** milestones — the suite will show their scores rise
when those land.

## Layer 1 — deterministic runner (`evals/graph/run.py`, extended)

- Loads all scenarios; per scenario runs `gather_context(entry, level)` (and/or `walk` per the
  scenario's needs) and computes: `recall` = |gold ∩ reached| / |gold|; `coverage` = fraction of gold
  **code** nodes reached *with substantive context*; `overfetch` = |reached \ gold|; `answer_reached`
  = the `gold_answer` node (if any) is present.
- Prints a **scorecard**: per scenario + aggregated **by category** and **by `expected`**
  (solvable/partial/gap). Writes `evals/graph/RESULTS.md`. A `gap` scenario is reported as *expected
  low* (not a failure).
- CI-safe and fast (no agents). This is the durable regression number.

## Layer 2 — agentic harness (`evals/graph/agentic.py`, new; Claude-Code-native)

A script the owner runs locally. Per scenario:

1. **Agent-under-test** — headless `claude`, isolated and tool-restricted so it can *only* query the
   graph (it cannot read source files — the eval measures the graph, not the agent's file reading):

   ```bash
   claude -p "<scenario.task + how to address nodes + the walk/context tool>" \
     --model claude-sonnet-5 --max-turns 8 --bare \
     --allowedTools "Bash(python -m tools.graph walk:*),Bash(python -m tools.graph context:*)" \
     --append-system-prompt "You are an eval agent. Explore ONLY via the graph CLI. Start coarse, \
        expand progressively. If the graph cannot answer, say so explicitly (do not infer relevance \
        from proximity). End with a clear answer." \
     --output-format stream-json --verbose
   ```
   The harness parses the stream for `tool_use` events (the **trajectory** — which `walk`/`context`
   commands, in order) and the final `result` (the **answer**).

2. **Judge** — a second headless `claude` (most capable model), given the scenario's task,
   `gold_context`/`gold_answer`, the agent's answer, and its trajectory; returns a JSON verdict:

   ```bash
   claude -p "<rubric + scenario gold + agent answer + agent trajectory>" \
     --model claude-opus-4-8 --bare --output-format json
   ```

3. **Aggregate** — collect verdicts into the scorecard alongside the Layer-1 numbers.

**Isolation/auth notes:** `--bare` skips this repo's hooks/skills/CLAUDE.md so the agent isn't fed
repo docs (clean measurement); `--allowedTools` scoped to the graph CLI enforces graph-only
exploration. The build verifies the exact flags/auth locally (subscription vs `ANTHROPIC_API_KEY`)
and the exact model IDs.

## The judge rubric (`evals/graph/RUBRIC.md`)

Independent LLM-as-judge, one verdict per scenario, with an **escape hatch** and **grade-the-goal-not-
the-path** discipline (per Anthropic's agent-eval guidance). Dimensions, each scored + justified:

- **Answer correctness** — does the agent's answer match the reference (`gold_answer`), or for a
  `gap` scenario, does it *correctly report the graph can't answer* (honesty)?
- **Context sufficiency** — did the trajectory reach the `gold_context` (recall)?
- **Trajectory quality** — did it explore *progressively* (coarse → walk-up-to-intent → horizontal),
  efficiently, without prescribing? (grade the shape, not an exact path.)
- **Honesty** — did it use the escape hatch when appropriate rather than hallucinating governance/
  relevance from proximity?

Overall verdict: `pass | partial | fail` + a one-line rationale. The rubric is versioned and its
wording is fixed so runs are comparable.

## Scope

**This milestone:** the ~16-scenario corpus (schema + hand-verified gold + gap flags); Layer 1
deterministic runner + scorecard by category/expected; Layer 2 agentic harness on the `claude` CLI
(agent-under-test + judge) + the rubric; a `RESULTS.md` scorecard; one full local run recorded as the
baseline. Makefile target(s) to run each layer locally.

**Deferred:** CI integration (deliberately local); auto-mining scenarios from PR history; the
flow/architecture-nodes and infra-modeling that would flip the `gap` scenarios to `solvable` (own
milestones); a golden-answer human-calibration pass on the judge.

## Testing

- **Scenario validity:** every `entry`/`gold_context` address resolves on the live graph (no dangling
  gold); each scenario JSON parses and has the required fields + a valid `category`/`expected`.
- **Layer 1 runner:** `score` computes recall/coverage/overfetch correctly on a fixture; on the real
  repo it runs all scenarios and a `solvable` control (e.g. `explore-ask-subsystem`) scores high while
  a `gap` scenario (e.g. `pipeline-write-path`) scores low — both as expected.
- **Layer 2 harness:** the `claude`-CLI invocation is exercised on at least one scenario end-to-end
  (agent → trajectory captured → judge → JSON verdict parsed); the script is resilient to a failed/
  empty agent run (records a `fail`, never crashes the suite). Full local baseline run recorded.
- **Freshness/regression:** adding `evals/` files doesn't disturb generated indexes beyond the new
  test nodes; full unit suite green.

## ADR

No new ADR — the suite is measurement tooling under the existing program (ADR-0016/0023 govern
non-blocking visibility; the eval *method* is grade-the-goal-not-path per the earlier research). It
does not change the graph model.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-21.

| domain | touched? | note |
| --- | --- | --- |
| evals (new area) | yes — scenario corpus + Layer-1 runner + Layer-2 `claude`-CLI harness + rubric + results | the suite; a measurement harness, not a guarded domain |
| graph / code | yes (read-only) — Layer 1 consumes `gather_context`/`walk`; Layer 2 drives the `walk`/`context` CLI | consumed, not changed |
| capabilities / adr | no (logic) — referenced as scenario gold only | gold references |

**Verdict:** reconciled — a durable, Claude-Code-native agentic-fitness suite: a deterministic
regression core + a full local agentic harness over a broad scenario corpus that doubles as a gap
roadmap. No graph-model change, no new ADR; CI integration and the gap-closing milestones are deferred.
