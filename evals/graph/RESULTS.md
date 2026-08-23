# Graph agentic-fitness eval — Layer 1 (deterministic) scorecard

Run `make eval-graph` (or `python -m evals.graph.run --results`) to refresh.
`gap`/`partial` scenarios are *expected* to score low — they document the roadmap (flow/architecture nodes, infra modeling).

```
scenario                       cat          exp      recall cover over
----------------------------------------------------------------------
fix-calls-resolution           bug-fix      solvable   1.00  1.00   10
fix-speaker-inference          bug-fix      solvable   1.00  1.00    8
deploy-neo4j-schema-blast      deployment   partial    0.75  0.67   21 (expected low)
deploy-projection-service      deployment   gap        0.50  0.50    3 (expected low)
deploy-service-topology        deployment   gap        0.33  0.00   44 (expected low)
explore-tools-graph            exploration  solvable   1.00  1.00    2
govern-event-envelope          governance   solvable   1.00  1.00   27
govern-projection-service      governance   solvable   1.00  1.00    2
govern-superseded-near-ingestion governance   partial    1.00  1.00   22 (expected low)
trace-classify-obligation      implement    solvable   1.00     —   27
add-enrichment-extractor       new-component solvable   1.00  1.00    2
add-projection-handler         new-component solvable   1.00  1.00    2
pipeline-ingestion-flow        pipeline     gap        0.25  0.25   23 (expected low)
pipeline-write-path            pipeline     gap        0.25  0.25   13 (expected low)
refactor-resolution-engine     refactor     solvable   1.00  1.00   14
split-export-bundler           refactor     solvable   1.00  1.00    9
spec-code-intake               spec         solvable   0.80  1.00   81

By category (mean recall / coverage):
  bug-fix        recall=1.00  coverage=1.00  n=2
  deployment     recall=0.53  coverage=0.39  n=3
  exploration    recall=1.00  coverage=1.00  n=1
  governance     recall=1.00  coverage=1.00  n=3
  implement      recall=1.00  coverage=—  n=1
  new-component  recall=1.00  coverage=1.00  n=2
  pipeline       recall=0.25  coverage=0.25  n=2
  refactor       recall=1.00  coverage=1.00  n=2
  spec           recall=0.80  coverage=1.00  n=1

By expected (mean recall) — gap/partial are meant to be low (the roadmap):
  gap        recall=0.33  n=4
  partial    recall=0.88  n=2
  solvable   recall=0.98  n=11
```

## Layer 2 (agentic) — baseline

**Mechanism (finding):** the headless-subscription probe (`--probe`) returns `True` for a trivial call,
but a full multi-turn tool-using `claude -p` agent nested inside a driving session did **not** complete
reliably here. Per the owner's constraint (subscription only, "if headless doesn't work, pre-commit
routine here"), **Mode B — the subagent-driven routine (`AGENTIC.md`)** — is the shipping path. It uses
subagents (subscription-backed) as the agent-under-test and the judge; `agentic.py`'s prompt builders,
`.runs/` records, and rubric are shared. Mode A stays wired for environments where headless runs cleanly.

The first proof was `govern-projection-service` (pass 2/2/2/2). The **full 17-scenario baseline (KG-1)**
is below; in it, Mode B runs autonomous subagents that drive the graph CLI themselves (they gained
`python -m tools.graph` access), so no interactive relay is needed — one dispatch per agent + one judge.
## Layer 2 (agentic) — full baseline (KG-1)

Run 2026-08-23 via **Mode B autonomous subagents** (agent-under-test drives the graph CLI itself; judge scores against RUBRIC.md). 16 of 17 scenarios completed; `spec-code-intake` was truncated by a session limit and is pending a re-run (not counted). **Every completed scenario passed** — and every `gap`/`partial` scenario passed by *honestly reporting the graph's limitation*, never by fabricating.

| scenario | category | expected | verdict |
| --- | --- | --- | --- |
| add-enrichment-extractor | new-component | solvable | **pass** |
| add-projection-handler | new-component | solvable | **pass** |
| deploy-neo4j-schema-blast | deployment | partial | **pass** |
| deploy-projection-service | deployment | gap | **pass** |
| deploy-service-topology | deployment | gap | **pass** |
| explore-tools-graph | exploration | solvable | **pass** |
| fix-calls-resolution | bug-fix | solvable | **pass** |
| fix-speaker-inference | bug-fix | solvable | **pass** |
| govern-event-envelope | governance | solvable | **pass** |
| govern-projection-service | governance | solvable | **pass** |
| govern-superseded-near-ingestion | governance | partial | **pass** |
| pipeline-ingestion-flow | pipeline | gap | **pass** |
| pipeline-write-path | pipeline | gap | **pass** |
| refactor-resolution-engine | refactor | solvable | **pass** |
| spec-code-intake | spec | solvable | incomplete |
| split-export-bundler | refactor | solvable | **pass** |
| trace-classify-obligation | implement | solvable | **pass** |

**By expected:**
- solvable: 10/11 pass
- partial: 2/2 pass
- gap: 4/4 pass

**Highlights (honesty / escape-hatch on gap+partial):**
- `pipeline-write-path` / `pipeline-ingestion-flow` (gap): correctly reported the graph has no runtime data-flow edge across the event-sourced choreography — one distinguished the single real edge (ingestion→enrichment) from the absent ones and explained the store+projection indirection.
- `deploy-service-topology` / `deploy-projection-service` (gap): stated the graph models no service/container topology, then gave a *labeled* static-import inference (Neo4j, EventStoreDB) without fabricating env vars/ports/compose.
- `govern-superseded-near-ingestion` (partial): found the governing ADR, then ran a **control check** and honestly reported supersession status is indeterminate from the traversal.
- `deploy-neo4j-schema-blast` (partial): full write-side blast radius; honest that the read-side is only partially recoverable (label-string match, no edge) — refused to fabricate.

**Method caveats:** Mode-B autonomous subagents aren't tool-restricted like Mode A's `--allowedTools`, so isolation is prompt-enforced ("graph CLI only, no file reads") + judge-verified; one run (`explore-tools-graph`) read graph-CLI *output* via a scratchpad file (not source), reflected in its trajectory score. Three verdicts (pipeline-ingestion-flow, deploy-neo4j-schema-blast, trace-classify-obligation) are controller-judged against the fixed rubric because the judge-subagents hit the same session limit; the rest are judge-subagent (Opus) verdicts.

**Eval-surfaced graph findings (roadmap backlog):** supersede/superseded_by ADR→ADR edges don't appear to be surfaced by walk/context/neighbors; and the Neo4j-schema→read-consumer link is only a label-string match, not a graph edge (a 'reads-shape-of' edge type would make schema blast-radius traversable).
