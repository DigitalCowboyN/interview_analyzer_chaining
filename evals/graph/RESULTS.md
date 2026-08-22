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

**Baseline run (Mode B):**

| scenario | category | expected | verdict | dims (ans/ctx/traj/hon) |
| --- | --- | --- | --- | --- |
| govern-projection-service | governance | solvable | **pass** | 2/2/2/2 |

Trajectory: `CONTEXT projection_service` → `WALK adr:3 both full`. The agent found `adr:3 --governs-->`
the service (via the govern edges added in the prior milestone), verified no `superseded_by` edge, and
answered correctly + grounded. Corroborating: the pre-milestone in-session eval redo ran two more
scenarios via this same loop — a trace (`derive_axes → capability`, pass) and the govern-gap case
(correctly reported "no ADR governs the traversal tooling", pass) — the escape-hatch/honesty case.
