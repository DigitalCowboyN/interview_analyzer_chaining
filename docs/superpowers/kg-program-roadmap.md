# Knowledge-graph program — roadmap (living)

> **This is the living roadmap for the first-class knowledge-graph / OKF program** (the guarded
> knowledge graph over this repo's own codebase). It tracks *forward* work; each milestone still
> gets its own disposable spec + plan. Program spec:
> `docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`. The product
> roadmap (event-sourcing, projections, UI — M1–M5.x) lives separately in `docs/ROADMAP.md`.
>
> **Last updated:** 2026-08-23.

## Shipped

| Phase / milestone | PR | What |
| --- | --- | --- |
| R1 forward loop | #39 | change a surface → its domain check + capture loop |
| L0 corpus substrate | #40 | type-primary intake; domains as projections over `okf_records` |
| L1 traversal engine | #41 | `walk(entry, direction, depth)` over the assembled graph |
| L2 completeness | #42 | reachability + unregistered-type checks (finishes R2) |
| Hierarchical code intake | #43 | code derived from source (packages+modules), overlay retired; ADR-0026 |
| Docstring backlog + first evals | #44 | 61 module docstrings → node context; lean agentic-fitness eval (0→100% lift) |
| Symbols + lazy walk | #45 | symbol-grain nodes + frontier-lazy `walk`; progressive disclosure/discovery; ADR-0027 |
| Graph self-governance + minimal context | #46 | `governs` edges for the tooling ADRs; `gather_context`; CLI `--level`/`context` |
| Durable eval suite | #47 | 17-scenario corpus + Layer-1 scorecard + subscription-only agentic harness + rubric |

**Baseline the suite records (PR #47):** Layer-1 recall — solvable 0.98 / partial 0.88 / **gap 0.33**
(pipeline 0.25, deployment 0.53). Those two gap categories are the quantified roadmap below.

## Upcoming

Ordered by leverage. Each becomes its own spec → plan → build. The eval suite (PR #47) now *scores*
these — closing a gap should visibly move its scenarios' recall.

| # | Milestone | Why now | Eval signal it moves | Status |
| --- | --- | --- | --- | --- |
| KG-1 | **Complete the agentic baseline** | Turn the validated Layer-2 *mechanism* into a full agentic scorecard — run the remaining ~14 scenarios through Mode B (subagent routine), record verdicts. Cheap; banks the eval investment. | Layer-2 coverage: 1 judged → 17 judged | **NEXT (in progress)** |
| KG-2 | **Flow / architecture nodes** | The biggest structural gap, and the eval now *proves* it costs usability: pipeline scored 0.25 because the graph has no runtime-flow edges (command→event→projection→read-model; ingestion→enrichment→lens→export). Authored, linked, drift-guarded flow nodes for the behavioral seams static analysis can't derive. | pipeline `gap` scenarios 0.25 → higher | queued |
| KG-3 | **Infra / deployment modeling** | Closes the other gap: deployment scored 0.53. Model the container/service topology (compose services, config, what must be up) as graph nodes/edges so "what does X need to run / what breaks if Y changes" is answerable. | deployment `gap` scenarios 0.33–0.50 → higher | queued |
| KG-4 | **L3 governance on shapes** | The program's original payoff (R3): rules/policies/hooks keyed on graph shapes + traversals (canonical: editing a CodeUnit → walk inbound `governs` → surface the ADR's scope → flag out-of-scope drift). Mechanism (hook vs rule) still to brainstorm. | (new checks; not a current eval category) | parked (needs its own brainstorm) |

## Parked / backlog (smaller, not lost)

- **Symbol-docstring backlog** — many symbols are thin (signature only); `check_missing_symbol_docstring`
  exists (opt-in) but no burn-down done. A generated `docs/code/` backlog + burn-down, like the module one.
- **Reverse `called_by`** — symbol callers aren't resolvable (only `calls`); deferred at PR #45 because
  it needs scanning unvisited bodies against the frontier-lazy model.
- **Walk perf** — every `walk` rebuilds the module base via `harvest` (~3 s); fine for tooling, sluggish
  for an interactive agent loop. A shared/cached base index would help (cache deferred per ADR-0025).
- **Eval suite follow-ups** — CI integration (deliberately local today); human judge calibration of the
  rubric; auto-mining scenarios from PR history; Mode A hardening if headless-nested becomes reliable.
- **`render_signature`** — lossy for keyword-only / positional-only args (0 occurrences today).
- **Corpus/domain projections** — migrate remaining doc-readers to project over `okf_records`.

## How this maps to the program phases

R1 = forward loop (done). R2 = L0 + L1 + L2 (done). The milestones since (code intake → eval suite)
harden the substrate's *content and usability*. **KG-2/KG-3 extend the graph's coverage** (behavioral
+ infra seams); **KG-4 = L3 = R3**, the governance payoff, built once the substrate is trustworthy and
complete enough to hang policy on. KG-1 (full agentic baseline) is the measurement that tells us when
that's true.
