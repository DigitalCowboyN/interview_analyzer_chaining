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
| KG-1 | **Complete the agentic baseline** | Full Layer-2 agentic scorecard via Mode-B autonomous subagents (agent drives the graph CLI itself + judge). | **16/17 pass** (spec-code-intake truncated by a session limit → 1 re-run pending); every gap/partial passed by honest reporting | ✅ **DONE** (RESULTS.md Layer 2) |
| KG-2 | **Flow / architecture nodes** | The biggest structural gap. Shipped as a **derived** event-and-label overlay (ADR-0028: `emits`/`handled_by`/`writes`/`reads` over existing event-class symbols + glossary labels), not authored flow nodes — the coupling was already latent. | pipeline `gap` scenarios: KG-1 passed by *reporting* the missing edge → KG-2 re-run **traverses** the write path + full schema blast-radius (RESULTS.md "KG-2 re-run") | ✅ **DONE** |
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
- ~~**Supersede edges not surfaced by traversal**~~ — **RETRACTED (false alarm, verified 2026-08-23).**
  The KG-1 `govern-superseded` agent claimed `walk`/`context` don't expose supersede edges, but that was
  an unlucky control (it only tested ADRs with `supersedes: []`). Verified: the one real edge
  `adr:14 → adr:8` **is** in `harvest()` and `walk(adr:8, both)` surfaces it. No graph bug. (adr:5, near
  ingestion, is genuinely not superseded — so the correct answer was "not superseded.") **Eval-quality
  lessons instead:** (a) `govern-superseded` is arguably mis-graded `partial` — the graph *can* answer it,
  so it's closer to `solvable`; (b) the rubric should require a *positive control* before crediting a
  confident "the tool can't do X" as honesty rather than an unverified negative.
- ~~**Schema→consumer is only a label-string match**~~ — **CLOSED by KG-2 (2026-08-24).** The `reads`/`writes`
  overlay (ADR-0028) makes it a real edge: `walk(glossary:<Label>, in)` now reaches both `written_by` handler
  modules and `read_by` graph-queries (→ `consumed_by` api/export). The `deploy-neo4j-schema-blast` re-run
  traversed the full blast radius for `glossary:Fragment` (1 writer + 20 readers).
- **Command-handler → aggregate `calls` hop missing** (surfaced by KG-2's `pipeline-write-path` re-run, verified 2026-08-24):
  `commands.handlers.*._handle_*` don't link to `events.aggregates.*` methods, and those aggregate methods have
  zero outbound edges — the aggregate is loaded from the repository (`fragment = repo.load(id); fragment.edit(...)`),
  so its type isn't statically inferable (the documented pragmatic-`calls` ceiling, ADR-0027/0028). The write spine
  stays connected via the `emits` overlay through the `create_*_event` factories, but this specific edge is absent.
  A `# calls:` marker on the load-then-mutate handlers (or repo-load return-type hints) would close it. Small, deferred.
- **spec-code-intake agentic re-run** — one KG-1 scenario truncated by a session limit; a single autonomous
  Mode-B dispatch away from a complete 17/17 Layer-2 scorecard.

## How this maps to the program phases

R1 = forward loop (done). R2 = L0 + L1 + L2 (done). The milestones since (code intake → eval suite)
harden the substrate's *content and usability*. **KG-2/KG-3 extend the graph's coverage** (behavioral
+ infra seams); **KG-4 = L3 = R3**, the governance payoff, built once the substrate is trustworthy and
complete enough to hang policy on. KG-1 (full agentic baseline) is the measurement that tells us when
that's true.
