# Use-Cases Domain — design

**Status:** approved by owner 2026-08-06 (brainstorm dialogue).
**Program:** the **source layer** of the guarded knowledge graph — the user-centered
"why" that capabilities, code, ADRs, and (next round) tests all serve. This is the
first domain to record *direct user-centered input*: requirements, user stories,
features, and formal use cases, each carrying acceptance criteria when known. It answers
the question the capabilities rounds left open — *why does a capability exist, and why is
one only partially implemented?* — by giving capabilities something above them to point
at. Our graph becomes a **Requirements Traceability Matrix**: use-case → capability →
code (→ test, next round), with coverage surfaced as a derived state.

This is **Round 3a**. Round 3b (the **tests** domain — `Test` node, `verifies` edge,
verification-grade coverage) is a separate spec that builds directly on this one.

## Framing (locked in brainstorm + research)

Grounded in the industry sources reviewed 2026-08-06 (Cockburn's *Writing Effective Use
Cases*; the RTM literature — Ketryx/Perforce/TestRail; the modern test pyramid;
requirements-coverage frameworks):

- **Requirement / user story / feature / use case are one thing at different fidelity**,
  not four different things. A requirement is the goal; a user story is its lightweight
  Agile carrier (*"As a `<actor>`, I want `<action>` so that `<benefit>`"*); a use case
  is its structured form (actors + main success scenario + extensions). All carry
  **acceptance criteria** — the testable "definition of done." → **one node type with a
  `form` attribute**, not four node types.
- **Cockburn's use-case fields** are the richer-form vocabulary we borrow: primary
  actor, scope, **level** (user-goal / summary / subfunction — this maps onto the
  primary/child intuition), trigger, preconditions, main success scenario, extensions,
  end conditions. An **actor can be a person, an operator, or an external system** — which
  is exactly why use-cases are *product, operations, or support*, not product-only.
- **The RTM is the mental model.** "High code coverage with no traceability means you may
  be testing the wrong things intensively while requirements that matter to customers have
  no coverage at all" (TestRail). The use-cases layer is what makes that gap *visible*.
- **Coverage is a derived status, never a stored flag.** The standard enum is
  `NOT_COVERED | PARTIALLY_COVERED | FULLY_COVERED`. It is computed from graph edges,
  exactly as capability implementation-degree already is — no `status` field to rot.

**Hard constraint (owner):** **capabilities are read-only inputs.** Nothing in this round
un-implements a capability, removes an `implements` edge, or edits a capability file. The
capability tree is a *source* we reverse-engineer use-cases from; the use-cases are
net-new nodes *above* it. The `fulfilled_by` edge (below) is authored entirely on the
use-case side precisely to honor this.

**Extensibility is a first-class requirement (owner):** `form` and `category` are
**open, ordered sets** (add a value, don't fork the schema), mirroring the capability
`category` axis. New forms (e.g. `job-story`, `epic`) and new categories cost one edit.

## The node — one `UseCase` type, fidelity in `form`

One file per use-case in `docs/use-cases/`, `type: UseCase` in frontmatter. A **minimal
required core** every form carries, plus an **optional Cockburn block** only richer forms
fill in (KISS/YAGNI — a user story is not forced to write extensions).

**Required core (any form):**

| field | meaning |
| --- | --- |
| `id` / title | the goal as a short verb phrase (slug = filename) |
| `form` | **open set**: `user-story \| feature \| requirement \| use-case` |
| `category` | **reuses the capability axis** (open set): `product \| operations \| support` |
| `actor` | who wants it — person, operator, or external system |
| `statement` (body) | the narrative: "As a `<actor>`, I want … so that …", or a goal-in-context sentence |
| `acceptance_criteria` | a **list of free-text strings** (each may be Given/When/Then or a rule sentence); **may be empty** — empty is legal and *meaningful* (surfaced by the guard, not hidden) |
| `fulfilled_by` | list of capability slugs whose current implementation reaches toward this intent (may be empty) |

**Optional Cockburn block (richer `use-case` form):** `level`
(`user-goal | summary | subfunction`), `preconditions`, `main_scenario`, `extensions`,
`end_conditions`. Absent on lightweight forms.

**Acceptance criteria = list of strings** (not a rigid Gherkin struct). YAGNI: a string
list validates cleanly today; a later round can parse Given/When/Then structure if the
tests domain needs to bind to individual clauses.

**Reserved, not built (extensibility, shown not built):** use-case↔use-case hierarchy
(Cockburn `summary` contains `user-goal` contains `subfunction`). The `level` field
records the depth as *data* now; a `refines` edge (UseCase → UseCase) is a one-entry
registry addition when something needs to *traverse* the hierarchy. Not this round.

## The edge — `fulfilled_by`, authored on the use-case side

One new edge activates the graph's reserved `UseCase` node type:

```python
EdgeType("fulfilled_by", inverse="fulfills",
         from_type="UseCase", to_type="Capability", source="authored",
         field="fulfilled_by", resolve="id",
         description="A use-case's intent is reached toward by a capability's implementation.")
```

- **Direction of authoring keeps capabilities untouched.** The traceability verb is
  *"a capability fulfills a use-case"* (`fulfills: Capability → UseCase`). We store it as
  the **use-case declaring its fulfilling capabilities** (`fulfilled_by: [slug]`), so
  every edit lands in `docs/use-cases/` and no capability file is ever opened. The
  canonical `fulfills` direction is the computed inverse — the graph is identical.
- **Node type activation:** add `"UseCase": "use-cases"` to the graph registry's
  `NODE_DOMAINS`, and a reader adapter `("UseCase": (load_use_cases, "slug"))` so the
  harvester can enumerate and resolve use-case endpoints. Addressing: `use-cases:<slug>`.
- **`resolve="id"`** — a `fulfilled_by` value is a capability slug, resolved directly
  against the capability node set. A dangling one (typo, renamed capability) is caught by
  the existing **graph-check** cross-domain endpoint sweep — no new cross-domain check
  needed; the graph guard already covers it.

## Coverage — a derived state, transitive through capabilities

A use-case's coverage is **computed**, never stored, from its `fulfilled_by` edges and
each fulfilling capability's own implementation-degree (which the capabilities domain
already derives from `implements` → code):

| state | condition |
| --- | --- |
| `NOT_COVERED` | no `fulfilled_by` capability — a bare intent nothing yet reaches toward |
| `PARTIALLY_COVERED` | has fulfilling capabilities, but at least one is unimplemented / partially implemented (a gap in the fulfilled_by → implements → code chain) |
| `FULLY_COVERED` | fulfilled by capabilities whose implementation is complete |

**Progressive by round.** In 3a, `FULLY_COVERED` means *"implemented"* — the chain from
intent to code is whole. It does **not** yet mean *"tested against its acceptance
criteria"*; **verification-grade coverage** arrives in 3b when the `verifies` edge
(Test → UseCase / Capability) lets us distinguish *implemented* from *proven*. The state
enum is stable across both rounds; 3b refines the `FULLY_COVERED` predicate, it doesn't
rename the states.

## Domain machinery (follows the established per-domain pattern)

- **`docs/use-cases/`** — its own root folder: a `README.md` concept doc (what a use-case
  is, the forms, that coverage is derived, capabilities-are-read-only), one file per
  use-case, and generated `index.md`.
- **`tools/usecase/`** (Python package — no hyphen, mirrors `tools/capability`):
  - `reader.py` — `UseCase` dataclass; `FORMS` + reuse of capability `CATEGORIES`;
    `load_use_cases(root, uc_dir="docs/use-cases")` (parse frontmatter, skip `index.md`).
  - `render.py` — `render_index(use_cases, coverage)` (catalog grouped by category/form,
    with derived coverage column). Pure.
  - `check.py` — `Finding`; the checks below; `run_all(root=".")`. Non-blocking, `return 0`.
  - `__main__.py` — `index | check | coverage`.
  - `coverage.py` (or a function in reader) — `coverage(use_cases, capabilities)` →
    `{slug: state}`, the derived-state logic above, reused by render, check, and CLI.
- **Makefile** — `usecase-index`, `usecase-check`; add `usecase` to the `health` loop.
- **`.githooks/pre-commit`** — no change needed (graph-check already sweeps the new
  `fulfilled_by` endpoints); `usecase-check` runs under `make health` / on demand.
- **Cascade + registry** — add `("use-cases", "usecase")` to `tools/knowledge`'s
  `DOMAINS`, a row to `docs/index.md`, and the `UseCase` entry to the graph registry.

## The guard — `make usecase-check` (non-blocking, exit 0)

All advisory, `return 0` — findings inform, never block:

- **form-in-set** — `form` not in the open `FORMS` set → finding.
- **category-in-axis** — `category` not in the capability `CATEGORIES` axis → finding.
- **empty-acceptance-criteria** — a use-case with no criteria → advisory finding
  ("no criteria to test against yet"). This is *surfaced, not hidden* — it is exactly the
  signal that a stated intent isn't yet testable.
- **uncovered-intent** — a `NOT_COVERED` use-case (no `fulfilled_by`) → advisory finding.
  A bare intent nothing reaches toward is the gap this domain exists to reveal.
- **index-sync** — committed `docs/use-cases/index.md` matches a fresh render.

Cross-domain endpoint integrity (a `fulfilled_by` pointing at a nonexistent capability)
is **already** covered by `graph-check` — deliberately not duplicated here.

## CLI

`python -m tools.usecase {index | check | coverage}`. `coverage` prints each use-case
with its derived state (the RTM view from the use-case end); `neighbors use-cases:<slug>`
in the **graph** CLI already answers "what fulfills this" once the node type is
registered — no new traversal CLI needed here.

## Two phases (one round)

**Phase 1 — machinery + exemplars.** Build the schema, `tools/usecase/`, the guard, the
graph wiring, cascade/registry rows, Makefile/`health`. Land it with **2-3 exemplar
use-cases** across forms and categories (e.g. one product `use-case`, one operations
`user-story`, one `NOT_COVERED` requirement) that prove the pipeline end-to-end: they
render, the guard runs clean-or-advisory, coverage derives, `graph-check` stays clean,
`neighbors` traverses `fulfilled_by`.

**Phase 2 — full corpus derivation (same round, reviewed before merge).** Diligently
**reconstruct the originating intents** — the use-cases that, had someone written them
down first, would have *led to* this system. This is retrospective derivation: work
**backward from the capability tree to the real human problem** that motivated it, then
draw `fulfilled_by` back to the capabilities that reach toward it.

> **The anti-restatement rule (design guidance, enforced in review).** A use-case must
> reach *past* the capability to the human problem. "The system extracts fragments" →
> "As a user I want fragments extracted" is a **restatement** and is rejected. "As an
> analyst drowning in raw transcripts, I want the signal surfaced so I stop missing what
> matters" is a **use-case**. Every derived use-case's `statement` must name an actor and
> a benefit that would make sense to someone who has never seen the code. Operations and
> support use-cases follow the same rule with operator/maintainer actors.

> **The trajectory test (how we know derivation is honest, not restatement).** A correctly
> derived corpus **overshoots the current build**. If every use-case maps 1:1 onto an
> existing capability, the derivation has merely mirrored the code — restatement at corpus
> scale. Following a real user-problem honestly points *past* what is built today to where
> the system's direction is clearly heading (e.g. deriving an import/onboarding use-case,
> or a "revisit and correct a past extraction" use-case, that no capability yet fulfills).
> **These uncovered and partially-covered use-cases are the expected, desired output** —
> the `NOT_COVERED` / `PARTIALLY_COVERED` states are the domain doing its job (surfacing
> where intent outruns implementation), not defects to be pruned. A corpus with **zero**
> uncovered intents is a red flag in review that we restated rather than derived.

The corpus is authored, `fulfilled_by` links drawn, coverage reviewed (expect genuine
`NOT_COVERED` / `PARTIALLY_COVERED` results — those are the point, not defects), and the
owner reviews the corpus before merge.

## Testing

- **Unit** — `load_use_cases` parses core + optional fields, skips `index.md`, ignores
  non-`UseCase` files; `FORMS`/`CATEGORIES` open-set membership; `coverage()` yields
  `NOT_COVERED` (no `fulfilled_by`), `PARTIALLY_COVERED` (fulfilled by an unimplemented
  capability), `FULLY_COVERED` (fulfilled by a fully-implemented capability) over a
  synthetic capability+use-case fixture; each `check_*` flags its case and passes on a
  clean one; `render_index` groups by category/form with the coverage column; **assert no
  check raises**. Graph: `harvest` emits `fulfilled_by` (UseCase→Capability) from a
  fixture use-case; `nodes` includes the `UseCase` set; a dangling `fulfilled_by` is
  flagged by `graph.check_endpoints` (the reserved-edge-activation test).
- **Smoke** — `make usecase-index` writes `docs/use-cases/index.md` over the real repo
  (Phase-1 exemplars); `make usecase-check` clean-or-advisory; `make graph-index` +
  `make graph-check` clean with `fulfilled_by` live and counted; `make knowledge-check`
  (cascade row + `DOMAINS` entry) and `make cli-check` (`usecase-*` targets catalogued)
  clean; `make health` runs the use-cases check in the sweep.

## Non-goals (this round)

- **The tests domain / `verifies` edge** — Round 3b; reserved, not built. This round's
  `FULLY_COVERED` means implemented, not verified.
- **Editing any capability file** — the `fulfilled_by` edge is authored on the use-case
  side precisely so this never happens.
- **Use-case↔use-case hierarchy traversal** — `level` records depth as data; the
  `refines` edge is reserved.
- **A rigid Gherkin/acceptance-criteria parser** — criteria are strings this round.
- **Blocking** on any finding — the guard is advisory, `return 0`.
- **Modifying other domains' generated files** — the use-case views live in
  `docs/use-cases/`; capability/code/graph outputs are untouched.

## Capture as ADR

Capture **ADR-0021**: adopt a use-cases domain as the graph's source layer — one
`UseCase` node type with an open `form` axis + Cockburn optional block, acceptance
criteria as strings, coverage as a **derived** `NOT/PARTIALLY/FULLY_COVERED` state
transitive through capabilities, and a `fulfilled_by` edge **authored on the use-case
side** to keep capabilities read-only. `source:` = this spec. Refines the domain-family
ADRs and ADR-0019 (capabilities-as-intent) and ADR-0020 (graph-links); supersedes nothing.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-06.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| use-cases | yes | the new source-layer domain | — |
| capabilities | yes (read-only) | `fulfilled_by` targets its slugs; coverage reads its implementation-degree; **no capability file edited** | the round's hard constraint |
| graph | yes | activate reserved `UseCase` node type + `fulfilled_by` edge; `graph-check` covers cross-domain endpoints | one registry entry + one adapter |
| code | yes (read-only) | coverage chain terminates at code via capability `implements` | no change |
| cli | yes | new `usecase-*` + `health` update → `cli-index`; `cli-check` clean | — |
| adr | yes | ADR-0021 | — |
| knowledge | yes | cascade row + `DOMAINS` entry for `use-cases` | — |
| glossary / api / prompts / graph-queries | no | — | unaffected |

**Verdict:** reconciled — use-cases (subject) + graph (activation) touched;
capabilities/code (read-only) consulted; cli/adr/knowledge reconciled in the plan.
