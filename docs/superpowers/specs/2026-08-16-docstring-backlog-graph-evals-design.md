# Docstring backlog + graph agentic evals (design)

**Status:** proposed (brainstorm dialogue with owner, 2026-08-16).
**Program:** the first-class knowledge graph
(`docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`). This follows the
hierarchical-code-intake milestone (PR #43): the code domain is now derived from source, node context
comes from module docstrings, and `check_missing_docstring` produces a 61-module backlog. This
milestone burns that backlog down and, alongside it, stands up the first **agentic evals** for the
graph.

## Two things, one deliverable

1. **Docstring backlog** — write module docstrings for the 61 flagged modules, so the graph's node
   context is real. This is the immediate task.
2. **Graph agentic evals (lean, proof-of-concept)** — the first evals that measure whether the graph
   is *usable by an agentic tool*, run before/after the backlog to quantify the lift.

## What an "eval" is here (and is not)

An eval is a realistic **agentic task** — the kind an agent (or a human driving one) actually does —
run **using the graph**, that measures whether the graph lets that agent accomplish the task to the
graph's full potential. It is *not* a hook, audit, or check: those are deterministic guards
(`graph-check`, `code-check`, reachability, freshness) that need no agent and already exist. Evals sit
at the opposite end — they measure **fitness for agentic use**.

The use-case categories an eval can cover (owner's framing):

- **Exploration** — "what does X do / where is Y" (project, repo, code).
- **Meta** — "how many of these things".
- **Summaries** — "what use cases are covered / which capabilities show gaps".
- **Spec** — grab the *correct but also minimal* context for a task.
- **Plan** — use the graph to check a plan against the ADRs.
- **Implement** — verify what an implementation reports as fulfilling (capability, use-case, …).

Docstrings connect the two halves: they are the node *context* an agent reads when it walks the
graph, so the evals are how we know the backlog (and the graph generally) is improving agentic
usability — not merely passing checks.

## Layering (decided) — v1 realizes the deterministic layer + one agentic proof

Evals are **layered**:

- **Layer 1 — deterministic** (the reproducible baseline; CI-able later): assert the graph *surfaces*
  the right, *small* context for a task — compared to a hand-verified gold set. Fast, reproducible,
  cheap.
- **Layer 2 — agentic** (on-demand): dispatch an agent with the graph tools and a task, then judge
  its answer and the context it used. Realistic; non-deterministic; costs tokens.

**v1 scope (owner: "backlog-first, evals as proof"):** the full framework is deferred. v1 builds the
deterministic lift metric over **3 scenarios mined from real artifacts**, plus **one** Layer-2
agentic proof. No CI gate, no make-target ceremony.

## The docstring backlog

- **Scope:** the 61 modules in `docs/code/docstring-backlog.md` (nearly all `tools.*` — the knowledge
  tooling documenting itself — plus `utils.text_processing`).
- **Order:** package by package (the backlog's existing grouping); ~13 `tools.*` packages.
- **Who:** a subagent per package (or two packages per subagent). They read the module and write a
  docstring — no code execution needed. The main session runs `make code-index` (the backlog shrinks)
  and the test suite (stays green), and commits.
- **Quality bar:** 1–3 sentences stating the module's **responsibility and role in its package** —
  purpose-first, not restating the code — matching existing `src/` docstring style. Example, for
  `tools/graph/traverse.py`: *"Ephemeral graph traversal: `walk(entry, direction, depth)` rebuilds the
  reachable subgraph from source each call, for use as an LLM working-context substrate (ADR-0025)."*
- **Done when:** `docs/code/docstring-backlog.md` is empty (0 modules) and `check_missing_docstring`
  is silent; `make regen-derived && git diff` clean; the full suite green.

## The eval framework (lean)

### Layout

A lean `evals/graph/` directory:

- One **scenario file** per scenario (`evals/graph/scenarios/<id>.json`): `{id, category, task,
  entry, gold_context, gold_answer, source}` where `entry` is the graph address(es) an agent starts
  from, `gold_context` is the hand-verified set of node addresses that are the necessary+sufficient
  context, `gold_answer` is the expected answer (for exploration/meta), and `source` records the real
  artifact the gold was mined from (PR/spec/plan).
- A **runner** (`evals/graph/run.py`) that, given a scenario and a checkout, computes the Layer-1
  metric and prints a table. It reuses `tools.graph.traverse.walk` / `harvest` — it does not
  re-implement traversal.

No new Make target, no CI wiring, no scoring thresholds this milestone (framework deferred).

### The scenarios (mined from real artifacts)

| # | Category | Task | Gold mined from | Docstring-sensitive |
| --- | --- | --- | --- | --- |
| 1 | Exploration | "What does the `tools.graph` package and each of its modules do?" | hand-verified one-liners per module | **High** — pure node context |
| 2 | Spec | "Gather the minimal necessary context to spec the hierarchical-code-intake change." | what the PR #43 spec depended on: `tools.code.reader`, `tools.graph.{reader,registry,traverse}`, the KG capabilities, ADR-0019/0020/0024, the code/graph catalogs | **Med** — recall + minimality + usable context |
| 3 | Implement | "What obligation does `tools/graph/classify.py` fulfill?" | walk **up** `contained_by` (module → `tools.graph` package) → its implementing capability + governing ADR | Low — edge traversal; exercises walk-up via `contains` |

### The Layer-1 lift metric (deterministic)

For a scenario, run `walk(entry, …)` and take the induced subgraph. Two numbers:

- **Context coverage** — of the scenario's gold **code** nodes reached, the fraction whose
  `walk`-surfaced context (docstring) is non-empty and substantive (≥ a small word threshold). This
  is the docstring-sensitive number: scenario 1 goes ~0% → ~100% across the backlog.
- **Recall / minimality** (scenarios with a `gold_context`) — recall = fraction of `gold_context`
  reached within the scenario's walk depth; minimality = how much the walk over-fetches beyond
  `gold_context` (lower is better). These are docstring-*insensitive* (structure, not context) and
  serve as a sanity frame that the graph reaches the right neighborhood at all.

The runner prints these for a given checkout. The headline deliverable is the **before/after table**:
run the runner against the pre-backlog commit and the post-backlog commit (everything is in git, so
no snapshot is needed up front) and show context coverage rising.

### The Layer-2 agentic proof (one scenario)

For scenario 1, dispatch a subagent **restricted to graph tools** — `python -m tools.graph walk`
and the generated catalogs (`docs/graph/`, `docs/code/`) — and *not* free-reading of all source, and
ask it to summarize `tools.graph` and its modules. Run it against the pre-backlog and post-backlog
states. Before: node contexts are blank, the agent cannot answer from the graph alone (it would have
to read the source it is forbidden from reading). After: the graph alone answers it. This is the
"spontaneous short-term memory" claim, demonstrated rather than asserted. Judged qualitatively (the
main session compares the two answers); no automated judge this milestone.

## Sequence

1. Mine and write the 3 scenario files (gold hand-verified from the real artifacts).
2. Record the **baseline** by running the runner against the current commit (pre-backlog).
3. Burn down the docstring backlog (61 modules), package by package.
4. Re-run the runner (post-backlog) → produce the before/after lift table.
5. Run the one Layer-2 agentic proof (before/after) and record the two answers.
6. Write up the results (a short `evals/graph/RESULTS.md`).

## Scope

**This milestone:** the 61-module docstring backlog; a lean `evals/graph/` (3 mined scenarios, a
runner, the deterministic lift metric, one agentic proof); a results write-up.

**Deferred:** the full eval framework (all 6 categories, a scenario corpus, metric thresholds, a
`make evals` target, a CI gate for Layer 1, an automated Layer-2 judge); evals over the transcript /
product graph (same harness, later); **symbols** (finer code grain — the next milestone after this).

## Testing

- **Runner:** on a fixture graph, context-coverage / recall / minimality compute correctly (a node
  with an empty docstring counts against coverage; a reached gold node counts for recall; an
  over-fetched node counts against minimality). Runs against the real repo without error.
- **Scenario files:** each parses; every `entry`/`gold_context` address resolves to a real node on
  the current graph (no dangling gold — a gold address that does not resolve is a scenario bug).
- **Docstrings:** after the backlog, `check_missing_docstring` returns empty; `docstring-backlog.md`
  regenerates to 0 modules; `make regen-derived && git diff` clean; full unit suite green.
- **Lift:** the before/after table shows scenario-1 context coverage rising from near-0 to near-100%.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-16.

| domain | touched? | note |
| --- | --- | --- |
| code | yes — 61 module docstrings added (node context); `docstring-backlog.md` burns down to empty | the backlog subject |
| graph | yes (read-only) — the runner consumes `walk`/`harvest`; no change to graph logic | evals read the graph |
| evals (new) | yes — new `evals/graph/` area (scenarios + runner + results); not a guarded knowledge domain, a measurement harness | the new artifact |
| capabilities / use-cases / adr / tests | no (logic) — scenarios reference their nodes as gold, unchanged | gold references only |

**Verdict:** reconciled — code gains node context (docstrings) and the graph gains its first
agentic-fitness evals (a lean, git-reproducible before/after harness), with the full eval framework
and symbols explicitly deferred. Evals are a measurement harness, distinct from the deterministic
guards; they are not a new guarded knowledge domain.
