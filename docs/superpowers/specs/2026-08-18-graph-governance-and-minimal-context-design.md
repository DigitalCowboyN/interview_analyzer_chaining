# Graph self-governance + minimal-context retrieval (design)

**Status:** proposed (brainstorm dialogue with owner, 2026-08-18).
**Program:** the first-class knowledge graph
(`docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`). Closes the
top gaps surfaced by the agentic evals after the symbols + lazy-walk milestone (PR #45): the graph
could not point an agent at the decision that governs its own tooling, symbols were unreachable via
the CLI, and a "gather context" walk over-fetched a 644-node blast radius.

## Context (the gaps, ground-truthed by the evals)

- **Govern gap:** 80 `governs` edges exist, all targeting `src/` application code; **zero** target
  `tools/`. So the knowledge-graph tooling's own ADRs (0020/0024/0025/0026/0027) are structural
  islands — an agent walking up from `tools/graph/traverse` reaches capabilities but no governing
  ADR, at any depth.
- **CLI gap:** `python -m tools.graph walk` exposes only `entry/--dir/--depth` — no `--level`, so a
  CLI-driven agent cannot reach the symbol grain the last milestone built.
- **Over-fetch:** an exhaustive walk from one code node reaches 644 nodes; there is no bounded
  "give me the small necessary context" retrieval.
- **Phantom call edges:** `calls_of` emits false edges like `NODE_DOMAINS.get` (calling a dict
  method on an imported name), materializing empty phantom nodes.

Phase 1 (already committed on this branch, `8c8c77d`) captured the traversal work as intent
(capability `walk-the-graph-for-context`, use-case `gather-context-with-the-graph`). This spec is
Phase 2 — the fixes.

## Decision 1 — Author `governs` edges for the KG-tooling ADRs

`governs` (ADR→CodeUnit, `resolve="path"`) matches code nodes whose directory path starts with the
listed path. Audit the knowledge-graph-program ADRs and add a `governs:` frontmatter list to each,
pointing at the `tools/` code it actually decides:

| ADR | governs (paths) |
| --- | --- |
| 0020 — typed-edge graph model | `tools/graph` |
| 0024 — corpus substrate | `tools/corpus` |
| 0025 — ephemeral traversal substrate | `tools/graph/traverse`, `tools/graph/neighbors` |
| 0026 — hierarchical code intake | `tools/code` |
| 0027 — lazy walk + symbols | `tools/graph/traverse`, `tools/graph/neighbors`, `tools/code/reader` |

- Paths are **directories** (`tools/graph/`) for broad ADRs and **specific `.py` files**
  (`tools/graph/traverse.py`) for precise ones — `_units_under` was enhanced to resolve a `.py` path
  to its module node (via `_unit_of_file`), so the `.py` form both resolves the graph edge AND passes
  `adr-check`'s on-disk existence check + matches the `governed-by` marker key. Reciprocal
  `# governed-by:` markers are added to the governed files, and `adr-check` scans `tools/`.
- Only edges that are **genuinely true** are added (the ADR really constrains that code). This is
  authored intent, guarded by `adr-check` staleness like every other `governs` edge — not a blanket
  auto-link.
- **Effect:** an agent walking up (`in`) from `tools/graph/traverse` now reaches ADR-0025/0027; the
  reachability check gains real ADR explanations for the tooling.

## Decision 2 — `gather_context(entry)`: minimal task-context retrieval

A new function in `tools/graph/traverse.py` that returns the **small necessary context** for a task
targeting `entry`, instead of the full closure. It realizes the owner's "walk up progressively until
intent, then bounded local" shape:

```
gather_context(entry, root=".", level="module") -> Subgraph
  1. Progressive walk-UP: for d in 1..MAX_UP (default 6):
        up = walk(entry, direction="in", depth=d, root, level)
        if up contains any capabilities:/use-cases:/adr: node -> stop (shortest path to intent)
  2. Bounded LOCAL: out = walk(entry, direction="out", depth=1, root, level)
        (the entry's own dependencies / calls / contained symbols)
  3. Return the union of (up, out) as one Subgraph.
```

- **Why this is "minimal":** it stops climbing the moment it reaches governing intent (not the whole
  ancestor closure), and it takes only one hop of local structure — the context a spec/plan/implement
  task actually needs (the target, what it uses, and the decision/capability that governs it).
- **Composes with Decision 1:** once the tooling ADRs have `governs` edges, `gather_context` on
  `tools/graph/traverse` surfaces ADR-0025/0027 in a handful of nodes rather than 644.
- If no intent is found within `MAX_UP`, it returns the deepest up-walk it reached plus the local
  neighborhood, and the caller can see intent was not reached (honest, per the use-case's
  acceptance criteria).

## Decision 3 — Expose the CLI: `--level` and a `context` command

In `tools/graph/__main__.py`:

- `walk` subcommand gains `--level {module,symbol}` (default `module`), passed to `walk()`. Symbol
  grain becomes reachable from the CLI.
- New `context` subcommand: `python -m tools.graph context <entry> [--level …]` runs
  `gather_context` and prints the resulting subgraph (nodes with context + edges), the agent-facing
  way to get minimal task context.

## Decision 4 — Trim phantom call edges

In `tools/code/reader.py::calls_of`, an `Attribute` call `base.attr()` currently resolves to
`<base-target>.attr` even when `attr` is a builtin container/string method (`get`, `append`,
`items`, …) — producing false edges and empty phantom nodes. Add a small denylist of common
builtin method names and skip attribute-calls whose `attr` is in it. This kills phantoms like
`NODE_DOMAINS.get` without losing real submodule calls (`reader.harvest`), because real function
names aren't container-method names.

## Scope

**This milestone:** `governs` edges for the 5 tooling ADRs; `gather_context` + its CLI `context`
command; `walk --level`; the phantom-edge denylist. (Plus the Phase 1 intent nodes already committed.)

**Deferred (own milestones):** authored **flow/architecture nodes** for behavioral seams; the durable
**eval suite**; the **symbol-docstring backlog** burn-down; reverse `called_by`; walk **perf**
(harvest base ~3 s); `render_signature` kw-only/pos-only rendering.

## Testing

- **Govern edges:** after authoring, `graph-check` reports no dangling; a test asserts each new
  `governs` edge resolves to the intended `tools/` node (e.g. a harvested `governs` edge
  `adr:27 → code:tools.graph.traverse` exists), so an agent walking `in` from that code now reaches
  the ADR. (Reachability counts are unchanged — the tooling was already reached via capabilities; the
  win is that the *governing decision* is now on the path.)
- **`gather_context`:** on a fixture graph, it stops at the first intent layer (returns a small set
  containing the entry, its 1-hop out-neighbors, and the nearest capability/ADR) and does NOT return
  the full closure; on the real repo, `gather_context("code:tools.graph.traverse")` returns a small
  subgraph that now includes ADR-0025/0027 (post-Decision-1) — far fewer than 644 nodes.
- **CLI:** `walk --level symbol` surfaces symbols; `context <entry>` prints the minimal subgraph;
  both exit 0.
- **Phantom denylist:** a fixture with `d = {}; d.get(x)` on an imported name emits no `.get` edge;
  a real `reader.harvest()` call still emits its edge.
- **Regression:** module-grain `walk` unchanged (harvest-equivalence still green); full suite green;
  freshness clean.

## ADR

No new ADR. Decision 1 is authoring `governs` edges the existing ADRs (0020–0027) always implied;
Decisions 2–4 are additive tooling within the model ADR-0020/0025/0027 already established. `adr-check`
governs the new `governs` edges' staleness.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-18.

| domain | touched? | note |
| --- | --- | --- |
| adr | yes — `governs:` frontmatter added to 0020/0024/0025/0026/0027 | the govern-gap fix |
| graph | yes — `gather_context` in traverse; `--level` + `context` in the CLI | minimality + CLI |
| code | yes — `calls_of` phantom denylist | edge-quality |
| capabilities / use-cases | yes (Phase 1, already committed) — walk-the-graph-for-context + gather-context-with-the-graph | intent capture |

**Verdict:** reconciled — the knowledge-graph tooling becomes honest about its own governance
(`governs` edges for its ADRs), an agent can retrieve minimal task-context (`gather_context`) and
reach symbols from the CLI (`--level`/`context`), and phantom call edges are trimmed. Behavioral
flow-nodes, the eval suite, and the symbol backlog are deferred to their own milestones.
