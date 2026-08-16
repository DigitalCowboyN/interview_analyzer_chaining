# L1 — Traversal engine (design)

**Status:** proposed (brainstorm dialogue with owner, 2026-08-16).
**Program:** Phase L1 of the first-class knowledge graph
(`docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`). Realizes
**ADR-0025** (the graph is a first-class, ephemeral, rebuilt-from-source traversal substrate) and
extends the node set per **ADR-0020** (reserved node types cost nothing until their round). No
new ADR: L1 is these two decisions carried out.

## Purpose

Make the graph *walkable*. Today `neighbors` shows one hop; L1 gives the primitive the whole
program is named for: **materialize and walk a subgraph from any node (or entry set), any
direction, any depth (bounded or to exhaustion), any time** — and the subgraph it returns,
each node carrying its **claim + context**, *is* the working context an agent feeds a model.

Scope also **completes the known node set** so there is a fuller graph to walk: add
`GlossaryTerm`, `GraphQuery`, and `Prompt` as node types (Test/CodeUnit/Capability/ADR/UseCase
already exist).

## The traversal primitive

`tools/graph/traverse.py`:

```python
def walk(entry, direction="both", depth=None, root=".") -> Subgraph
```

- **entry** — a node address (`"code:api"`), or a selector:
  - `"type:Capability"` — every node of a type
  - `"under:src/api/"` — every CodeUnit whose unit dir is below a path (reuses `_units_under`)
- **direction** — `"out"` (follow edges forward), `"in"` (backward, via inverses), `"both"`.
- **depth** — an `int` (bounded — progressive discovery), or `None` = walk to exhaustion.
- **returns** — a **`Subgraph`**: the visited nodes (each with address, type, and its
  **claim + context** body) and the edges among them.

Algorithm: `harvest()` fresh → build a directed adjacency (both orientations indexed for `in`)
→ BFS from the entry set to the depth bound → collect visited nodes + induced edges. Rebuilt
from source every call (ADR-0025); no cache.

```python
@dataclass
class Node:
    address: str        # "<slug>:<id>"
    type: str           # "Capability" | "CodeUnit" | ...
    context: str        # the node's claim+context body (see resolution below)

@dataclass
class Subgraph:
    nodes: Dict[str, Node]      # keyed by address
    edges: List[Edge]           # induced edges among `nodes`
```

`neighbors(addr)` becomes the shorthand `walk(addr, "both", 1)` and its CLI/output is preserved.

## Claim + context resolution

A walk carries each node's body so the subgraph is usable as model context. An address
`<slug>:<id>` resolves to its source object's text via a per-type resolver (built on the
existing loaders):

| Type | slug | body field |
| --- | --- | --- |
| Capability | capabilities | `statement` |
| UseCase | use-cases | `statement` |
| CodeUnit | code | `description` |
| ADR | adr | `title` + `body` |
| Test | tests | name + `verifies` targets (no rich body) |
| GlossaryTerm | glossary | definition body |
| GraphQuery | graph-queries | docstring + Cypher |
| Prompt | prompts | the prompt text |

Full body by default — the subgraph *is* the context; `depth`/`direction` are how you bound its
size. (A summary-only mode is a future option, not built now.)

## Completing the node set

Three node types join the graph, each a registry + adapter addition (ADR-0020's anticipated
extension), with one derived edge apiece:

- **GlossaryTerm** (`type: Term`, id = `term`) — already flows through the L0 corpus as a
  frontmatter record; cleanest to add. Edge **`defined_in` → CodeUnit**, derived from the
  term's `source:` path (resolve=path, like ADR `governs`).
- **GraphQuery** (id = function `name`) — edge **`consumed_by` → CodeUnit**, derived from
  `QueryEntry.consumers`.
- **Prompt** (id = entry `name`) — edge **`consumed_by` → CodeUnit**, derived from
  `PromptEntry.consumers`.

`NODE_DOMAINS` un-reserves `graph-queries`/`prompts` and adds `glossary`; `EDGES` gains the
three derived edge types; `reader.py::_ADAPTERS` gains three `(loader, id_attr)` entries.

**Transitional caveat (owner-accepted):** GraphQuery and Prompt enter via their *current*
structural/positional discovery (a function containing Cypher; a keyed block in
`prompts/*.yaml`) — the same pre-migration state Test is already in. The explicit `# okf:`
marker migration (a later phase, ADR-0024) converts *all* structural discovery to explicit at
once. GlossaryTerm has no such caveat (it is frontmatter). This completes today's graph now and
disciplines discovery wholesale later, rather than leaving the graph partial until the
migration.

## CLI

```
python -m tools.graph walk <entry> --dir out|in|both --depth N|full
```

Prints the subgraph (nodes grouped by type with their context, then the induced edges).
`neighbors` stays as the depth-1 shorthand.

## Non-goals (this phase)

- **Predicate/query entry selectors** beyond node / `type:` / `under:` (YAGNI).
- **A materialized cache** (ADR-0025 — deferred).
- **The `# okf:` explicit-marker migration** (a later phase; L1 rides current discovery).
- **Detecting *new*, undeclared derived domains** as they appear — no mechanism today; it is the
  orphan-class blind spot and belongs in **L2 (completeness)**. Parked here deliberately.
- **api/cli as node types** (Endpoint/Command) — candidates, not this phase.

## Testing

- **walk core:** from a known node, depth 1 / depth N / to-exhaustion each return the expected
  node+edge set, in each direction; `in` uses inverses correctly; a cycle terminates; an
  unknown entry returns an empty subgraph.
- **selectors:** `type:Capability` returns all capability nodes; `under:src/api/` returns the
  right CodeUnits; a specific address returns just its neighborhood.
- **context:** each node in a subgraph carries non-empty context for types that have a body.
- **node set:** after adding the three types, `harvest()`/`nodes()` include them; a
  `GlossaryTerm defined_in CodeUnit` edge resolves (no dangling); `graph-check` clean.
- **neighbors parity:** `walk(addr, "both", 1)` equals the old `neighbors` neighborhood.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-16.

| domain | touched? | note |
| --- | --- | --- |
| graph | yes — new `traverse.py` (`walk`) + three node adapters + edges in the registry | the subject |
| glossary / graph-queries / prompts | yes — become graph node types (their readers reused as adapters) | first-class nodes |
| code | yes (read-only) | the target of all three new edges; `code-check` unaffected |
| adr | yes | no new ADR — realizes 0025, extends 0020 (reserved types) |
| corpus / capabilities / use-cases / tests | no (logic) | unaffected; Test already a node |

**Verdict:** reconciled — graph is the subject (traversal + node-set completion); glossary/
graph-queries/prompts become nodes via their existing readers; no new ADR (0025 + 0020 govern).
