# Graph-Links Model — design

**Status:** approved by owner 2026-08-06 (brainstorm dialogue).
**Program:** the cross-domain edge layer for the guarded knowledge graph. Every domain
so far invented its own link field and its own renderer. This introduces one
**extensible edge registry** + a shared harvester/renderer/guard, so the domains form a
single traversable graph — and so new edges (tests, use-cases, …) are *registry
additions, not rewrites*.

## Framing (locked in brainstorm + research)

- **OKF's native edges are untyped, undirected body links** — "the specific kind
  (parent/child, references, depends-on) is conveyed by the surrounding prose, not by
  the link itself." OKF v0.2 defines no typed-relationship frontmatter. So typing is an
  explicitly-open extension point.
- **We want the property-graph model** (native to Neo4j, which we already run): edges are
  **first-class, typed, directed, and carry properties**. Naming = **verb / verb-phrase**,
  direction clear from the name, each with a derivable inverse.
- **Borrow the software-traceability vocabulary** for edge names: `implements`,
  `depends-on`, `governs`, `supersedes`, `verifies`, `fulfills`, `refines`, `derives`.
- **So our model is an OKF-conformant extension:** keep OKF body links for portability;
  treat our **typed frontmatter fields as the machine-readable graph**, defined once in a
  registry, using the field's verbs.

**Extensibility is a first-class requirement (owner):** the model must not be built for
only today's edges. A **tests** edge (typed `unit | integration | e2e | …` — an edge
*property*) and **use-case** edges are coming; adding them must be a registry entry, not
a redesign. Reserved-but-unfilled edge/node types cost nothing.

## The edge registry (the extensible heart)

One canonical list in `tools/graph/registry.py`. Each entry is an edge **type**:

```python
EdgeType(
    name="implements",            # verb (traceability vocabulary)
    inverse="implemented_by",     # "" if none
    from_type="Capability",
    to_type="CodeUnit",
    source="authored",            # authored | derived
    field="implemented_by",       # frontmatter key it is harvested from (authored edges)
    resolve="unit",               # how the target string maps to a node id (unit | path | slug | id)
    properties=[],                # PropSpec list — edge properties (e.g. test_type)
    description="...",
)
```

**Initial registry (today's real edges):**

| name | inverse | from → to | source | field / origin |
| --- | --- | --- | --- | --- |
| `implements` | `implemented_by` | Capability → CodeUnit | authored | `implemented_by` |
| `parent_of` | `child_of` | Capability → Capability | authored | `parent` (inverse harvested) |
| `depends_on` | `depended_on_by` | CodeUnit → CodeUnit | derived | `tools.code.dep_edges` |
| `governs` | `governed_by` | ADR → CodeUnit | authored | `governs` (path → units) |
| `supersedes` | `superseded_by` | ADR → ADR | authored | `supersedes` |

**Reserved (added later as registry entries — shown to prove extensibility, not built now):**

| name | from → to | properties | when |
| --- | --- | --- | --- |
| `verifies` | Test → CodeUnit / Capability | `test_type: unit\|integration\|e2e\|…` | tests round |
| `fulfills` | Capability → UseCase | — | round 3 (use-cases) |
| `defines` / `uses_term` | Prompt/Query → GlossaryTerm | — | vocab round |

Adding any of these is **one `EdgeType(...)` entry** (+ a node enumerator for a genuinely
new node type). `PropSpec(name, enum=[...])` is how edge properties like `test_type` are
declared and validated — unused by today's edges, reserved capacity.

## Node addressing — `<domain>:<id>`

Every node is referenceable across domains by `<domain-slug>:<id>`:
`code:tools.capability`, `capability:import-transcripts`, `adr:0019`. Bare id within a
domain; prefixed across. The registry maps each **node type** to a `(domain, enumerator,
id-field)` so the harvester can list a domain's nodes and resolve an edge's endpoints.

**Node types (now):** `CodeUnit` (code), `Capability` (capabilities), `ADR` (adr).
**Reserved:** `GlossaryTerm`, `Prompt`, `GraphQuery`, `Spec`, `Test`, `UseCase` — added
with their edges.

## Harvest → one typed graph

A **registry-driven reader** builds the unified edge set with **no new authoring surface**
— it reads edges where they already live:
- **authored** edges from each node's existing frontmatter field (the registry's `field`),
  resolving the target string to a node id per `resolve` (`unit` → code slug; `path` →
  the code units under that path prefix, reusing `tools.adr.code_links`; `slug`/`id`
  direct);
- **derived** edges from their domain tool (`depends_on` from `tools.code.dep_edges`);
- **inverses** computed, never authored twice.

Result: a list of `Edge(type, from="<domain>:<id>", to="<domain>:<id>", props={})`.

## Generated artifacts

- **`docs/graph/index.md`** — the **edge catalog** (every registered edge type: name,
  inverse, from→to, source, properties, **live instance count**; the node-type inventory;
  totals) **plus a meta-schema diagram**: a small Mermaid of *node types ↔ edge types*
  (`Capability --implements--> CodeUnit`, `ADR --governs--> CodeUnit`, …) — the shape of
  the whole graph at a glance.
- **`docs/graph/graph.md`** — the **full cross-domain instance graph**. A single 90-node/
  ~300-edge Mermaid is an unreadable hairball GitHub may not render, so "full" is rendered
  **digestibly**: one Mermaid section **per edge type** (all `implements` edges, all
  `depends_on`, all `governs`, …), each clustered by domain. Every edge instance appears;
  together the sections ARE the complete graph. The capability→code view (deferred from
  the capabilities rounds) is simply the `implements` section.

The **`neighbors` CLI** (below) is the targeted entry point for "what connects to X";
these files are the whole picture.

## The guard — `make graph-check` (non-blocking, exit 0)

- **dangling-endpoint** — an authored edge whose `from` or `to` doesn't resolve to a real
  node → finding. This is the cross-domain integrity check no single domain can do (e.g.
  a `capability:x implements code:renamed-unit`).
- **registry integrity** — an `EdgeType` referencing an unknown node type, or a `field`/
  `resolve` the harvester can't handle → finding.
- **index-sync** — committed `docs/graph/*.md` match a fresh render.

All non-blocking, `return 0`.

**Complement, not replace — with distinct roles (reasoned, not just asserted).** Per-domain
checks stay: when you work inside a domain they are the right tool — fast, scoped,
authoritative for that domain's own links. The graph guard's *unique* value is the case
no per-domain check covers: **cross-domain breakage from a single-domain edit** — you
rename a `CodeUnit` in a *code* commit, `code-check` passes (it doesn't know about
capabilities), yet a capability's `implements` and an ADR's `governs` now dangle. Only a
whole-graph sweep catches that regardless of what you touched. So:

- **`make graph-check`** — the aggregate cross-domain integrity sweep; **wired into
  `.githooks/pre-commit`** (non-blocking, alongside the existing `adr-check`) so every
  commit gets the whole-graph health signal.
- **`make health`** — a new target running every domain's `*-check` + `graph-check`: the
  full sweep for CI / on demand.
- Per-domain `make <domain>-check` — unchanged; for focused work.

The overlap (both the capability check and the graph check validate capability→code
endpoints) is deliberate defense-in-depth; the graph check is the only one that sees the
whole graph.

## CLI

`python -m tools.graph {index | check | neighbors <domain:id>}`. `neighbors` prints the
inbound + outbound edges of a node across all domains — the "what does changing this
touch" traversal in CLI form (e.g. `neighbors code:api` → the capabilities that implement
it, the ADRs that govern it, the code that depends on it).

## Module design — new `tools/graph/`

- `registry.py` — `EdgeType`, `PropSpec`, `EDGES` (the list), `NODE_TYPES` (type →
  domain/enumerator/id map). The single extensible source of truth.
- `reader.py` — `harvest(root) -> list[Edge]` (registry-driven); `nodes(root)` (all
  addressable nodes); `Edge` dataclass.
- `render.py` — `render_catalog(edges, nodes)` (+ the meta-schema diagram),
  `render_graph(edges)` (the full instance graph, per-edge-type Mermaid sections). Pure.
- `check.py` — `Finding`; `check_endpoints`, `check_registry`, `check_index_sync`,
  `run_all(root=".")`. Non-blocking.
- `__main__.py` — `index | check | neighbors <domain:id>`.
- **Makefile** — `graph-index`, `graph-check`, and **`health`** (runs every `*-check` +
  `graph-check`).
- **`.githooks/pre-commit`** — add `graph-check` (non-blocking) alongside `adr-check`.
- **Cascade + registry** — add `("graph", "graph")` to `tools/knowledge`'s `DOMAINS` and a
  row to `docs/index.md`.

## Extensibility walk-through (the point of this round)

Adding the **tests** edge later: append
`EdgeType("verifies", inverse="verified_by", from_type="Test", to_type="CodeUnit",
source="authored", field="verifies", resolve="unit",
properties=[PropSpec("test_type", enum=["unit","integration","e2e"])])` and register the
`Test` node type. No reader/render/check rewrite — they are registry-driven. Same for
`fulfills` (use-cases, round 3). This is validated by a test that a *reserved* edge type
added to a fixture registry harvests + renders without code change.

## Testing

- **Unit** — `harvest` over a synthetic bundle yields the right typed edges (authored
  `implemented_by` → `implements`; derived `depends_on`; `governs` path → the units under
  it); node addressing (`<domain>:<id>`); `check_endpoints` flags a dangling edge, passes
  on a resolvable one; `check_registry` flags an unknown node type; adding a **reserved
  EdgeType to a fixture registry** harvests with no code change (the extensibility test);
  `render_catalog` groups by edge type; `neighbors` returns inbound+outbound. Assert no
  check raises.
- **Smoke** — `make graph-index` writes `index.md` (catalog + meta-schema) + `graph.md`
  (full per-edge-type instance graph) over the real repo; `make graph-check` clean;
  `.githooks/pre-commit` runs `graph-check` non-blocking; `make health` runs the full
  sweep; `make knowledge-check` + `make cli-check` clean (cascade row + registry entry
  added; `graph-*`/`health` targets catalogued).

## Non-goals (this round)

- **Replacing per-domain link checks** — the graph guard complements them.
- **Modifying other domains' generated files** (e.g. injecting derived `implements` into
  `docs/code/*`) — the graph views are self-contained in `docs/graph/`.
- **Building the tests / use-case / vocab edges** — reserved in the registry, added in
  their own rounds.
- **A new authoring surface** — edges are harvested from existing frontmatter/derivations.
- **RDF/triple-store or an actual Neo4j load** — the graph is rendered from files;
  property-graph is the conceptual model, not a new runtime.
- **Blocking** on any finding.

## Capture as ADR

Capture **ADR-0020**: adopt an OKF-extension typed-edge graph model — an extensible edge
registry (property-graph shape, traceability verbs) + `<domain>:<id>` addressing +
harvest/render/guard, complementing per-domain checks. `source:` = this spec. Refines the
domain-family ADRs; supersedes nothing.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-06.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| graph | yes | the new meta-domain | — |
| code / capabilities / adr | yes (read-only) | edges harvested from their existing frontmatter/derivations; no change to them | reuse `tools.code`, `tools.adr.code_links` |
| cli | yes | new `graph-*` + `health` targets → `cli-index` in the plan; `cli-check` clean | — |
| adr | yes | ADR-0020 | — |
| knowledge | yes | cascade row + `DOMAINS` entry for `graph`; `graph-check` added to `.githooks/pre-commit` | — |
| glossary / api / prompts / graph-queries | no | — | their edges are reserved registry entries for later |

**Verdict:** reconciled — graph (subject) + code/capabilities/adr (read-only) consulted;
cli/adr/knowledge reconciled in the plan.
