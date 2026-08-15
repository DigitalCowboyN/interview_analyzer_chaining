# First-class knowledge graph — program design

**Status:** proposed (brainstorm dialogue with owner, 2026-08-15).
**Type:** umbrella / program spec. Establishes the north star, the architecture, and the
phased roadmap. Each phase gets its own implementation spec + plan; this document is not a
single plan.

## North star

**The knowledge graph is a first-class, reliable model of the whole-repo corpus that
anything — a human, a coding agent, a hook — can query on demand: materialize and walk a
subgraph from *any* node (or any entry set), in *any* direction, to *any* depth (bounded or
to exhaustion), at *any* time, and trust that it is current.**

The graph is **spontaneous short-term memory**: it lives for the moment it is needed. A walk
materializes the exact relevant subgraph — each node carrying its **claim + context** (the
record's own content) — and that subgraph *is* the working context handed to a model. It is
rebuilt from source each time; it is not a store that persists between uses. (We can imagine
wanting to cache it — e.g. to make CI cheaper — but that is a someday-maybe, explicitly out
of scope. See Materialization.)

Everything else in this program exists to serve that reliability. The causal chain, made
explicit:

- The graph must **work** for arbitrary, unpredictable queries → **priority #1**.
- You can only *rely* on it if it is **complete and current** → that is the entire purpose
  of the up-to-date mechanisms. They are not the point; they are the guarantee.
- Once it is reliable you can **expand** it and **attach governance** keyed on shapes (an
  ADR scoping a later code change; a policy that fires on a certain node-and-depth pattern)
  → the payoff.

This is a tool for **mapping and governing our own repo**. It is entirely separate from the
Neo4j read-model, which projects transcript/interview data — a different graph for a
different purpose. Do not conflate them.

## Framing (locked in brainstorm)

- **Rebuild from source, ephemeral.** The repo is the single source of truth; the graph is
  materialized *from* it on demand and lives only for the moment. No second store that can
  drift. "Generate fresh, then walk."
- **The repo is the corpus.** Not `docs/`. Code, tests, ADRs, capabilities, use-cases,
  glossary, prompts, graph-queries — all of it. Records are not confined to one folder.
- **Type-primary intake.** A record is discovered by *what it is* — `type:` frontmatter for
  OKF documents; path/AST for code and tests — not by which folder holds it.
- **Graph-first priority.** The first-class graph (materialize + traverse) must work before
  the governance payoff is built. Completeness/currency make it trustworthy; governance uses
  it.
- **Visibility, not gates** (inherited from ADR-0016/0023): checks report, they do not block,
  except the mechanical freshness case.

## The problem today: the architecture is inverted

The current system is a set of parallel domain silos (`tools/<domain>` each with
reader/render/check/CLI). The graph (`tools/graph`) is a thin aggregation bolted on top. Three
facts, all verified in the code:

1. **Discovery is duplicated per domain.** `load_capabilities` and `load_use_cases` are the
   same shape — glob a hardcoded folder, skip `index.md`, read, `parse_front_matter`, then
   filter by `type:`. ADR has its own variant in `load_bundle`. Record *parsing* is shared
   (`parse_front_matter`); record *discovery* is copy-pasted with a different folder baked
   into each copy. There is no single "find the records" layer.
2. **The graph depends on the domains, not the reverse.** `tools/graph/reader.py` imports
   every domain's loader (`load_capabilities`, `load_units`, `load_use_cases`, `load_bundle`,
   `load_tests`). The dependency arrow points from the graph to the silos; the graph is
   derived, not primary.
3. **"Territory" is a folder, per silo** — because each silo independently decided where to
   look. Nobody owns "the whole corpus."

Every problem we have hit is a symptom:

- The **folder-vs-type blind spot**: each reader globs its folder and uses `type:` as a
  *filter to reject intruders* instead of the *key to find records anywhere*, so a record of
  the right type in the wrong folder is invisible.
- **Depth-1 traversal**: `neighbors` shows one hop, in/out, from any node — but there is no
  depth, no multi-hop walk, no type/folder entry lens. There is no substrate to walk.
- The **R1 orphan files** (`src/config.py`, `celery_app`, `tasks`, `main`,
  `run_projection_service`): invisible because no silo's folder covered them and nothing owns
  "everything that exists."
- A straight **DRY violation** — N copies of discovery, each a place the next bug hides.

## Architecture: four layers

### L0 — Substrate (the inversion)

One corpus model is primary. It scans the whole repo once and classifies every record by what
it *is*:

- **OKF documents** (ADR, Capability, UseCase, and future GlossaryTerm / Prompt / GraphQuery /
  Spec): discovered by `type:` frontmatter, **anywhere in the repo** (minus an ignore list).
  The record's declared home folder becomes a *property to check against* (misfiled = a type
  outside its home), never the discovery key.
- **Code & tests**: discovered by walking the code roots (`src/`, `tools/`, `tests/`) via
  path/AST — `.py` files have no frontmatter, so their "type" is "source in the code tree,"
  and the orphan case is "a source file no unit claims" (R1's `check_top_level_modules`,
  generalized to all roots).

A **Node** carries: an address (`<type>:<local-id>`, e.g. `code:api`, `adr:23`), its type,
its provenance (the path it came from), derived/authored **properties**, and its **claim +
context** (the record's own content — so a walk carries an ADR's scope with it). **Edges**
stay the existing registry-driven typed set (`implements`, `depends_on`, `governs`,
`supersedes`, `fulfilled_by`, `verifies`, `child_of`), each with inverse, direction, and
authored/derived origin; the registry is how new node/edge types are added.

**Domains become projections over the substrate.** A domain reader stops globbing a folder and
instead *selects its node type + its authored edges* from the one intake. Discovery happens
once, type-first, corpus-wide. The folder-vs-type blind spot becomes structurally impossible;
the duplicated discovery collapses to one layer.

Migration is **incremental** — introduce the substrate, migrate one domain onto it at a time,
keep every check non-blocking throughout. Never a big-bang rewrite.

### L1 — Traversal engine (first-class query)

The capability the whole program is named for. A single traversal primitive:

```
walk(entry, direction, depth) -> Subgraph
```

- **entry**: a node address, or a *selector* — by type (`type=Capability`), by folder
  (`under=src/api/`), or by predicate. (Type primary; folder and predicate also supported —
  enter from any lens.)
- **direction**: `out` | `in` | `both` (follow edges forward, backward, or both).
- **depth**: an integer, or *to exhaustion* (walk until it hits an end). Choosing the depth up
  front is what makes discovery *progressive*.
- **returns**: the materialized **Subgraph** — its nodes (with claim + context) and the edges
  among them. This subgraph is the artifact: the context you feed a model, the input a policy
  matches on, the answer to "what governs / depends on / verifies this?"

Materialized fresh from source on every call ("generate before you walk"). This upgrades
today's depth-1 `neighbors` into the real primitive; `neighbors` becomes `walk(node, both, 1)`.
CLI: extend `tools.graph` with a `walk` subcommand (`--dir`, `--depth N|full`, entry
selectors).

### L2 — Completeness & currency (the guarantee)

The graph is only trustworthy if it is complete and current. This is the former "R2 backward
loop," now a property of the substrate rather than N reader patches:

- **Orphan**: a corpus item that exists but has no node (a source file no unit claims — R1's
  guard generalized to every root and every OKF type).
- **Misfiled**: a record whose `type:` places it outside its declared home folder.
- **Dangling**: an edge whose endpoint is not a real node (the existing graph-check; kept).
- **Reachability** (uses L1): "what code is reached by no ADR / UseCase / Capability?" — a walk
  from every intent node, flagging the unreached. A richer gap signal than a flat
  set-difference, and only possible once L1 exists.

**Currency** is the forward loop (R1, already shipped: changed-domain pre-commit + CI freshness
gate) plus the "regenerate before you walk" discipline. All checks non-blocking except the
mechanical freshness gate (ADR-0016/0023).

### L3 — Governance on shapes (the payoff)

Once the graph is reliable, attach rules/policies/hooks keyed on graph **shapes** and
**traversals**. The canonical example: *editing a CodeUnit → walk inbound `governs` → surface
the governing ADR's scope + context → flag a change that drifts out of that scope.* The ADR's
context is naturally in-scope because the walk carries each node's claim + context.

The **mechanism is deliberately undecided** — a git hook, a rule the agent reads, a CI check,
or something we have not thought of. L3 establishes the *capability* (query shapes, attach
policy to them), not any specific policy. It also owns **lifecycle**: superseded ADRs and
transitional decisions must not pollute active context (the classic knowledge-graph failure
mode — a decision graph decaying into stale noise). The parked decisions from earlier rounds
land here too: infra-coverage policy, the `verified_units` marker question, inbound dep-edges.

## Roadmap and how R2/R3 map on

| Phase | Layer(s) | Was | Priority |
| --- | --- | --- | --- |
| Substrate | L0 | (new — the inversion) | **1st** — the graph must work |
| Traversal | L1 | R2 "B" | with/after L0 |
| Completeness & currency | L2 | R2 "A" (backward loop) | after L0; L1 enables reachability |
| Governance & lifecycle | L3 | R3 | last — the payoff, captured now so it is not lost |
| Currency (forward loop) | — | **R1, DONE** (PR #39) | folded in retroactively |

- **R2 = L0 + L1 + L2** — the substrate, the traversal engine, and completeness, as one
  design (A + B together, on the inverted model).
- **R3 = L3** — governance + lifecycle + the parked policy decisions.
- Each phase is decomposed into its own implementation spec + plan when we reach it; this
  umbrella locks the architecture and the order.

## Materialization decision

**Rebuild from source, every time. No persisted store.** The repo is truth; the graph is a
faithful, ephemeral materialization of it. This keeps "source is truth" absolute (zero drift,
no sync problem) and matches the short-term-memory framing. A **materialized cache** (rebuilt
on change, e.g. to make deep CI walks cheaper) is an acknowledged possible future — but it is
**out of scope** and unbuilt until a real need appears (YAGNI). We revisit only if walks get
slow.

## Relationship to existing decisions

The inversion changes the ADR-0016-era assumption that domains are primary and the graph is a
derived cascade view. This program will capture an **ADR** (refines/extends ADR-0016) stating
that the corpus substrate is primary and domains are projections over it. ADR-0023 (forward
loop) already supplies the currency half of L2 and is unchanged. New ADRs are captured per
phase as decisions lock (`python -m tools.adr new …`), per CLAUDE.md policy.

## Non-goals

- **A persisted / cached graph store** — deferred until a real need (CI cost) appears.
- **Reusing the transcript Neo4j** — a different graph; not touched.
- **Deciding specific L3 policies or the hook-vs-rule mechanism** — L3 builds the capability;
  the policies are separate, later decisions.
- **A big-bang rewrite** — domains migrate onto the substrate incrementally, checks staying
  non-blocking throughout.
- **Making judgment checks blocking** — only the mechanical freshness gate enforces
  (ADR-0023).

## Testing / verification (per phase, high level)

- **L0**: intake finds every `type:`-declared record repo-wide (including a deliberately
  misfiled fixture) and every code/test root; a migrated domain's node set is identical to its
  pre-migration folder-glob set (no regressions) plus the previously-invisible records.
- **L1**: `walk` from a known node returns the expected subgraph at depth 1, depth N, and to
  exhaustion, in each direction; folder and type selectors return the right entry sets;
  `neighbors` parity as `walk(node, both, 1)`.
- **L2**: orphan/misfiled/dangling/reachability each fire on a planted gap and stay silent on a
  clean corpus; all non-blocking.
- **L3**: a shape query (e.g. CodeUnit → governing ADR) returns the ADR's context; a planted
  out-of-scope change is flagged; superseded decisions are excluded from active context.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-15. This is a program/architecture spec; it
reshapes how the graph domain relates to the others, so nearly every domain is downstream.

| domain | touched by the program? | note |
| --- | --- | --- |
| graph | yes — becomes the substrate (L0) + traversal engine (L1) | the center of gravity shifts here |
| knowledge | yes — `DOMAINS`/`surfaces` (R1) is the seed of type/folder selectors and the corpus registry | reuse, extend |
| code / capabilities / use-cases / tests / glossary / prompts / graph-queries / adr / api / cli | yes (eventually) — each migrates from folder-scanner to projection over the substrate | incremental, non-regressing |
| adr | yes — new ADR captures the inversion (refines 0016); `governs` is L3's canonical shape | — |

**Verdict:** program-level — the substrate/traversal layers are the subject; every domain is a
downstream projection migrated incrementally. Per-phase specs will carry their own, narrower
knowledge-graph check.
