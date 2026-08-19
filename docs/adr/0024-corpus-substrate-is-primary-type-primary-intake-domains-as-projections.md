---
type: ADR
id: 24
title: Corpus substrate is primary — type-primary intake, domains as projections
status: accepted
date: 2026-08-15
supersedes: []
superseded_by: []
governs:
  - tools/corpus/
tags: [adr, knowledge-management, okf, graph, tooling]
source: docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md
---
## Context
The knowledge tooling grew as parallel domain silos (`tools/<domain>`, each with
reader/render/check/CLI), and the graph (`tools/graph`, ADR-0020) was assembled on top by
importing each domain's loader. That leaves the dependency arrow pointing from the graph to
the domains, and it makes each domain independently responsible for *finding its own records*.
In practice they all do the same thing the same wrong way: `load_capabilities` and
`load_use_cases` are the same shape — glob a hardcoded folder, skip `index.md`, parse
front-matter, then use `type:` as a filter to reject intruders. Record *parsing* is shared
(`parse_front_matter`); record *discovery* is copy-pasted with a different folder baked into
each copy.

Three costs follow. (1) **A folder-vs-type blind spot**: a record of the right `type:` sitting
in the wrong folder is invisible, because discovery keys on the folder and only uses the type
to exclude. (2) **Orphans**: things that exist but sit outside every silo's folder are seen by
no one (the R1 top-level modules — `config`, `celery_app`, `tasks`, `main`,
`run_projection_service` — were exactly this). (3) **A DRY violation**: N copies of discovery,
each a place the next bug hides. Nobody owns "everything in the repo that exists."

## Decision
Invert the model: a single **corpus substrate** is primary, and the domains become
**projections** over it.

- **The repo is the corpus.** Intake scans the whole repo (minus an ignore list), not one
  folder per domain.
- **Discovery is type-primary, by explicit self-declaration.** Every node declares *what it
  is*, in place, the same way across the whole corpus: OKF documents via `type:` (and `kind:`)
  front-matter — `ADR`, `Capability`, `UseCase`, `CodeUnit`, `Term` — discovered anywhere;
  code-derived nodes (`Test`, `GraphQuery`, and future kinds) via a uniform in-code marker
  `# okf: type=… kind=…`; prompt entries via `type:`/`kind:` keys on the YAML entry itself. A
  record's declared home is a **property to check against** (misfiled = declared outside its
  home), never the discovery key.
- **Explicit, not positional or structural.** A node is never recognized by its folder, its
  filename, or a heuristic on its shape (a `test_` prefix, a body that happens to contain
  Cypher). It is recognized only by its own self-declaration. This is uniform — docs and code
  declare identically — so a new node kind costs a *marker*, not a new parser. The accepted
  cost is a one-time migration: existing tests, queries, and prompts must be tagged (they are
  invisible to the graph until they are), and every new one carries its marker. `kind` becomes
  uniform across the corpus as a side effect.
- **One intake, not N.** The duplicated per-domain folder-scanners collapse into the single
  substrate; a domain reader now *selects its node type + its authored edges* from the
  substrate instead of globbing a folder.
- **Migration is incremental.** Introduce the substrate, migrate one domain onto it at a time,
  keep every check non-blocking throughout — never a big-bang rewrite. Each migrated domain's
  node set must equal its pre-migration folder-glob set (no regressions) plus the
  previously-invisible records.

This **refines ADR-0016** (the cascade and honesty check stand; what changes is that domains
are no longer independent silos that each own discovery — they are views over one corpus) and
**refines ADR-0020** (the typed-edge registry, `<domain>:<id>` addressing, and
rendered-from-source stance are unchanged; what changes is that nodes come from one
type-primary corpus intake rather than being harvested per-domain). It supersedes nothing.
Paired with ADR-0025, which makes that substrate a first-class ephemeral traversal surface.

## Consequences
- The folder-vs-type blind spot becomes **structurally impossible**: there is one type-first
  intake, so a misfiled record is found and flagged rather than silently dropped.
- Completeness becomes a property of the single intake (orphan / misfiled / dangling /
  reachability checks over the corpus) instead of something each silo must re-implement — the
  "backward loop" the program's L2 layer builds.
- Discovery lives in one place, so the next discovery bug has one place to hide, not N.
- The migration is the risk: each domain must move onto the substrate without changing what it
  reports for the records it already saw. The equal-node-set test per domain is the guard, and
  the non-blocking posture means a mid-migration gap informs rather than breaks.

## Alternatives considered
- **Patch each reader to also scan by type** (rejected: adds an (N+1)th copy of discovery and
  deepens the DRY problem while calling it a fix — this was the wrong instinct that prompted
  the inversion).
- **Keep domains primary, add a cross-domain completeness check on top** (rejected: the check
  would itself need a corpus-wide, type-primary scan — i.e. the substrate — so we would build
  the substrate anyway, just without the deduplication benefit).
- **Big-bang cutover of all domains at once** (rejected: high risk for no benefit; the
  substrate can stand while domains migrate one at a time behind unchanged, non-blocking
  checks).
- **Recognize code-derived nodes by shape** — a `test_` prefix, a function body containing
  Cypher — instead of requiring an explicit marker (rejected: structural recognition is the
  same failure mode as folder-recognition — fragile, silent on edge cases, and yields no
  uniform `kind`; explicit self-declaration is the whole point, and the owner accepted the
  one-time tagging cost to get it).
- **Per-domain marker formats** (`graphq:`, `# verifies:`, bespoke keys) instead of one shared
  `# okf:` convention (rejected: since every node is tagged anyway, per-domain formats buy
  nothing and cost a new parser per domain and per future node kind; existing `# verifies:`
  edge markers are unaffected — they declare relationships, not type).
