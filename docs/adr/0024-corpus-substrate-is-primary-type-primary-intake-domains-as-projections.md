---
type: ADR
id: 24
title: Corpus substrate is primary — type-primary intake, domains as projections
status: accepted
date: 2026-08-15
supersedes: []
superseded_by: []
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
- **Discovery is type-primary.** A record is found by *what it is*: `type:` front-matter for
  OKF documents (discovered anywhere), and path/AST for code and tests (walked from the code
  roots `src/`, `tools/`, `tests/`, since `.py` files carry no front-matter). A record's
  declared home folder becomes a **property to check against** (misfiled = a `type:` outside
  its home), never the discovery key.
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
