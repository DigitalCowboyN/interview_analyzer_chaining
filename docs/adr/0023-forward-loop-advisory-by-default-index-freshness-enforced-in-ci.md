---
type: ADR
id: 23
title: Forward loop — advisory by default, index freshness enforced in CI
status: accepted
date: 2026-08-15
supersedes: []
superseded_by: []
tags: [adr, knowledge-management, okf, context-engineering, tooling, ci]
source: docs/superpowers/specs/2026-08-15-lifecycle-r1-forward-loop-design.md
---
## Context
The guarded knowledge graph only stays honest if drift surfaces automatically as the
repo evolves — the *forward loop*. Before this decision that loop was the weak link:
the pre-commit hook ran only 2 of ~11 domain checks (adr, graph), there was no CI, and
the hook only *checked*, never regenerated. Editing capabilities / tests / glossary /
code could silently drift until someone remembered `make health`.

ADR-0016 established the cascade and its *visibility-not-gates* principle: the checks
report drift, they do not block. That principle is right for judgment findings (an
unclaimed unit, an UNVERIFIED use-case, a stale glossary term) — those need a human's
read, not a hard stop. But it left one mechanical case ambiguous: a committed generated
index that no longer matches a fresh render is not a judgment call, it is objective drift
("you didn't run `make <domain>-index`") that a living asset must not accumulate.

## Decision
The forward loop is **advisory by default, with a single mechanical exception**.

- **Changed-domain pre-commit, always non-blocking.** `tools/knowledge/check.py::DOMAINS`
  gains per-domain `surfaces` (path prefixes); `tools/knowledge/surfaces.py::changed_domains`
  maps staged files to the domains they touch. The hook runs only those domains' checks
  plus the cross-cutting `graph` check, and always exits 0 — fast, relevant, informative.
- **Full sweep + freshness gate in CI.** `.github/workflows/health.yml` runs the complete
  `make health` drift report (advisory, never fails), then an **index-freshness gate**:
  `make regen-all` followed by `git diff --exit-code`. A non-empty diff means committed
  indexes were stale → the job fails, printing the diff. This is implemented by
  regenerate-then-diff, not by parsing finding text.

This **refines ADR-0016** — it does not reverse it. Every judgment finding remains
non-blocking visibility. The only thing that can now fail a build is a generated index
that its author forgot to regenerate: a deterministic, mechanical fact, not a matter of
taste.

## Consequences
- Drift for any domain surfaces on the commit that introduces it (locally, fast) and on
  every push (CI, complete) — no more silent drift until the next manual `make health`.
- Contributors cannot merge stale generated indexes; the fix is mechanical and the failure
  message says exactly what to run (`make regen-all`).
- `DOMAINS` is now the single source of truth for both the cascade rows and the
  changed-domain surface map; adding a domain means one row (plus its `docs/index.md` row).
- Regeneration must stay deterministic (same inputs → byte-identical output) or the gate
  flaps; `regen-all` runs the domain indexes first and `graph` last (it aggregates them).
- The pre-commit surfaces are intentionally slightly broad (err toward running a check when
  a change *could* matter); CI's full sweep is the guarantee, the hook is fast feedback.

## Alternatives considered
- **Keep everything advisory (no CI gate).** Rejected: stale generated indexes then
  accumulate unnoticed, which is exactly the drift the graph exists to prevent, and the fix
  is mechanical — a poor fit for human judgment.
- **Make judgment checks blocking too.** Rejected: reverses ADR-0016's core principle;
  judgment findings need a human read, and hard-stopping on them punishes honest
  work-in-progress.
- **Auto-regenerate and commit from the hook (self-healing).** Rejected as out of scope:
  a hook that rewrites and stages generated files hides drift instead of surfacing it, and
  couples committing to regeneration. The loop diagnoses; the human/agent regenerates
  (`make regen-all` is the convenience).
- **Run all domain checks on every commit.** Rejected: slow and mostly irrelevant per
  commit; the changed-domain resolver gives the same coverage guarantee (via CI) with fast
  local feedback.
