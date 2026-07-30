# ADR + OKF Knowledge System (design)

**Status:** approved by owner 2026-07-30 (recovered brainstorm + reload).
**Origin:** a claude.ai web session that hit its spend limit mid-brainstorm; its
findings were verified against the repo in a follow-up CLI session and reloaded
here. This spec is deliberately **general** — it is about keeping the project's
architectural knowledge, not about any one milestone (contrast M5.2, which is
telemetry-specific).

## Goal

Make the project's architectural decisions **discoverable, durable, and
drift-resistant**, and give agents working the repo an instruction surface that
points at them. Today the decisions are real but buried and can rot silently:

- 4 specs carry explicit `Decisions locked` / `Rejected alternatives`
  (`mine-layers`, `layer4-schema-v2`, `okf-export`, `m46-graphrag-ask`); the
  architecture README has a **"Load-bearing ideas:"** block (line 48).
- There is already one **silent supersession**: `m46-graphrag-ask-design.md`
  overrides the 2026-07-04 spec's "borrow neo4j-graphrag-python" line — in prose,
  in a *different* file, with no back-pointer from the superseded text.
- There is **no `CLAUDE.md`, `AGENTS.md`, or CI** — no agent is told the
  conventions (or the ADR policy) exist.

The system closes a loop: **read** past decisions when making new ones →
**capture** new decisions as ADRs → **guard** the corpus's integrity and detect
drift.

> Captured as ADR-0015.

## What is already in place (why this is low-risk)

- **The OKF v0.1 format exists.** `src/export/renderer.py` renders an OKF bundle
  with a reserved `index.md`, a `RESERVED_PROPS` set, and a `_frontmatter()`
  discipline. The ADR bundle **conforms to that format** rather than reusing the
  code (the renderer emits lens items from Neo4j; ADRs are authored files — a
  separation the M5.2 spec already established for its telemetry domain).
- **Hook + make infra is ready.** `.claude/settings.local.json` exists (no
  `hooks` key yet — the guards are net-new). The `Makefile` already has
  `lint` / `format` / `test` / `help` targets; `adr-check` slots in beside them.
- **The decision corpus already exists** in prose — this is a one-time harvest,
  not new authorship.

## Deliverables

1. **`docs/adr/` — an OKF v0.1-conformant ADR bundle.**
2. **`CLAUDE.md`** at repo root — the agent instruction surface, with an
   ADR-policy section.
3. **~15 backfilled ADRs** harvested from the specs + architecture README,
   including the one real supersession edge.
4. **A five-layer, non-blocking knowledge loop** (read → capture → guard).

## Deliverable 1 — the ADR bundle

`docs/adr/` is an OKF-conformant bundle. Each ADR is one markdown file: OKF
frontmatter + a short [MADR](https://adr.github.io/madr/) body that links **out**
to the milestone spec rather than duplicating it.

```
docs/adr/
  index.md                          # reserved OKF name — GENERATED (id · title · status)
  log.md                            # reserved OKF name — GENERATED (chronological)
  0001-esdb-single-source-of-truth.md
  0002-cqrs-write-read-split.md
  ...
```

```yaml
---
type: ADR
id: 0001
title: EventStoreDB is the single source of truth
status: accepted            # proposed | accepted | superseded | deprecated
date: 2026-07-04
supersedes: []              # list of ADR ids
superseded_by: []           # list of ADR ids
tags: [event-sourcing, write-side]
source: docs/architecture/README.md   # where the reasoning was harvested from
---
## Context
## Decision
## Consequences
## Alternatives considered
```

**Division of labor (locked):** a **spec** describes *how we build a milestone*
(time-bound, disposable once shipped); an **ADR** records *what we decided and
why* (durable, cross-milestone). ADRs stay short and link to the spec for detail.

`index.md` and `log.md` are **generated**, never hand-edited (they carry the
reserved OKF names, so authoring them by hand would drift from the bundle).

## Deliverable 2 — `CLAUDE.md`

Root `CLAUDE.md`, modeled on `getzep/graphiti`'s (the closest public stack
match — Python / Neo4j / FastAPI / pytest / hybrid retrieval), adapted to this
repo's **real** tooling: `flake8` + `black` (not ruff), `pytest` with
`integration` markers, `make run-api` / `run-worker` / `ui-dev`, the `src/`
layout, and provider-specific model notes. It carries an **ADR-policy section**
that states the read/capture rules below in prose, so the policy survives even if
a hook is disabled. Seeded from concrete real-world CLAUDE.md samples, not generic
templates.

## Deliverable 3 — backfilled ADRs (one-time harvest)

~15 ADRs distilled from the existing decision corpus. Harvest sources and the
atomic decisions they carry (illustrative, finalized during implementation):

- **`docs/architecture/README.md` "Load-bearing ideas"** — ESDB as single source
  of truth; CQRS write/read split; event-sourced projections into Neo4j.
- **`mine-layers-design.md`** — the layered mine (ingestion → analysis → lens →
  export) and the overlay-not-rewrite constraint.
- **`layer4-schema-v2-design.md`** — schema v2 decisions.
- **`okf-export-design.md`** — read-side exporter over Neo4j, zero per-lens code.
- **`m46-graphrag-ask-design.md`** — the graphrag approach **and the supersession
  edge**: its ADR sets `supersedes: [<neo4j-graphrag ADR>]` and that ADR gets
  `superseded_by` — the drift the format is built to make visible.

Backfill is **human-curated**, not machine-generated from specs.

## Deliverable 4 — the five-layer knowledge loop

All layers are **non-blocking**: they warn, surface, or remind; none ever fails a
command, blocks a commit, or stops a tool call.

### Read (context in) — *the owner's addition*

Surface relevant ADRs at the moment a new architectural decision is being made,
so prior decisions inform new ones instead of being silently re-litigated.

- **CLAUDE.md policy:** "before locking an architectural decision, consult
  `docs/adr/index.md`; if the decision changes one, write/supersede an ADR."
- **`UserPromptSubmit` hook** (`.claude/hooks/inject-adr-context.sh`): when the
  prompt matches architectural/design intent (brainstorm, design, architecture,
  "should we", "trade-off", spec, plan…), inject the compact ADR index
  (id · title · status) as additional context. No match → no injection (keeps it
  quiet on non-architectural turns).

### Capture (write out)

- **`PostToolUse` hook** on `Write` to `docs/superpowers/specs/*` — after a design
  doc lands (i.e., a brainstorm concluded), print a reminder: "this spec locks
  decisions — capture them as ADR(s) and set `source:` to this spec."

### Guard (integrity + drift)

A single tool, `make adr-check` → `python -m tools.adr check`, runs three
non-blocking validations (exit 0 always; findings to stderr):

- **Structural integrity** — frontmatter schema, unique ids, `index.md`/`log.md`
  in sync with the files on disk, and every `supersedes`/`superseded_by`
  **bidirectional**.
- **Spec-references-ADR** — a spec containing `Decisions locked` /
  `Rejected alternatives` but referencing no ADR id → warn (the guard against
  silent supersession like the m46 case).
- **Staleness** — if an ADR's `source` file has git-changed more recently than
  the ADR itself → warn (the decision's basis may have moved).

`adr-check` also runs from a **non-blocking git `pre-commit` hook** (prints
warnings, exits 0) so it fires regardless of which agent — or human — commits.

## Module design

Mirrors the repo's guard → read → compute → render → write split:

```
tools/adr/
  __init__.py
  model.py        # ADR frontmatter dataclass + parse/validate (the schema lives here)
  index.py        # (re)generate index.md + log.md from the bundle — pure
  check.py        # structural + spec-reference + staleness checks — returns findings
  scaffold.py     # `adr new <title>` — next id + template stub
  __main__.py     # `python -m tools.adr {index|check|new}` CLI
.claude/hooks/
  inject-adr-context.sh   # UserPromptSubmit — read side
  nudge-adr.sh            # PostToolUse(Write→specs) — capture side
CLAUDE.md
docs/adr/                 # the bundle (deliverables 1 & 3)
Makefile                  # + adr-check, adr-index targets
```

`model.py` / `index.py` / `check.py` are pure and unit-testable; only
`__main__.py` and the hooks touch I/O or the environment.

## Build order (for the plan)

1. **Corpus** — `tools/adr` (`model`, `index`, `check`, `new`) + the ~15
   backfilled ADRs + generated `index.md`/`log.md`. The bundle must exist before
   anything can guard or surface it.
2. **`CLAUDE.md`** — references the now-real ADR policy.
3. **The loop** — hooks (`inject-adr-context`, `nudge-adr`), `make adr-check` /
   `adr-index` wiring, and the non-blocking `pre-commit` hook.

## Testing

- **Unit** — `model` parse/validate (good + malformed frontmatter); `index`
  output shape and idempotence; `check` over synthetic bundles asserting each
  finding fires (schema violation, duplicate id, one-directional supersede edge,
  out-of-sync index, a spec-with-decisions-no-ADR, a stale source). All checks
  return findings; **none raises**.
- **Smoke** — the `inject-adr-context` intent matcher classifies a small labeled
  set of prompts (architectural vs not) correctly; `make adr-check` on the real
  `docs/adr/` exits 0.

## Non-goals

- **Any blocking enforcement.** No hook, check, or commit ever fails. Adoption is
  by visibility, not gates.
- **Machine-generating ADR content** from specs — backfill is human-curated,
  one-time. The tooling generates only `index.md`/`log.md`.
- **Putting ADRs in the OKF *export* bundle** — the ADR bundle merely conforms to
  the OKF format; it is not part of the interview-mining export.
- **A web UI / dashboard** for ADRs — the bundle is read as markdown in the repo.
- **`AGENTS.md`** now — CLAUDE.md is the only surface this harness auto-loads; add
  an AGENTS.md pointer later only if another agent tool works the repo.
