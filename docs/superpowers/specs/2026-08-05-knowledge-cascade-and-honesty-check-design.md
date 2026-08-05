# Knowledge Cascade + Spec/Plan Honesty Check — design

**Status:** approved by owner 2026-08-05 (brainstorm dialogue).
**Program:** the disclosure layer for the *guarded knowledge graph over the codebase*.
Seven knowledge domains now exist (adr, api, cli, code, glossary, graph-queries,
prompts), each an OKF-conformant bundle with a non-blocking `make <domain>-check`.
This work makes them **discoverable when needed** without loading all of them at
once — and adds an honesty check so a spec/plan can't silently skip a domain it
should have consulted.

## The problem (locked in brainstorm)

The seven domains are real but **six of them have zero presence in `CLAUDE.md` or
any hook** — pull-only, undiscoverable to a fresh session. Only ADR is surfaced
(via a `UserPromptSubmit` hook that injects the full 15-row decision table on every
architectural prompt). Two failure modes to avoid:

- **"All at once"** — dumping seven policy blocks into `CLAUDE.md`. The owner
  explicitly rejected this, and Anthropic's own guidance agrees.
- **"Undiscoverable"** — a domain the agent never learns exists at the moment of
  need, so its `make *-check` never runs.

### Grounding (Anthropic + OKF guidance, researched 2026-08-05)

- *"Keep your CLAUDE.md lightweight… spend most of the tokens on gotchas."* Over
  **80% of Claude Code's own system prompt was removed** for the 5-gen models with
  no eval loss — the direction is *less* upfront, not more.
- Just-in-time context = the agent holds *"lightweight identifiers (file paths,
  stored queries, web links)"* and loads on demand. Our `docs/<domain>/index.md`
  bundles are those identifiers; `make <domain>-check` are the stored queries.
- *"Folder hierarchies, naming conventions… provide important signals that help
  both humans and agents understand how and when to utilize information."*
  Structure **is** the disclosure mechanism.
- OKF's own consumption model: an agent *"starts at the root `index.md`, reads the
  entries it needs, and follows links deeper"*; a folder's **description is the
  primary ranking signal.** We have seven per-domain indexes but **no root index** —
  that missing table-of-contents is the actual gap.

Conclusion: **progressive disclosure here is achieved by structure, not by
injection.** The fix is a discovery cascade + a lightweight pointer + trusting
judgment — not a fleet of content-injecting hooks.

## Components

### 1. The cascade root — `docs/index.md` (authored)

The OKF cascade entry point. One row per knowledge domain: a **one-line
description** (the ranking signal), a link to that domain's own `index.md`, and its
`make <domain>-check`. Authored, not generated — seven near-static rows whose
descriptions want human curation. Example shape:

```markdown
# Knowledge map

Guarded knowledge domains. Land here, then follow the one you're in.

| domain | what it holds | reconcile with |
| --- | --- | --- |
| [adr/](adr/index.md) | architectural decisions — consult before locking one | `make adr-check` |
| [glossary/](glossary/index.md) | canonical vocabulary pinned to code enums | `make glossary-check` |
| [code/](code/index.md) | package/module map: roles, derived deps + I/O, pipeline | `make code-check` |
| [api/](api/index.md) | HTTP surface vs openapi.json | `make api-check` |
| [cli/](cli/index.md) | command surface | `make cli-check` |
| [prompts/](prompts/index.md) | probabilistic components (LLM prompts) | `make prompt-check` |
| [graph-queries/](graph-queries/index.md) | Neo4j read-query registry | `make graphq-check` |
```

`docs/index.md` also briefly names the *non-knowledge* doc areas
(`architecture/`, `product/`, `superpowers/`) so the root is a true map, but those
are pointers only.

### 2. `CLAUDE.md` pointer (lightweight)

Two–three lines, not seven policy blocks:

> ## Knowledge map
> This repo keeps guarded knowledge domains under `docs/` — see
> [`docs/index.md`](docs/index.md) for the map. Each has a non-blocking
> `make <domain>-check`. When you change a surface one covers, consult its bundle
> and run its check.

Existence disclosed cheaply and always; content pulled only when relevant. The
existing ADR-policy section stays (it is the one domain with a hard "consult before
locking" rule) but is trimmed of anything now carried by the cascade.

### 3. The honesty check — after-only, recorded as an addendum

**Trigger:** one hook — `PostToolUse(Write)` matching `docs/superpowers/specs/` **or**
`docs/superpowers/plans/`. It prompts a **mini spec/plan review through the
knowledge-graph lens**: for each domain, *does this spec/plan touch it? was it
consulted / reconciled?*

**Record:** the result is written as an addendum section **in the spec/plan file**:

```markdown
## Knowledge-graph check

Reviewed against `docs/index.md` on <date>.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| adr | yes | ADR-0004 (frozen wire) checked; no supersede | — |
| glossary | yes | adds term "capability" → added to glossary | — |
| code | no | — | no cross-package deps changed |
| api | no | — | — |
| … | | | |

**Verdict:** clean — every touched domain consulted.
```

The addendum **is** the "it was checked" record, and it travels with the artifact
into review. The semantic judgment (does this spec touch domain X?) is the agent's
to reason through — it is not mechanizable cheaply.

### 4. The loop + override (the part with teeth)

When the checklist finds something **in a domain's scope but not consulted /
reconciled** — new context the design didn't account for — it does not silently pass:

- **Mechanical gaps** the agent reconciles directly (e.g. spec introduces a new
  term → add it to the glossary; then note it in the addendum).
- **Design-affecting gaps** escalate to the owner as a decision:
  - **(a) Loop back** — re-enter brainstorm/plan for that piece; likely a design
    change.
  - **(b) Override** — the owner accepts the original and consciously overrules the
    check.
- Either outcome (including override rationale) is recorded in the addendum:
  *"gap: domain X in scope but not reconciled → owner overrode 2026-08-05 because Y."*
  An audit trail, not a silent pass.

**Verdict values:** `clean` · `reconciled` (gaps found and fixed) · `overridden`
(gap accepted by owner, rationale recorded).

### 5. Slim the ADR "before" signal to a provisional pointer

The existing `UserPromptSubmit` hook injects the full 15-row ADR table on every
architectural prompt. Change it to emit a **one-line pointer**, keeping the existing
keyword gate (`is_architectural`):

> Locking an architectural decision? 15 ADRs exist — see `docs/adr/index.md`
> before you do.

Rationale (locked after explicit owner challenge): the before-signal is the only
mechanism that fires *while the decision is being formed*, before a wrong turn —
prevention, not cure — and ADRs are the one domain where a miss (silently
contradicting a locked decision) is catastrophic rather than cheap to fix later.
The cascade (a pull mechanism) and the after-check (a net) do not replicate that.
But re-injecting the full table is wallpaper, and the guidance's direction is *less*
injection, so we slim it to awareness-plus-location.

**Provisional — with a retirement criterion.** This pointer's job is to become
unnecessary. **Retire it once the cascade demonstrably gets ADRs consulted without
it** (owner judgment after living with both; a reasonable bar: two consecutive
architectural specs whose honesty check shows ADRs were consulted with the pointer
removed). No other domain gets a before-signal — that would be scope creep the
cascade is meant to handle.

### 6. Mechanized guard — `tools/knowledge/` + `make knowledge-check`

The only honestly mechanizable slice. Non-blocking, `return 0`, mirrors the
established `check → CLI` shape. Two checks:

- **`check_addendum_present`** — a spec/plan under `docs/superpowers/{specs,plans}/`
  lacking a `## Knowledge-graph check` section → finding. Guards that the check
  *happened*, not that its judgment was right.
  - **Grandfathering:** specs/plans that predate this feature are exempt via an
    explicit list (`tools/knowledge/GRANDFATHERED` or an in-module constant), so the
    guard reports only *new* misses and never becomes persistent noise. (Retroactive
    addenda would falsely claim a review that never happened.)
- **`check_cascade_covers_domains`** — a known domain directory
  (`docs/<domain>/index.md` for domain in the registry) with no row in
  `docs/index.md` → finding. Keeps the cascade root from going stale when domain #8
  (capabilities) lands.

`tools/knowledge/` holds `check.py` (+ `Finding`, `run_all`) and `__main__.py` with
two subcommands: **`check`** (the guard, for `make knowledge-check`) and **`nudge`**
(the `PostToolUse(Write)` honesty-check message — reads the hook's stdin JSON, emits
the reconcile prompt only when the written path is under `docs/superpowers/{specs,
plans}/`, mirroring today's `tools.adr nudge`). No `reader`/`render` — the cascade
root is authored, not generated. The domain registry (the 7 names) lives as a module
constant, the single source of truth both the check and the nudge message read.

### 7. Hook wiring

`.claude/settings.json` after this work:

- `UserPromptSubmit` → slimmed ADR pointer (component 5), still keyword-gated,
  still via the yaml-resolving interpreter script. Stays in `tools.adr context`
  (it is ADR-specific).
- `PostToolUse(Write)` → the honesty-check nudge (component 3), matching
  specs **or** plans. Moves from `tools.adr nudge` to `tools.knowledge` (the
  honesty check spans all domains, not ADR — SRP). The interpreter script is
  generalized to run an arbitrary project module (`scripts/with-project-py.sh
  <module> <args>`) so both `tools.adr` and `tools.knowledge` can share the
  yaml-resolution logic; `scripts/with-adr-py.sh` becomes a thin shim or is
  replaced.

The nudge message points at `docs/index.md`, asks for the per-domain review + the
addendum, and keeps the ADR-capture reminder for specs (a spec that locks decisions
still becomes ADR(s)).

## Module boundaries

| unit | responsibility | authored / code |
| --- | --- | --- |
| `docs/index.md` | cascade root; per-domain ranking signals | authored |
| `CLAUDE.md` (Knowledge map §) | always-on pointer to the cascade | authored |
| addendum template (in this spec + a skill/README) | the checklist shape | authored |
| `tools/knowledge/check.py` | presence guard + cascade-coverage guard | code |
| `tools/knowledge/__main__.py` | `python -m tools.knowledge check` | code |
| `scripts/with-project-py.sh` | shared yaml-resolving interpreter for hooks | code |
| `.claude/settings.json` | hook wiring (before slim, after generalize) | config |
| Makefile | `knowledge-check` target | config |

## Testing

- **Unit** — `check_addendum_present` over a fixture specs dir (one with the
  section → no finding, one without → finding, one grandfathered → no finding);
  `check_cascade_covers_domains` (a domain dir missing from a fixture `index.md` →
  finding, all present → none); assert **no check raises**; `run_all` returns a list.
- **Guard behavior** — the domain registry constant is the single source both the
  cascade-coverage check and any docs reference read from.
- **Smoke** — `make knowledge-check` clean after `docs/index.md` is authored with
  all 7 rows and the existing specs are grandfathered.
- **Hook** — `tools.knowledge` nudge fires on a specs/ *and* a plans/ path, silent
  on a non-spec path; the slimmed `tools.adr context` emits the pointer (not the
  table) on an architectural prompt, silent otherwise.

## Capture as ADR

This brainstorm locks an architectural decision: **adopt a cascade-root discovery
model + spec/plan honesty check; slim the ADR read-hook to a provisional pointer.**
Per `CLAUDE.md` policy, capture it as **ADR-0016** with `source:` = this spec, after
the plan lands. It does not supersede ADR-0015 (the ADR corpus) — it extends the
disclosure model around it.

## Non-goals (this round)

- **A "before" hook for any domain other than ADR** — the cascade handles the rest;
  a second before-signal is scope creep.
- **Generating `docs/index.md`** — authored; the coverage guard is enough to keep it
  honest.
- **Mechanizing the semantic review** — "does this spec touch domain X" is the
  agent's judgment; only *presence* of the addendum is guarded.
- **Blocking** on any finding — everything stays non-blocking, `return 0`.
- **Retroactively stamping existing specs** as reviewed — they are grandfathered,
  not back-dated.
- **Capabilities / use-case domains** — still the next two rounds; unaffected here
  except that the cascade root will gain rows when they ship.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-05.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| adr | yes | captured as ADR-0016 (`source:` = this spec) | extends ADR-0015's disclosure model; no supersede |
| cli | yes | added `make knowledge-check` → ran `cli-index`; `cli-check` clean | new make target enters the CLI catalog |
| code | no | — | `tools/knowledge/` lives under `tools/`, not a `src/` package |
| glossary | no | — | no new domain vocabulary pinned to code enums |
| api | no | — | no HTTP surface change |
| prompts | no | — | no LLM prompt added or changed |
| graph-queries | no | — | no Neo4j read-query change |

**Verdict:** reconciled — every touched domain consulted; ADR-0016 + cli catalog regenerated.
