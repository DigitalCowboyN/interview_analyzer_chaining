# Graph lifecycle — Round 1: the forward loop (design)

**Status:** approved by owner 2026-08-15 (brainstorm dialogue).
**Program:** the guarded knowledge graph is a *living diagnostic* asset — but only if drift
surfaces automatically as the repo evolves. Today the forward loop is the weak link:
**pre-commit runs only 2 of ~11 domain checks** (`adr`, `graph`), there is **no CI**, and
pre-commit only *checks* (never regenerates). So editing capabilities / tests / glossary /
code can silently drift until someone remembers `make health`. This round makes drift
surface on every change — fast locally (only the domains you touched), complete in CI.

This is **Round 1** of the lifecycle mini-program (R2 = backward-completeness reader audit;
R3 = the durable lifecycle doc + policy decisions).

## Framing (locked in brainstorm)

- **Changed-domain pre-commit + full `make health` in CI.** Local commits run only the
  checks for the domains whose surfaces the staged files touch (fast, relevant); CI runs
  the whole sweep as the safety net.
- **Advisory everywhere, except index-sync blocks in CI.** Judgment findings (unclaimed
  unit, `UNVERIFIED` use-case, staleness, unmapped test) stay non-blocking — reported,
  never fail — honoring ADR-0016's *visibility-not-gates* principle. But the **mechanical
  "generated index is out of date"** case *blocks CI*: that's objective drift ("you didn't
  run `make *-index`") a living asset must not accumulate. Implemented robustly by
  **regenerate-then-`git diff`**, not by parsing finding text.
- **Non-blocking stays non-blocking locally.** Pre-commit is always `exit 0` — it informs,
  never gates. Only CI enforces (and only index freshness).

## The surface registry — each domain declares what it reconciles

The single source of truth for domains (`tools/knowledge/check.py::DOMAINS`) gains, per
domain, the **path prefixes it reconciles** — so "which check does this file affect?" has
one authoritative answer, not a mapping that drifts.

```python
@dataclass
class Domain:
    slug: str            # docs/<slug>/  (cascade row + graph addressing)
    make: str            # `make <make>-check` / `python -m tools.<make>`
    surfaces: List[str]  # path prefixes whose change can cause this check to find new drift


DOMAINS = [
    Domain("adr",           "adr",      ["docs/adr/", "src/"]),          # governs code
    Domain("api",           "api",      ["src/api/", "frontend/openapi.json"]),
    Domain("cli",           "cli",      ["Makefile", "tools/"]),
    Domain("code",          "code",     ["src/", "tools/"]),
    Domain("capabilities",  "capability", ["docs/capabilities/", "src/", "tools/"]),
    Domain("glossary",      "glossary", ["src/", "docs/glossary/"]),
    Domain("graph",         "graph",    ["docs/"]),                       # cross-domain
    Domain("graph-queries", "graphq",   ["src/projections/", "docs/graph-queries/"]),
    Domain("prompts",       "prompt",   ["src/", "docs/prompts/"]),
    Domain("use-cases",     "usecase",  ["docs/use-cases/"]),
    Domain("tests",         "testmap",  ["tests/"]),
]
```

`check_cascade_covers_domains` and any other `DOMAINS` consumer are updated to the dataclass
(`d.slug` / `d.make`) — the only consumers live in `tools/knowledge`, so blast radius is
one package. Surfaces are intentionally slightly broad (err toward running a check when a
change *could* matter); **CI's full sweep is the guarantee**, pre-commit is fast feedback.

## The changed-domain resolver

`tools/knowledge/surfaces.py`:

```python
def changed_domains(files: Iterable[str], domains=DOMAINS) -> list[str]:
    """The `make`-names of domains whose surface any of `files` touches (sorted, deduped)."""
    hit = set()
    for f in files:
        f = f.replace("\\", "/")
        for d in domains:
            if any(f.startswith(p) for p in d.surfaces):
                hit.add(d.make)
    return sorted(hit)
```

CLI: `python -m tools.knowledge changed-domains` reads newline-separated paths from **stdin**
and prints the touched domains' `make`-names, one per line. (`graph` is cross-domain and
always run by the hook regardless, so callers add it.)

## Pre-commit — fast, relevant, advisory

Rewrite `.githooks/pre-commit` to run only the touched domains' checks, plus `graph`
(cross-cutting), all non-blocking:

```bash
#!/usr/bin/env bash
# Non-blocking drift report for the domains this commit touches (+ the cross-domain graph).
files="$(git diff --cached --name-only)"
domains="$(printf '%s\n' "$files" | bash scripts/with-project-py.sh tools.knowledge changed-domains 2>/dev/null)"
for d in $domains graph; do
    bash scripts/with-project-py.sh "tools.$d" check || true
done
exit 0
```

A commit that touches nothing a domain covers runs only `graph` (fast). Editing a capability
runs `capability` + `graph`. `exit 0` always — informs, never blocks.

## CI — the full safety net

New `.github/workflows/health.yml`, on push + pull_request:

1. **Setup:** checkout, `actions/setup-python@v5` with 3.10.7, `pip install -r requirements.txt`.
   (The checks import the app with placeholder keys — see `tools/api/reader.py` — and
   otherwise only read files, so no real API keys / services are needed.)
2. **Advisory sweep (never fails):** run `make health` and surface all findings in the job
   log / step summary. This is the complete drift report; it does not fail the build.
3. **Index-freshness gate (blocks):** regenerate every generated file
   (`make adr-index capability-index code-index cli-index api-index glossary-index
   graphq-index prompt-index usecase-index testmap-index graph-index`) then
   `git diff --exit-code` — a non-empty diff means committed indexes were stale
   (`make *-index` wasn't run) → **fail**, printing the diff so the fix is obvious.

The `make health` loop and the index-target list are catalogued by `cli-check`, keeping this
workflow's target set honest against the real CLI surface.

## ADR

Capture **ADR-0023** (refines ADR-0016): the forward loop is *advisory by default, with a
single mechanical exception* — generated-index freshness is enforced in CI via
regenerate-then-diff, while all judgment findings remain non-blocking visibility. This is a
deliberate, bounded refinement of *visibility-not-gates*, not a reversal. `source:` = this spec.

## Non-goals (this round)

- **Making judgment checks blocking** anywhere — only index freshness is enforced, only in CI.
- **Auto-regeneration on commit** (self-healing) — pre-commit diagnoses; the human/agent
  regenerates. (A `make regen-all` convenience target is fine; auto-committing generated
  files from a hook is out of scope.)
- **The backward-completeness reader audit** (Round 2) and **the durable lifecycle doc /
  policy decisions** (Round 3).
- **Per-file precision** in the surface map — slightly-broad surfaces + CI's full sweep is
  the intended tradeoff.

## Testing

- **Unit (`tests/knowledge/`):** `changed_domains(["docs/capabilities/x.md"])` →
  `["capability"]` (not the whole world); a `src/api/foo.py` change →
  includes `api`, `code`, `capability`, `glossary`, `prompt` (its surface hits), excludes
  `usecase`/`testmap`; an unmatched path → `[]`; the `DOMAINS` dataclass migration keeps
  `check_cascade_covers_domains` passing (every `d.slug` still checked).
- **Smoke:** `python -m tools.knowledge changed-domains <<< "docs/use-cases/x.md"` prints
  `usecase`; the rewritten `.githooks/pre-commit` runs clean on a real staged change
  (touches a capability → runs capability + graph, exit 0); `make cli-check` clean (the CI
  target list catalogued); the CI workflow YAML is valid and its steps match the `make`
  targets that exist.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-15.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| knowledge | yes | `DOMAINS` → dataclass with `surfaces`; new `changed_domains` + CLI | single source of truth for domain surfaces |
| cli | yes | new `make` targets (regen-all, if added) + the CI target list → `cli-index`; `cli-check` clean | — |
| adr | yes | ADR-0023 (refines 0016) | — |
| all domain checks | yes (read-only) | invoked by the new pre-commit/CI wiring; their logic unchanged | forward-loop plumbing only |
| graph / code / capabilities / use-cases / tests / glossary / api / prompts / graph-queries | no (logic) | — | unaffected internals |

**Verdict:** reconciled — knowledge (surface registry + resolver) is the subject; cli/adr
reconciled; the domain checks are invoked, not modified.
