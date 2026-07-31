# CLI-Surface Domain — generated + reconciled (design)

**Status:** approved by owner 2026-07-31 (brainstorm dialogue).
**Program:** sub-project "D" of the *guarded knowledge graph over the codebase*.
Follows B (ADR↔code linking, PR #20). Builds a **generated** catalog of the
project's command surface + a non-blocking guard that keeps documentation honest.

## Goal

The project's command surface (59 Makefile targets + 7 `python -m <module>` entry
points) is real and machine-discoverable, but its *documentation* is a
hand-maintained `make help` echo block that already drifted (it omits `adr-check`
and `adr-index`). Make the command surface **self-describing and drift-checked**:

1. Every command is catalogued, tagged **everyday** vs **internal**.
2. `make help` (human-facing) is generated from the commands — it can never go
   stale again, and shows only everyday commands.
3. `docs/cli/index.md` (agent/machine-facing) lists **all** commands, labeled — so
   an agent working the repo can look up the exact command instead of guessing.
4. A guard flags docs that reference a command that no longer exists, and new
   commands that arrived without a description.

Consumers are humans **and agents** — the full labeled catalog exists specifically
so an agent has a complete, current command map.

## Design decisions (locked in brainstorm)

- **Generated, not authored.** The catalog and help are rendered from the real
  surface (Makefile targets + their `##` doc comments, and the `__main__.py`
  entry points). Near-zero authoring; nothing to keep in sync by hand.
- **Catalog everything, differentiate.** All targets are documented and
  catalogued; a per-target tag marks **everyday** vs **internal**. `make help`
  shows everyday only; `docs/cli/index.md` shows all, labeled. (Rejected:
  cataloguing only a public subset — a full map is what helps the agent.)
- **`make help` becomes generated** — the hand-typed echo block is replaced by
  rendered output (same look, self-updating source).
- **Stdlib-only tooling** — `tools/cli` imports no third-party packages (notably
  not `yaml`), so generated `make help` runs under any interpreter and never hits
  the PATH/deps trap the ADR hooks did.

## The self-doc convention

A target's documentation lives in a trailing `##` comment on its rule line:

```makefile
test-unit: ## Run unit tests only (no integration markers)
run-api:   ## Start the FastAPI app
wait-neo4j-test: ##@ Wait for the test Neo4j to accept connections
```

- `## <description>` → **everyday** command (shown in `make help` + catalog).
- `##@ <description>` → **internal** command (catalog only, labeled `internal`;
  hidden from `make help`).
- A target with **no** `##` comment → flagged by `cli-check` as *undocumented*
  (informational), so the catalog stays complete as new targets land.

The 7 `python -m <module>` entry points (any package with a `__main__.py`:
`src/ask`, `src/enrichment`, `src/export`, `src/ingestion`, `src/lens`,
`src/resolution`, `tools/adr`) are catalogued with their **module docstring's
first line** as the description; they are treated as everyday commands.

## Generated artifacts

`tools/cli` renders, on demand:

- **`make help`** — grouped list of everyday commands (`name — description`),
  rendered from the `##` docs. Replaces the current echo block.
- **`docs/cli/index.md`** — the full catalog: every make target (everyday +
  internal, labeled) grouped, plus the `python -m` entry points. Generated,
  never hand-edited (guarded for sync).

## The reconciliation guard — `make cli-check` (non-blocking)

`python -m tools.cli check` returns findings; **never raises, always exits 0**
(same contract as `adr-check`):

1. **docs-reference-real-command** — scan `CLAUDE.md` and `README.md` for command
   mentions (`make <target>`, `python -m <module>`); flag any that is **not** in
   the real surface. This is the core value — it kills the "documented command
   that vanished" class (e.g. the `make help` omission that motivated this).
2. **catalog-in-sync** — `docs/cli/index.md` matches the current real surface
   (like the `adr-index` sync check); if a target/entrypoint was added or removed
   without regenerating → finding.
3. **undocumented-target** (informational) — a Makefile target with no `##`
   comment; keeps catalog coverage from rotting.

## Module design — new `tools/cli/`

Mirrors `tools/adr`'s reader → render → check → CLI split. **Stdlib only.**

- `reader.py` — `parse_makefile(path) -> list[Command]` (target name, deps-free,
  `##` description, `everyday|internal` tag) + `module_entrypoints(root) -> list[Command]`
  (walk for `__main__.py`, read the package docstring's first line). Returns a
  simple `Command` dataclass (`name, kind, description, visibility`).
- `render.py` — `render_help(commands) -> str`, `render_catalog(commands) -> str`.
  Pure.
- `check.py` — `check_docs_reference_real`, `check_catalog_in_sync`,
  `check_undocumented`, `run_all(root=".") -> list[Finding]`. Non-blocking.
- `__main__.py` — `python -m tools.cli {help|index|check}`.
- **Makefile** — `help:` → `@$(PYTHON) -m tools.cli help`; add `cli-index`
  (writes `docs/cli/index.md`) and `cli-check`. The existing echo block is deleted.

`Command` and `Finding` may be local to `tools/cli` (kept independent of
`tools/adr` so the two domains evolve separately; the shared idea is the pattern,
not the code).

## Reuse & relationship to B

Reuses the **pattern** (reader→render→check, a generated reserved artifact, the
doc-mention scan like `check_specs_reference_adr`), not `tools/adr` code. Commands
do not carry `governed-by` markers in v1 — linking commands to the ADRs that
govern them is the authored overlay, deferred (below).

## Testing

- **Unit** — `parse_makefile` over a synthetic Makefile (everyday `##`, internal
  `##@`, undocumented target, a non-target line); `module_entrypoints` over a
  synthetic tree; `render_help` (everyday only) and `render_catalog` (all,
  labeled) shape; each check fires on a fixture (doc names a missing command;
  catalog out of sync; undocumented target). Assert **no check raises**; assert
  `tools.cli` imports nothing outside the stdlib.
- **Smoke** — after migration, `make help` renders the everyday commands; `make
  cli-check` on the real repo exits 0 and is clean (or lists only known
  informational undocumented targets).

## Migration (one-time)

Move the text from the current `help:` echo block into `##` comments on the
matching targets; add short `##`/`##@` docs to the remaining targets (everyday vs
internal by judgement); delete the echo block; wire `help:` to the generator;
generate `docs/cli/index.md`; `make cli-check` clean.

## Non-goals (v1)

- **Deep argparse subcommand introspection** — catalog `python -m tools.adr`, not
  its `index/check/new/where/…` subcommands (v-next).
- **`governed-by` overlay** linking commands to ADRs (v-next; the authored layer).
- **Cataloguing shell scripts / one-off tools** beyond make targets and
  `python -m` entry points.
- **Blocking** on any finding — non-blocking throughout.
