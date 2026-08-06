# Tests Domain — design

**Status:** approved by owner 2026-08-06 (brainstorm dialogue).
**Program:** Round 3b of the guarded-knowledge-graph vertical stack. Adds the **tests
domain** — the layer that closes the Requirements Traceability Matrix. With use-cases
(3a) the graph traces intent → capability → code; this round adds
**test → code / intent**, so coverage gains a second, orthogonal dimension:
*is it verified?*, distinct from *is it implemented?*. Builds directly on ADR-0021
(use-cases) and ADR-0020 (graph-links); activates the graph's reserved `Test` node and
`verifies` edge.

## Framing (locked in brainstorm + research)

Grounded in the test-taxonomy and RTM literature reviewed 2026-08-06 (the modern test
pyramid — unit/integration/e2e + acceptance/contract; the RTM's separation of *planned /
implemented / verified* trace states):

- **Tests are code artifacts, not authored intent.** With ~1,597 test functions, `Test`
  nodes are **derived** from the suite (like the code map's `depends_on`), never authored
  markdown. Authoring Test nodes would drift instantly. → **derived, per test file.**
- **A test's target splits by how it connects.** Unit tests (colocated in `tests/<pkg>/`)
  verify a specific **code unit** — derivable by the tests-mirror-source convention.
  Integration/e2e tests (`tests/integration/`) span the pipeline and verify a **use-case's
  acceptance criteria or a capability** — a semantic link only a human knows, so
  **authored** via a marker. Derive the ~1,590 bulk; author the handful that matter.
- **Verification is a distinct axis from implementation.** The RTM literature separates
  "implemented" from "verified." So we keep implementation coverage untouched and add an
  **orthogonal verification axis** — a use-case can honestly read *FULLY_COVERED +
  UNVERIFIED* (built but unproven). This **revises** ADR-0021's forward-looking note that
  3b would "refine the FULLY_COVERED predicate"; the two-axis model is cleaner and doesn't
  silently reclassify the existing corpus.

**Extensibility (owner):** `test_type` is an **open ordered set** (`unit | integration |
e2e`, reserving `acceptance | contract`), mirroring `form`/`category`. The `verifies` edge
target is **polymorphic** (CodeUnit, UseCase, or Capability) — a generalization of the
edge model, not a special case.

## Naming

- Tool package **`tools/testmap/`** (not `tools/tests/` — avoids confusion with the real
  `tests/` suite; parallels the `map-the-*` self-registration naming).
- Docs **`docs/tests/`**; cascade/domain slug **`tests`**; graph node type **`Test`**,
  addressing **`tests:<slug>`** where slug = the test file's path under `tests/` with `/`→`.`
  and `.py` dropped (`tests/capability/test_check.py` → `tests:capability.test_check`).
- Make targets **`testmap-index` / `testmap-check`**; `DOMAINS` entry `("tests", "testmap")`;
  health-loop token `testmap`.

## The node — `Test`, derived per file

`tools/testmap/reader.py`:

```python
@dataclass
class Test:
    slug: str            # "capability.test_check"  (path under tests/, /→., no .py)
    path: str            # "tests/capability/test_check.py"
    test_type: str       # unit | integration | e2e  (derived)
    target: str          # derived code unit slug ("" if unresolved)
    verifies: List[str]  # authored "<domain>:<id>" markers (may be empty)
    n_tests: int         # count of `def test_` in the file (metadata, not nodes)
```

`load_tests(root, tests_dir="tests")` walks `tests/**/test_*.py`, skipping
`__pycache__`, `conftest.py`, `__init__.py`, and `fixtures/`.

- **`test_type` (derived, open `TEST_TYPES`):** under `tests/integration/` → `e2e` if the
  filename matches `test_e2e_*` / `test_end_to_end_*` / `*_smoke` else `integration`;
  otherwise `unit`.
- **`target` (derived — the code convention):** take the first path segment under `tests/`
  (`seg`); resolve against the code-unit registry (`tools.code.reader.real_code_units`):
  `seg` if present, elif `tools.{seg}` present → `tools.{seg}`, else `""` (unresolved —
  e.g. `api_surface`, `pipeline`, `services`, `integration`, root-level files). Reuses the
  code map as the single source of unit truth.
- **`verifies` (authored markers):** module-level lines `# verifies: <domain>:<id>`
  (grep-able, consistent with the `graphq:` docstring markers). Multiple lines allowed.
  The `<domain>:<id>` carries its own domain, so the target is prefix-resolved.

## The `verifies` edge — derived, polymorphic target

One registry entry activates the reserved edge. Because its endpoints are already fully
`<domain>:<id>`-addressed, it is modeled as a **derived** edge whose handler lives in the
tests domain — this sidesteps the registry's single-`to_type` assumption and lets the
**existing** `check_endpoints` (which resolves each endpoint's domain by prefix) validate
dangling markers generically.

- `tools/testmap/reader.py` exposes
  `verifies_edges(root) -> list[tuple[str, str, str]]` = `(src_addr, dst_addr, test_type)`:
  - **derived → code:** for each test with a resolved `target`, `("tests:<slug>",
    "code:<target>", test_type)`.
  - **authored → intent:** for each `# verifies: <domain>:<id>` marker,
    `("tests:<slug>", "<domain>:<id>", test_type)`.
- `tools/graph/registry.py`: add `NODE_DOMAINS["Test"] = "tests"`, and one `EDGES` entry:

```python
EdgeType("verifies", "verified_by", "Test", "CodeUnit|UseCase|Capability", "derived",
         field="verifies_edges", resolve="id",
         properties=[PropSpec("test_type", enum=["unit", "integration", "e2e"])],
         description="A test proves a code unit works, or an acceptance test proves an intent.")
```

  `to_type` is a display string; the derived handler emits fully-addressed endpoints, so no
  polymorphic resolver is needed. `tools/graph/reader.py`: add the `Test` adapter
  (`load_tests`, `slug`) to `_ADAPTERS`, and a `_DERIVED["verifies_edges"]` handler that
  wraps `tools.testmap.reader.verifies_edges`. `graph-check` then flags any authored marker
  pointing at a nonexistent node — the cross-domain integrity we want, reusing existing code.

## The verification axis — derived, orthogonal, transitive

`tools/testmap/verification.py`, pure functions over the derived edges + the code /
capability / use-case readers. **Does not modify the use-cases or capabilities domains** —
the verification view lives in `docs/tests/`.

- **CodeUnit:** `verified` iff ≥1 `verifies → code:<unit>` edge exists.
- **Capability** (`verified_capability`): `VERIFIED` iff `implemented_by` non-empty and all
  its units are verified; `PARTIALLY_VERIFIED` if some; `UNVERIFIED` if none/empty.
- **UseCase** (`verified_use_case`): `VERIFIED` iff **directly** verified (an authored
  `verifies` marker targets this use-case *or* a capability that fulfills it) **or** all its
  fulfilling capabilities are `VERIFIED`; `PARTIALLY_VERIFIED` if some direct/transitive
  verification; `UNVERIFIED` if none. States: `UNVERIFIED | PARTIALLY_VERIFIED | VERIFIED`.

This is the same transitivity shape as implementation coverage, so the two axes read
consistently. A use-case is now describable on both axes independently
(e.g. `FULLY_COVERED` + `UNVERIFIED`).

## Generated artifact — `docs/tests/index.md`

`render_index(tests, verification)` (pure, deterministic, single trailing newline):

- **Suite catalog** grouped by `test_type` (open-set order), each test: slug, target (or
  `—`), authored `verifies` targets, `n_tests`.
- **Verification rollup** — the RTM verification view: each use-case and capability with its
  derived verification state (the payoff — where implemented-but-unproven shows up).

## The guard — `make testmap-check` (non-blocking, exit 0)

All advisory, `return 0`:

- **test_type-in-set** — `test_type` outside `TEST_TYPES` → finding (shouldn't happen from
  derivation; guards a future authored override).
- **unmapped-test** — a test file with no resolved `target` *and* no authored `verifies`
  marker → advisory ("test verifies nothing the graph can see" — e.g. an orphan or a
  cross-cutting test that should carry a marker).
- **unverified-intent** — a use-case that is `UNVERIFIED` → advisory (the honest gap: an
  implemented intent nothing proves). Expected and desired for the aspirational corpus.
- **index-sync** — committed `docs/tests/index.md` matches a fresh render.

Cross-domain endpoint integrity (an authored `# verifies:` marker pointing at a nonexistent
node) is covered by **`graph-check`** — deliberately not duplicated.

## CLI

`python -m tools.testmap {index | check | verification}`. `verification` prints each
capability + use-case with its derived verification state (the RTM verification view in the
terminal). `neighbors tests:<slug>` in the graph CLI already answers "what does this test
verify" once the node type is registered.

## Tooling, wiring, self-registration (the established pattern)

- `tools/testmap/` — `reader.py` (`Test`, `TEST_TYPES`, `load_tests`, `verifies_edges`),
  `verification.py`, `render.py`, `check.py`, `__main__.py`. Non-blocking guard.
- Graph: `Test` node + `verifies` edge activated (one registry entry + one adapter + one
  derived handler).
- Cascade + registry: `("tests", "testmap")` in `tools/knowledge`'s `DOMAINS`; a
  `docs/index.md` row containing `tests/`; the `Test` entry in the graph registry.
- Makefile: `testmap-index`, `testmap-check`; add `testmap` to the `health` loop.
- Self-registration: `docs/code/tools.testmap.md` (CodeUnit, role tooling) + additive
  `docs/capabilities/map-the-tests.md` (child of `maintain-a-guarded-knowledge-graph`,
  `implemented_by: [tools.testmap]`) — mirrors `map-the-code` / `link-the-domains`.
- **Seed authored markers:** add `# verifies:` markers to the few real e2e/integration
  tests that clearly validate a use-case (e.g. `tests/integration/test_e2e_user_edits.py`
  → the correction/edit use-cases), proving the authored path end-to-end and lighting up at
  least one use-case to `VERIFIED`. Editing test files to add a comment is allowed (they are
  this domain's own surface); no production code changes.

## Testing

- **Unit** — `load_tests` derives `test_type` (unit/integration/e2e by path+filename),
  resolves `target` against the code registry (bare vs `tools.` prefix; unresolved → ""),
  parses `# verifies:` markers, counts `def test_`; `verifies_edges` emits derived-code and
  authored-intent tuples with `test_type`; `verified_capability` / `verified_use_case` yield
  the three states over a synthetic code+capability+use-case+test fixture (direct and
  transitive); each `check_*` flags its case and passes clean; `render_index` groups by
  test_type and shows the rollup; **assert no check raises**. Graph: `harvest` includes
  `verifies` edges with `test_type` props; `nodes` includes the `Test` set; a dangling
  authored marker is flagged by `check_endpoints`; a reserved-target `verifies` (e.g. a
  marker to a capability) resolves.
- **Smoke** — `make testmap-index` writes `docs/tests/index.md` over the real suite;
  `make testmap-check` clean-or-advisory; `make graph-index` + `make graph-check` clean with
  `verifies` live and counted; `make knowledge-check` + `make cli-check` clean; `make health`
  runs the tests check; at least one use-case reads `VERIFIED` via a seeded marker.

## Non-goals (this round)

- **Running tests or measuring line coverage.** Verification here means *a test verifies
  this node in the graph*, derived from structure + markers — not pytest execution or
  coverage %. (A future round could ingest pass/fail or coverage data as edge properties.)
- **Modifying the use-cases or capabilities domains' generated files** — the verification
  view lives in `docs/tests/`; their indexes stay implementation-only.
- **Per-test-function nodes** — files are the node grain; function counts are metadata.
- **`acceptance` / `contract` test types** — reserved in the open set, not derived yet.
- **Blocking** on any finding.

## Capture as ADR

Capture **ADR-0022**: adopt a tests domain that derives `Test` nodes per file and a
polymorphic `verifies` edge (derived→code by convention, authored→intent by marker), adding
an **orthogonal verification axis** (`UNVERIFIED/PARTIALLY_VERIFIED/VERIFIED`) alongside
implementation coverage. `source:` = this spec. **Refines ADR-0021** (correcting its
"refine FULLY_COVERED predicate" note to the two-axis model) and ADR-0020 (a new derived
edge + node type); supersedes nothing.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-06.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| tests | yes | the new domain (subject) | — |
| graph | yes | activate reserved `Test` node + `verifies` edge (derived handler, prefix-resolved endpoints) | one registry entry + one adapter + one derived handler |
| code | yes (read-only) | `verifies → code` resolves against the code-unit registry; verification axis reads `implemented_by` → units | reuse `tools.code.reader` |
| capabilities | yes | read-only for the axis; **one additive** `map-the-tests` self-registration child | no existing capability edited |
| use-cases | yes (read-only) | verification rolls up through `fulfilled_by`; use-case files unchanged | authored markers may reference `use-cases:<id>` |
| cli | yes | `testmap-*` + `health` → `cli-index` | — |
| knowledge | yes | cascade row + `DOMAINS` entry for `tests` | — |
| adr | yes | ADR-0022 (refines 0021, 0020) | — |
| glossary / api / prompts / graph-queries | no | — | unaffected |

**Verdict:** reconciled — tests/graph (subject + activation) and code/cli/knowledge
(convention + wiring) reconciled here; capabilities/use-cases consulted read-only with a
single additive self-registration node.
