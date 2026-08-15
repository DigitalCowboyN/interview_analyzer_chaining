# Lifecycle R1 — Forward loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make graph drift surface automatically — a fast changed-domain pre-commit locally, and a full `make health` CI sweep (advisory, except index freshness blocks).

**Architecture:** `tools/knowledge`'s `DOMAINS` registry gains per-domain surface paths (dataclass); a `changed_domains` resolver + CLI maps staged files → touched domains; pre-commit runs only those (+ graph), advisory; a CI workflow runs the full sweep and enforces index freshness via regenerate-then-diff.

**Tech Stack:** Python 3 (stdlib), pytest, GNU Make, GitHub Actions. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-15-lifecycle-r1-forward-loop-design.md`.

## Global Constraints

- **Pre-commit is always `exit 0`** — it informs, never blocks.
- **Only index freshness is enforced, only in CI** — via `make regen-all` + `git diff --exit-code`. All judgment findings stay advisory.
- **`DOMAINS` is the single source of truth** for domain surfaces; its only consumers are in `tools/knowledge` + `tests/knowledge`.
- **Surfaces may be slightly broad** — CI's full sweep is the guarantee; pre-commit is fast feedback. Do not over-narrow.
- **Names verbatim:** `Domain` (dataclass: `slug`, `make`, `surfaces`), `changed_domains`, make target `regen-all`, workflow `.github/workflows/health.yml`.

---

### Task 1: Surface registry (`Domain` dataclass) + `changed_domains` resolver + CLI

**Files:**
- Modify: `tools/knowledge/check.py` (`DOMAINS` → dataclass; `check_cascade_covers_domains`)
- Create: `tools/knowledge/surfaces.py`
- Modify: `tools/knowledge/__main__.py` (add `changed-domains` subcommand)
- Test: `tests/knowledge/test_check.py`, `tests/knowledge/test_surfaces.py`

**Interfaces:**
- Produces: `Domain` dataclass; `DOMAINS: list[Domain]`; `changed_domains(files, domains=DOMAINS) -> list[str]`; CLI `python -m tools.knowledge changed-domains` (stdin paths → make-names).

- [ ] **Step 1: Write the failing test** — `tests/knowledge/test_surfaces.py`:

```python
from tools.knowledge.surfaces import changed_domains


def test_changed_domains_maps_capability_edit():
    assert changed_domains(["docs/capabilities/x.md"]) == ["capability"]


def test_changed_domains_src_touches_code_family():
    got = set(changed_domains(["src/api/routers/foo.py"]))
    assert {"api", "code", "capability", "glossary", "prompt"} <= got
    assert "usecase" not in got and "testmap" not in got


def test_changed_domains_unmatched_path_is_empty():
    assert changed_domains(["README.md"]) == []


def test_changed_domains_dedupes_and_sorts():
    out = changed_domains(["tests/a.py", "tests/b.py"])
    assert out == ["testmap"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/knowledge/test_surfaces.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.knowledge.surfaces'`.

- [ ] **Step 3: Migrate `DOMAINS` to a dataclass** in `tools/knowledge/check.py`. Replace the `DOMAINS: List[Tuple[str, str]] = [...]` block with:

```python
from dataclasses import dataclass


@dataclass
class Domain:
    slug: str            # docs/<slug>/  (cascade row + graph addressing)
    make: str            # `make <make>-check` / `python -m tools.<make>`
    surfaces: list        # path prefixes whose change can cause this check to find drift


# Single source of truth for the knowledge-graph domains. `surfaces` drives the
# changed-domain pre-commit (tools.knowledge.surfaces). Add a row here (+ a docs/index.md
# row) when a new domain ships.
DOMAINS = [
    Domain("adr", "adr", ["docs/adr/", "src/"]),
    Domain("api", "api", ["src/api/", "frontend/openapi.json"]),
    Domain("cli", "cli", ["Makefile", "tools/"]),
    Domain("code", "code", ["src/", "tools/"]),
    Domain("capabilities", "capability", ["docs/capabilities/", "src/", "tools/"]),
    Domain("glossary", "glossary", ["src/", "docs/glossary/"]),
    Domain("graph", "graph", ["docs/"]),
    Domain("graph-queries", "graphq", ["src/projections/", "docs/graph-queries/"]),
    Domain("prompts", "prompt", ["src/", "docs/prompts/"]),
    Domain("use-cases", "usecase", ["docs/use-cases/"]),
    Domain("tests", "testmap", ["tests/"]),
]
```

Update `check_cascade_covers_domains`'s loop from `for slug, _make in domains:` to:

```python
    for d in domains:
        if f"{d.slug}/" not in text:
            findings.append(Finding(
                f"knowledge: cascade root docs/index.md has no row for '{d.slug}/'"))
```

(Leave `Tuple` import if still used elsewhere; remove it if now unused to keep flake8 clean.)

- [ ] **Step 4: Create `tools/knowledge/surfaces.py`:**

```python
from __future__ import annotations

from typing import Iterable, List

from tools.knowledge.check import DOMAINS


def changed_domains(files: Iterable[str], domains=DOMAINS) -> List[str]:
    """The `make`-names of domains whose surface any of `files` touches (sorted, deduped)."""
    hit = set()
    for f in files:
        f = f.replace("\\", "/")
        for d in domains:
            if any(f.startswith(p) for p in d.surfaces):
                hit.add(d.make)
    return sorted(hit)
```

- [ ] **Step 5: Add the CLI subcommand** in `tools/knowledge/__main__.py`:

Add import: `from tools.knowledge.surfaces import changed_domains`. Add the handler:

```python
def cmd_changed_domains(args) -> int:
    files = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
    for make in changed_domains(files):
        print(make)
    return 0
```

Register it: `sub.add_parser("changed-domains")` and add `"changed-domains": cmd_changed_domains` to the dispatch dict.

- [ ] **Step 6: Run tests to verify pass**

Run: `python -m pytest tests/knowledge -v`
Expected: PASS (new surfaces tests + the migrated cascade test still green).

- [ ] **Step 7: Smoke the CLI**

Run: `printf 'docs/use-cases/x.md\n' | python -m tools.knowledge changed-domains`
Expected: prints `usecase`.

- [ ] **Step 8: Commit**

```bash
git add tools/knowledge/check.py tools/knowledge/surfaces.py tools/knowledge/__main__.py tests/knowledge/
git commit -m "feat(knowledge): Domain surfaces registry + changed_domains resolver + CLI"
```

---

### Task 2: Changed-domain pre-commit

**Files:**
- Modify: `.githooks/pre-commit`

- [ ] **Step 1: Rewrite the hook** to run only touched domains' checks + graph, advisory:

```bash
#!/usr/bin/env bash
# Non-blocking drift report for the domains this commit touches (+ the cross-domain graph).
# Never fails the commit — visibility, not a gate.
files="$(git diff --cached --name-only)"
domains="$(printf '%s\n' "$files" | bash scripts/with-project-py.sh tools.knowledge changed-domains 2>/dev/null)"
for d in $domains graph; do
    bash scripts/with-project-py.sh "tools.$d" check || true
done
exit 0
```

- [ ] **Step 2: Smoke — touch a capability, confirm only capability + graph run, exit 0**

```bash
# stage a trivial capability touch and run the hook manually
git add docs/capabilities/README.md   # or any capability-surface file already staged
bash .githooks/pre-commit ; echo "hook exit=$?"
```
Expected: prints `capability-check: ...` and `graph-check: ...` only (not usecase/testmap/etc.), and `hook exit=0`. Unstage afterward if nothing real changed.

- [ ] **Step 3: Smoke — a docs/use-cases change runs only usecase + graph**

```bash
printf 'docs/use-cases/surface-the-signal.md\n' | bash scripts/with-project-py.sh tools.knowledge changed-domains
```
Expected: `usecase`.

- [ ] **Step 4: Commit**

```bash
git add .githooks/pre-commit
git commit -m "feat(hooks): changed-domain pre-commit — run only touched domains' checks (+ graph)"
```

---

### Task 3: CI health workflow + `regen-all`

**Files:**
- Modify: `Makefile` (add `regen-all`)
- Create: `.github/workflows/health.yml`
- Regenerate: `docs/cli/index.md` (new `regen-all` target catalogued)

- [ ] **Step 1: Add the `regen-all` target** to `Makefile` (near the other index targets):

```makefile
.PHONY: regen-all
regen-all: ## Regenerate every generated index/doc (used by the CI index-freshness gate)
	@$(MAKE) code-index capability-index usecase-index testmap-index glossary-index \
	         api-index graphq-index prompt-index adr-index cli-index graph-index
```

(Order: code first, graph last — graph aggregates every domain's nodes.)

- [ ] **Step 2: Create `.github/workflows/health.yml`:**

```yaml
name: knowledge-graph health
on:
  push:
  pull_request:

jobs:
  health:
    runs-on: ubuntu-latest
    env:
      # the checks import the app with placeholder keys; set them so imports never prompt
      ANTHROPIC_API_KEY: ci-placeholder
      OPENAI_API_KEY: ci-placeholder
      PYTHON: python
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10.7"
      - name: Install deps
        run: pip install -r requirements.txt
      - name: Drift report (advisory — never fails)
        run: make health
      - name: Index freshness gate (blocks on stale generated files)
        run: |
          make regen-all
          if ! git diff --exit-code; then
            echo "::error::committed generated indexes are stale — run 'make regen-all' and commit the result"
            git diff --stat
            exit 1
          fi
```

- [ ] **Step 3: Verify locally** (CI itself runs on push):

```bash
make regen-all          # regenerates every index
git diff --exit-code docs/ && echo "indexes fresh (gate would pass)" || echo "gate would fail — regen + commit"
make cli-index          # catalog the new regen-all target
make cli-check          # clean — regen-all catalogued
python -c "import yaml; yaml.safe_load(open('.github/workflows/health.yml')); print('workflow YAML valid')"
```

Expected: `regen-all` runs all indexes; the freshness diff is clean on a fresh tree; `cli-check` clean; YAML valid.

- [ ] **Step 4: Commit**

```bash
git add Makefile .github/workflows/health.yml docs/cli/index.md
git commit -m "feat(ci): health workflow (advisory sweep + index-freshness gate) + regen-all"
```

---

## After all tasks

Capture **ADR-0023** (`python -m tools.adr new "Forward loop — advisory by default, index freshness enforced in CI"`, `source:` = the spec, note it refines ADR-0016; then `make adr-index`). Regenerate `make graph-index` (ADR node count changed) and confirm `graph-check` clean. Push the branch to trigger the new CI workflow and confirm it runs (advisory sweep green; freshness gate green on a fresh tree). Run the final whole-branch review on the most capable model, then use **superpowers:finishing-a-development-branch**.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-15.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| knowledge | yes | `DOMAINS` → dataclass + `surfaces`; new `surfaces.py` + `changed-domains` CLI | single source of truth |
| cli | yes | new `regen-all` target → `cli-index`; `cli-check` clean | — |
| adr | yes | ADR-0023 (refines 0016; after tasks) | — |
| all domain checks | yes (read-only) | invoked by the new pre-commit/CI; logic unchanged | plumbing only |
| graph / code / capabilities / use-cases / tests / glossary / api / prompts / graph-queries | no (logic) | — | unaffected internals |

**Verdict:** reconciled — knowledge (registry + resolver) subject; cli/adr reconciled; domain checks invoked, not modified.
