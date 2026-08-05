# Knowledge Cascade + Spec/Plan Honesty Check — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the seven guarded knowledge domains discoverable when needed (a cascade root + lightweight pointer) and add a non-blocking spec/plan honesty check that records, per artifact, which domains were consulted.

**Architecture:** Authored `docs/index.md` cascade root + a 3-line `CLAUDE.md` pointer do the discovery (structure over injection). A `PostToolUse(Write)` hook nudges a per-domain review recorded as a `## Knowledge-graph check` addendum in the spec/plan. A thin `tools/knowledge/` guard mechanizes the only checkable invariants (addendum present on new specs/plans; cascade root covers every domain). The ADR "before" hook is slimmed to a provisional pointer.

**Tech Stack:** Python 3 (stdlib only — no yaml/AST needed here), pytest, Make, bash hook scripts, `.claude/settings.json`.

## Global Constraints

- **Non-blocking, always:** every check returns `list[Finding]`; no check raises; every CLI command `return 0`. `make knowledge-check` must never exit non-zero.
- **Interpreter:** run Python as `~/.pyenv/shims/python`.
- **Run tests:** `~/.pyenv/shims/python -m pytest <path> -p no:cacheprovider -q -o addopts=""` (pyenv python; coverage gate off).
- **Idiom:** match the established domain tools — `Finding` is a message-only dataclass; CLI prints `<domain>-check: N warning(s):` / `<domain>-check: clean`; mirror `tools/adr/__main__.py`'s stdin-JSON hook handling.
- **Domain registry** (`DOMAINS` in `tools/knowledge/check.py`) is the single source of truth for the 7 domains — the cascade-coverage check reads it; nothing hardcodes the list twice.
- **Adoption date:** `2026-08-05`. Specs/plans whose filename date is `>=` this must carry the addendum; earlier ones are grandfathered (they predate the process — no false back-stamping).
- DRY, YAGNI, TDD, frequent commits.

---

### Task 1: `tools/knowledge/check.py` — the guard

**Files:**
- Create: `tools/knowledge/__init__.py` (empty)
- Create: `tools/knowledge/check.py`
- Test: `tests/knowledge/test_check.py`

**Interfaces:**
- Produces: `DOMAINS: list[tuple[str,str]]` (slug, make-name); `ADOPTION_DATE: str`; `ADDENDUM_HEADING: str`; `@dataclass Finding(message)`; `check_addendum_present(specs_dir, plans_dir, adoption_date=ADOPTION_DATE) -> list[Finding]`; `check_cascade_covers_domains(root=".", domains=DOMAINS) -> list[Finding]`; `run_all(root=".") -> list[Finding]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/knowledge/test_check.py
import os
from tools.knowledge.check import (
    Finding, DOMAINS, ADDENDUM_HEADING,
    check_addendum_present, check_cascade_covers_domains, run_all,
)


def _write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


def test_addendum_missing_on_new_spec_is_flagged(tmp_path):
    specs = tmp_path / "specs"
    _write(str(specs / "2026-08-05-new-thing-design.md"), "# New thing\nno addendum here\n")
    msgs = " ".join(f.message for f in check_addendum_present(str(specs), str(tmp_path / "plans")))
    assert "new-thing" in msgs


def test_addendum_present_on_new_spec_is_clean(tmp_path):
    specs = tmp_path / "specs"
    _write(str(specs / "2026-08-05-new-thing-design.md"), f"# New thing\n{ADDENDUM_HEADING}\nreviewed\n")
    assert check_addendum_present(str(specs), str(tmp_path / "plans")) == []


def test_pre_adoption_spec_is_grandfathered(tmp_path):
    specs = tmp_path / "specs"
    _write(str(specs / "2026-07-04-old-design.md"), "# Old\nno addendum, but predates the process\n")
    assert check_addendum_present(str(specs), str(tmp_path / "plans")) == []


def test_cascade_root_missing_domain_is_flagged(tmp_path):
    docs = tmp_path / "docs"
    _write(str(docs / "index.md"), "# Knowledge map\n[adr/](adr/index.md)\n")  # only adr
    msgs = " ".join(f.message for f in check_cascade_covers_domains(str(tmp_path)))
    assert "glossary" in msgs and "code" in msgs


def test_cascade_root_absent_is_one_finding(tmp_path):
    findings = check_cascade_covers_domains(str(tmp_path))  # no docs/index.md
    assert len(findings) == 1 and "cascade root" in findings[0].message


def test_run_all_returns_list_never_raises(tmp_path):
    assert isinstance(run_all(str(tmp_path)), list)
```

- [ ] **Step 2: Run to verify they fail**

Run: `~/.pyenv/shims/python -m pytest tests/knowledge/test_check.py -p no:cacheprovider -q -o addopts=""`
Expected: FAIL with `ModuleNotFoundError: tools.knowledge`.

- [ ] **Step 3: Implement**

```python
# tools/knowledge/check.py
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

# Single source of truth for the knowledge-graph domains: (docs slug, make-name).
# Each has docs/<slug>/index.md and a `make <make-name>-check`. Add a row here (and
# to docs/index.md) when a new domain ships (e.g. capabilities).
DOMAINS: List[Tuple[str, str]] = [
    ("adr", "adr"),
    ("api", "api"),
    ("cli", "cli"),
    ("code", "code"),
    ("glossary", "glossary"),
    ("graph-queries", "graphq"),
    ("prompts", "prompt"),
]

ADOPTION_DATE = "2026-08-05"          # specs/plans dated >= this must carry the addendum
ADDENDUM_HEADING = "## Knowledge-graph check"
_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


@dataclass
class Finding:
    message: str


def _leading_date(path: str) -> str:
    m = _DATE.match(os.path.basename(path))
    return m.group(1) if m else ""  # "" (no date prefix) => not grandfathered


def check_addendum_present(specs_dir: str, plans_dir: str,
                           adoption_date: str = ADOPTION_DATE) -> List[Finding]:
    findings: List[Finding] = []
    for directory, kind in ((specs_dir, "spec"), (plans_dir, "plan")):
        for path in sorted(glob.glob(os.path.join(directory, "*.md"))):
            date = _leading_date(path)
            if date and date < adoption_date:
                continue  # grandfathered — predates the honesty-check process
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if ADDENDUM_HEADING not in text:
                findings.append(Finding(
                    f"knowledge: {kind} {os.path.basename(path)} has no "
                    f"'{ADDENDUM_HEADING}' addendum — was the knowledge-graph check run?"))
    return findings


def check_cascade_covers_domains(root: str = ".", domains=DOMAINS) -> List[Finding]:
    index_path = os.path.join(root, "docs", "index.md")
    try:
        text = open(index_path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return [Finding("knowledge: cascade root docs/index.md is missing — author it")]
    findings: List[Finding] = []
    for slug, _make in domains:
        if f"{slug}/" not in text:
            findings.append(Finding(
                f"knowledge: cascade root docs/index.md has no row for '{slug}/'"))
    return findings


def run_all(root: str = ".") -> List[Finding]:
    specs = os.path.join(root, "docs/superpowers/specs")
    plans = os.path.join(root, "docs/superpowers/plans")
    findings: List[Finding] = []
    findings += check_cascade_covers_domains(root)
    findings += check_addendum_present(specs, plans)
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/shims/python -m pytest tests/knowledge/test_check.py -p no:cacheprovider -q -o addopts=""`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/knowledge/__init__.py tools/knowledge/check.py tests/knowledge/test_check.py
git commit -m "feat(knowledge): guard — spec/plan addendum presence + cascade coverage"
```

---

### Task 2: CLI (`check` + `nudge`) + interpreter script + Makefile target

**Files:**
- Create: `tools/knowledge/__main__.py`
- Create: `scripts/with-project-py.sh` (generalized yaml-resolving interpreter, takes a module arg)
- Modify: `Makefile` (add `knowledge-check` near `code-check`)
- Test: `tests/knowledge/test_cli.py`

**Interfaces:**
- Consumes: `tools.knowledge.check.run_all`.
- Produces: `python -m tools.knowledge {check|nudge}`; `scripts/with-project-py.sh <module> [args...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/knowledge/test_cli.py
import json
import subprocess
import sys


def _run(args, stdin=""):
    return subprocess.run([sys.executable, "-m", "tools.knowledge", *args],
                          input=stdin, capture_output=True, text=True)


def test_check_exits_zero():
    proc = _run(["check"])
    assert proc.returncode == 0, proc.stderr
    assert "knowledge-check" in proc.stdout


def test_nudge_fires_on_spec_path():
    proc = _run(["nudge"], stdin=json.dumps(
        {"tool_input": {"file_path": "docs/superpowers/specs/2026-08-05-x-design.md"}}))
    assert proc.returncode == 0
    assert "docs/index.md" in proc.stdout


def test_nudge_fires_on_plan_path():
    proc = _run(["nudge"], stdin=json.dumps(
        {"tool_input": {"file_path": "docs/superpowers/plans/2026-08-05-x.md"}}))
    assert "Knowledge-graph check" in proc.stdout


def test_nudge_silent_on_other_path():
    proc = _run(["nudge"], stdin=json.dumps({"tool_input": {"file_path": "src/api/main.py"}}))
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
```

- [ ] **Step 2: Run to verify fail** — `No module named tools.knowledge.__main__`.

- [ ] **Step 3: Implement**

```python
# tools/knowledge/__main__.py
from __future__ import annotations

import argparse
import json
import sys

from tools.knowledge.check import run_all

_SPEC_PLAN_DIRS = ("docs/superpowers/specs/", "docs/superpowers/plans/")


def _read_stdin_json() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except Exception:
        return {}


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"knowledge-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("knowledge-check: clean")
    return 0  # NON-BLOCKING


def cmd_nudge(args) -> int:
    # PostToolUse(Write) hook: honesty-check reminder when a spec/plan lands.
    path = _read_stdin_json().get("tool_input", {}).get("file_path", "").replace("\\", "/")
    if any(d in path for d in _SPEC_PLAN_DIRS):
        print("This spec/plan likely touches the knowledge graph. Review it against "
              "docs/index.md: for each domain it affects, consult the bundle and run "
              "its `make <domain>-check`; record a '## Knowledge-graph check' addendum "
              "(per-domain touched/consulted + verdict). If it locks architectural "
              "decisions, also capture ADR(s).")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.knowledge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("nudge")
    args = parser.parse_args(argv)
    return {"check": cmd_check, "nudge": cmd_nudge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

```bash
# scripts/with-project-py.sh
#!/usr/bin/env bash
# Resolve a Python interpreter that has the project deps (yaml), then run a module.
# Usage: with-project-py.sh <module> [args...]
# Non-blocking: if none is found, exit 0 silently.
set -u
mod="${1:-}"
[ -n "$mod" ] || exit 0
shift || true
_pyenv_python="$(command -v pyenv >/dev/null 2>&1 && pyenv which python 2>/dev/null || true)"
for py in python python3 "$HOME/.pyenv/shims/python" "$_pyenv_python"; do
  [ -n "$py" ] || continue
  if "$py" -c "import yaml" >/dev/null 2>&1; then
    exec "$py" -m "$mod" "$@"
  fi
done
exit 0
```

Make `scripts/with-project-py.sh` executable: `chmod +x scripts/with-project-py.sh`.

Add to `Makefile` immediately after the `code-check` target (around line 79):

```makefile
.PHONY: knowledge-check
knowledge-check: ## Reconcile specs/plans + cascade root vs the knowledge domains (non-blocking)
	@$(PYTHON) -m tools.knowledge check
```

- [ ] **Step 4: Run test + smoke**

Run: `~/.pyenv/shims/python -m pytest tests/knowledge/test_cli.py -p no:cacheprovider -q -o addopts=""` → PASS (4 passed).
Smoke: `~/.pyenv/shims/python -m tools.knowledge check` → exit 0. Expect findings until Tasks 4/6 (no `docs/index.md`; this spec + plan lack addenda). Confirm NO exception.
Smoke: `bash scripts/with-project-py.sh tools.knowledge check` → same output (interpreter resolves).

- [ ] **Step 5: Commit**

```bash
git add tools/knowledge/__main__.py scripts/with-project-py.sh Makefile tests/knowledge/test_cli.py
git commit -m "feat(knowledge): CLI (check/nudge) + shared interpreter script + make target"
```

---

### Task 3: Slim the ADR "before" hook to a provisional pointer

**Files:**
- Modify: `tools/adr/__main__.py` (`cmd_context`)
- Test: `tests/adr/test_context.py` (new)

**Interfaces:**
- Consumes: `tools.adr.intent.is_architectural` (unchanged — the keyword gate stays).
- The `cmd_context` behavior changes from injecting the full ADR index table to emitting a one-line pointer.

- [ ] **Step 1: Write the failing test**

```python
# tests/adr/test_context.py
import json
import subprocess
import sys


def _context(prompt):
    return subprocess.run([sys.executable, "-m", "tools.adr", "context"],
                          input=json.dumps({"prompt": prompt}),
                          capture_output=True, text=True)


def test_architectural_prompt_gets_pointer_not_table():
    out = _context("should we change this architecture decision?").stdout
    assert "docs/adr/index.md" in out
    assert "| 0001 |" not in out  # the full table is gone


def test_non_architectural_prompt_is_silent():
    out = _context("fix this typo in the readme").stdout
    assert out.strip() == ""
```

- [ ] **Step 2: Run to verify fail** — current `cmd_context` prints the table, so `test_architectural_prompt_gets_pointer_not_table` fails on the `| 0001 |` assertion.

- [ ] **Step 3: Implement** — replace `cmd_context` in `tools/adr/__main__.py` with:

```python
def cmd_context(args) -> int:
    # UserPromptSubmit hook: a PROVISIONAL, lean before-signal. Emits a pointer (not
    # the full ADR table) only on architectural prompts. Retire once the cascade
    # reliably gets ADRs consulted without it (see the 2026-08-05 knowledge-cascade
    # spec's retirement criterion).
    prompt = _read_stdin_json().get("prompt", "")
    if is_architectural(prompt):
        print("Locking an architectural decision? Consult docs/adr/index.md before "
              "you do (and docs/index.md for the wider knowledge map). "
              "Supersede rather than silently override.")
    return 0
```

(No file read, no row count — the number would go stale; the pointer stands alone.)

- [ ] **Step 4: Run tests** — `~/.pyenv/shims/python -m pytest tests/adr/test_context.py -p no:cacheprovider -q -o addopts=""` → PASS. Also run the full `tests/adr/` to confirm nothing else asserted the table: `~/.pyenv/shims/python -m pytest tests/adr/ -p no:cacheprovider -q -o addopts=""` (test_hooks_wiring still green — it is updated in Task 5).

- [ ] **Step 5: Commit**

```bash
git add tools/adr/__main__.py tests/adr/test_context.py
git commit -m "refactor(adr): slim before-hook to a provisional pointer (not the full table)"
```

---

### Task 4: Author the cascade root + `CLAUDE.md` pointer

**Files:**
- Create: `docs/index.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Author `docs/index.md`** — one row per domain (link + one-line ranking-signal description + reconcile command). Every domain slug from `DOMAINS` must appear as `<slug>/` so `check_cascade_covers_domains` passes.

```markdown
# Knowledge map

Guarded knowledge domains over this codebase. Land here, then follow the one you're
working in — read its `index.md`, and run its `make <domain>-check` when you change a
surface it covers. All checks are non-blocking (visibility, not gates).

| domain | what it holds | reconcile with |
| --- | --- | --- |
| [adr/](adr/index.md) | architectural decisions (what & why) — consult before locking one | `make adr-check` |
| [glossary/](glossary/index.md) | canonical vocabulary (nodes, lenses, dimensions, graph labels) pinned to code enums | `make glossary-check` |
| [code/](code/index.md) | package/module map: roles, derived deps + I/O, Mermaid pipeline | `make code-check` |
| [api/](api/index.md) | HTTP surface vs. committed `openapi.json` | `make api-check` |
| [cli/](cli/index.md) | command surface (CLI + make targets) | `make cli-check` |
| [prompts/](prompts/index.md) | probabilistic components — the LLM prompts the agents use | `make prompt-check` |
| [graph-queries/](graph-queries/index.md) | Neo4j read-query registry (schema + output contract) | `make graphq-check` |

**Writing a spec or plan?** Record a `## Knowledge-graph check` addendum — the
per-domain review of what it touched and what you reconciled (`make knowledge-check`
flags a new one that skipped it). Verdict is one of: **clean** (every touched domain
consulted) · **reconciled** (gaps found and fixed) · **overridden** (a design-affecting
gap the owner accepted, rationale recorded). If the check surfaces a gap a domain
should have caught, don't silently pass: fix mechanical gaps directly; for
design-affecting ones, loop back (change the design) or record an owner override.

Other docs (not knowledge domains): `architecture/` (system overview, data flow),
`product/`, `superpowers/{specs,plans}/` (design specs + implementation plans).
```

- [ ] **Step 2: Add the pointer to `CLAUDE.md`** — insert a `## Knowledge map` section immediately after the `## Layout` section:

```markdown
## Knowledge map
This repo keeps guarded knowledge domains under `docs/` — see
[`docs/index.md`](docs/index.md) for the map. Each has a non-blocking
`make <domain>-check`. When you change a surface one covers, consult its bundle and
run its check. When you write a spec/plan, record a `## Knowledge-graph check`
addendum (`make knowledge-check` flags a new one that skipped it).
```

- [ ] **Step 3: Verify cascade coverage** — `~/.pyenv/shims/python -c "from tools.knowledge.check import check_cascade_covers_domains as c; print(c('.'))"` → `[]` (empty). The addendum check will still flag this spec + plan until Task 6 — expected.

- [ ] **Step 4: Commit**

```bash
git add docs/index.md CLAUDE.md
git commit -m "docs(knowledge): author cascade root docs/index.md + CLAUDE.md pointer"
```

---

### Task 5: Rewire the hooks

**Files:**
- Modify: `.claude/settings.json`
- Delete: `scripts/with-adr-py.sh` (replaced by `with-project-py.sh`)
- Modify: `tests/adr/test_hooks_wiring.py`

- [ ] **Step 1: Update the wiring test first** (TDD — assert the target state). Open `tests/adr/test_hooks_wiring.py` and replace the two assertions that reference `with-adr-py.sh`:

```python
    # read side: slimmed ADR pointer, still via the shared interpreter resolver
    assert "with-project-py.sh tools.adr context" in ups
    # capture side: knowledge-graph honesty-check nudge on spec/plan writes
    assert "with-project-py.sh tools.knowledge nudge" in ptu
```

(Keep the rest of the test — it reads `.claude/settings.json` and locates the `UserPromptSubmit` / `PostToolUse` command strings.)

- [ ] **Step 2: Run to verify fail** — `~/.pyenv/shims/python -m pytest tests/adr/test_hooks_wiring.py -p no:cacheprovider -q -o addopts=""` → FAIL (settings still point at `with-adr-py.sh`).

- [ ] **Step 3: Rewire `.claude/settings.json`**

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "bash scripts/with-project-py.sh tools.adr context" } ] }
    ],
    "PostToolUse": [
      { "matcher": "Write", "hooks": [ { "type": "command", "command": "bash scripts/with-project-py.sh tools.knowledge nudge" } ] }
    ]
  }
}
```

- [ ] **Step 4: Remove the superseded script + confirm no dangling references**

```bash
git rm scripts/with-adr-py.sh
grep -rn "with-adr-py" . --include='*.py' --include='*.json' --include='*.md' --include='Makefile' --include='*.sh' || echo "no dangling references"
```

If `grep` reports a hit outside this plan/spec doc, resolve it (repoint to `with-project-py.sh`). (If the shell `grep` alias misbehaves, use `~/.pyenv/shims/python` to scan instead.)

- [ ] **Step 5: Run tests + live hook smoke**

```bash
~/.pyenv/shims/python -m pytest tests/adr/ tests/knowledge/ -p no:cacheprovider -q -o addopts=""   # green
echo '{"prompt":"should we refactor this architecture?"}' | bash scripts/with-project-py.sh tools.adr context      # prints pointer
echo '{"tool_input":{"file_path":"docs/superpowers/specs/x-design.md"}}' | bash scripts/with-project-py.sh tools.knowledge nudge   # prints honesty-check reminder
```

- [ ] **Step 6: Commit**

```bash
git add .claude/settings.json tests/adr/test_hooks_wiring.py
git rm scripts/with-adr-py.sh
git commit -m "chore(hooks): rewire to shared interpreter; ADR pointer + knowledge nudge"
```

---

### Task 6: Dogfood — add the addendum to this spec + plan; reconcile clean

**Files:**
- Modify: `docs/superpowers/specs/2026-08-05-knowledge-cascade-and-honesty-check-design.md`
- Modify: `docs/superpowers/plans/2026-08-05-knowledge-cascade-and-honesty-check.md`
- Modify (regenerate): `docs/cli/index.md` (the new `knowledge-check` make target enters the CLI catalog)

- [ ] **Step 1: Append the `## Knowledge-graph check` addendum to BOTH this spec and this plan.** Use this content (identical review for both — they are the same feature):

```markdown
## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-05.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| adr | yes | captured as ADR-0016 (`source:` = this spec) | extends ADR-0015's disclosure model; no supersede |
| cli | yes | added `make knowledge-check` → ran `make cli-index`; `cli-check` clean | new make target enters the CLI catalog |
| code | no | — | `tools/knowledge/` lives under `tools/`, not a `src/` package |
| glossary | no | — | no new domain vocabulary pinned to code enums |
| api | no | — | no HTTP surface change |
| prompts | no | — | no LLM prompt added or changed |
| graph-queries | no | — | no Neo4j read-query change |

**Verdict:** reconciled — every touched domain consulted; ADR-0016 + cli catalog regenerated.
```

- [ ] **Step 2: Regenerate the CLI catalog** (the `knowledge-check` target must appear there, mirroring how `code-*` targets were added):

```bash
make cli-index    # or: ~/.pyenv/shims/python -m tools.cli index
```

- [ ] **Step 3: Reconcile clean**

```bash
~/.pyenv/shims/python -m tools.knowledge check   # expect: knowledge-check: clean
~/.pyenv/shims/python -m tools.cli check          # expect: cli-check: clean
```

`knowledge-check: clean` = cascade covers all 7 domains AND every adoption-dated spec/plan (this spec + this plan) now carries the addendum; pre-adoption files grandfathered.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-05-knowledge-cascade-and-honesty-check-design.md \
        docs/superpowers/plans/2026-08-05-knowledge-cascade-and-honesty-check.md docs/cli/
git commit -m "docs(knowledge): dogfood the honesty-check addendum on this spec+plan; refresh CLI catalog"
```

---

### Task 7: Capture ADR-0016

**Files:**
- Create: `docs/adr/0016-*.md` (via scaffold)
- Modify (regenerate): `docs/adr/index.md`, `docs/adr/log.md`

- [ ] **Step 1: Scaffold the ADR**

```bash
~/.pyenv/shims/python -m tools.adr new "Adopt knowledge cascade and spec/plan honesty check"
```

- [ ] **Step 2: Fill the scaffold** — `status: accepted`; `date: 2026-08-05`; `source:` = `docs/superpowers/specs/2026-08-05-knowledge-cascade-and-honesty-check-design.md`; `governs:` = `docs/index.md`, `tools/knowledge/`, `.claude/settings.json`. Body (durable what/why, no spec detail): the disclosure model is a cascade root + lightweight pointer (structure over injection, per the researched context-engineering guidance); a non-blocking spec/plan honesty check records per-domain consultation as an addendum; the ADR before-hook is slimmed to a provisional pointer with a retirement criterion. Does **not** supersede ADR-0015 — it extends the disclosure model around the ADR corpus.

- [ ] **Step 3: Regenerate + verify**

```bash
make adr-index
~/.pyenv/shims/python -m tools.adr check     # adr-check: clean (or only the 3 pre-existing staleness warnings)
```

- [ ] **Step 4: Commit**

```bash
git add docs/adr/
git commit -m "docs(adr): ADR-0016 — adopt knowledge cascade + spec/plan honesty check"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/knowledge/ tests/adr/ -p no:cacheprovider -q -o addopts=""` — all green.
- [ ] `make knowledge-check` — clean.
- [ ] `make cli-check` — clean (knowledge-check target catalogued).
- [ ] `make adr-check` — clean apart from the 3 known pre-existing staleness warnings.
- [ ] `docs/index.md` renders as a table on GitHub; every domain links to a real `index.md`.
- [ ] Live hooks: an architectural prompt yields the ADR pointer (not the table); a Write to a `specs/` or `plans/` path yields the honesty-check reminder; a Write elsewhere is silent.
- [ ] `scripts/with-adr-py.sh` is gone and nothing references it.
