# CLI-Surface Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A generated, drift-checked catalog of the project's command surface: every Makefile target + `python -m` entry point, tagged everyday/internal, with a self-updating `make help`, a full `docs/cli/index.md` for agents, and a non-blocking reconciliation guard.

**Architecture:** A new **stdlib-only** `tools/cli/` package mirroring `tools/adr`'s reader → render → check → CLI split. It parses the Makefile + walks for `__main__.py`, renders `make help` and `docs/cli/index.md`, and reconciles docs against the real surface.

**Tech Stack:** Python 3 **stdlib only** (os, re, dataclasses, typing — no third-party, no `yaml`, no `src.*`), pytest, Make.

## Global Constraints

- **Non-blocking, always.** Checks return `list[Finding]`; none raises; `make cli-check` / the CLI exit 0.
- **Stdlib-only** for `tools/cli/*` — no third-party imports, no `import yaml`, no `from src...`. (So generated `make help` runs under any interpreter, incl. the yaml-less Homebrew python3.)
- **Doc convention:** `target: … ## desc` → everyday; `target: … ##@ desc` → internal; no `##` → undocumented.
- **`make help`** shows everyday **make** targets only. **`docs/cli/index.md`** shows everything (targets labeled everyday/internal + module entry points).
- `Command` / `Finding` are local to `tools/cli` (independent of `tools/adr`).
- Run tests with `~/.pyenv/shims/python -m pytest <path> -v`.

---

### Task 1: `reader.py` — Command model + Makefile/entrypoint parsing

**Files:**
- Create: `tools/cli/__init__.py` (empty), `tools/cli/reader.py`
- Test: `tests/cli/__init__.py` (empty), `tests/cli/test_reader.py`

**Interfaces:**
- Produces:
  - `@dataclass Command(name: str, kind: str, description: str, visibility: str)` — `kind ∈ {"make","module"}`, `visibility ∈ {"everyday","internal","undocumented"}`
  - `parse_makefile(path: str) -> list[Command]`
  - `module_entrypoints(root: str, subdirs=("src","tools")) -> list[Command]`

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_reader.py
from tools.cli.reader import parse_makefile, module_entrypoints, Command

MAKEFILE = """\
PYTHON := python
.PHONY: test lint

test: ## Run the tests
\t$(PYTHON) -m pytest

lint: ## Lint the code
\tflake8 src

wait-db: ##@ Wait for the test DB
\tsleep 1

mystery:
\techo hi
"""

def test_parse_makefile_classifies(tmp_path):
    mk = tmp_path / "Makefile"; mk.write_text(MAKEFILE, encoding="utf-8")
    cmds = {c.name: c for c in parse_makefile(str(mk))}
    assert cmds["test"].visibility == "everyday" and cmds["test"].description == "Run the tests"
    assert cmds["wait-db"].visibility == "internal" and cmds["wait-db"].description == "Wait for the test DB"
    assert cmds["mystery"].visibility == "undocumented" and cmds["mystery"].description == ""
    assert "PYTHON" not in cmds        # := assignment is not a target
    assert all(c.kind == "make" for c in cmds.values())

def test_module_entrypoints(tmp_path):
    pkg = tmp_path / "src" / "thing"; pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""Thing tool.\n\nmore\n"""\n', encoding="utf-8")
    (pkg / "__main__.py").write_text("print('hi')\n", encoding="utf-8")
    cmds = module_entrypoints(str(tmp_path))
    assert any(c.name == "python -m src.thing" and c.description == "Thing tool." for c in cmds)
    assert all(c.kind == "module" and c.visibility == "everyday" for c in cmds)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/cli/test_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.cli'`

- [ ] **Step 3: Implement**

```python
# tools/cli/reader.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

_TARGET = re.compile(r"^([a-zA-Z][\w-]*):")


@dataclass
class Command:
    name: str
    kind: str          # "make" | "module"
    description: str
    visibility: str    # "everyday" | "internal" | "undocumented"


def parse_makefile(path: str) -> List[Command]:
    out: dict = {}
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("\t"):
            continue
        if ":=" in line.split("##", 1)[0]:
            continue  # variable assignment, not a target
        m = _TARGET.match(line)
        if not m:
            continue
        name = m.group(1)
        if "##@" in line:
            desc, vis = line.split("##@", 1)[1].strip(), "internal"
        elif "##" in line:
            desc, vis = line.split("##", 1)[1].strip(), "everyday"
        else:
            desc, vis = "", "undocumented"
        # prefer a documented entry if the name recurs
        if name not in out or (not out[name].description and desc):
            out[name] = Command(name, "make", desc, vis)
    return list(out.values())


def _package_doc(dirpath: str) -> str:
    for fn in ("__init__.py", "__main__.py"):
        p = os.path.join(dirpath, fn)
        if not os.path.exists(p):
            continue
        m = re.search(r'"""(.*?)"""', open(p, encoding="utf-8").read(), re.DOTALL)
        if m:
            first = m.group(1).strip().splitlines()[0].strip() if m.group(1).strip() else ""
            if first:
                return first
    return ""


def module_entrypoints(root: str, subdirs=("src", "tools")) -> List[Command]:
    cmds: List[Command] = []
    for base in subdirs:
        start = os.path.join(root, base)
        if not os.path.isdir(start):
            continue
        for dirpath, _dirs, files in os.walk(start):
            if "__main__.py" in files:
                dotted = os.path.relpath(dirpath, root).replace(os.sep, ".")
                cmds.append(Command(f"python -m {dotted}", "module", _package_doc(dirpath), "everyday"))
    return sorted(cmds, key=lambda c: c.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/cli/test_reader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/cli/__init__.py tools/cli/reader.py tests/cli/__init__.py tests/cli/test_reader.py
git commit -m "feat(cli): Command model + Makefile/entrypoint reader (stdlib-only)"
```

---

### Task 2: `render.py` — help + catalog rendering

**Files:**
- Create: `tools/cli/render.py`
- Test: `tests/cli/test_render.py`

**Interfaces:**
- Consumes: `tools.cli.reader.Command`
- Produces: `render_help(commands) -> str` (everyday make targets only), `render_catalog(commands) -> str` (all, labeled)

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_render.py
from tools.cli.reader import Command
from tools.cli.render import render_help, render_catalog

CMDS = [
    Command("test", "make", "Run the tests", "everyday"),
    Command("wait-db", "make", "Wait for the test DB", "internal"),
    Command("mystery", "make", "", "undocumented"),
    Command("python -m src.lens", "module", "Lens engine.", "everyday"),
]

def test_render_help_shows_only_everyday_make():
    out = render_help(CMDS)
    assert "test" in out and "Run the tests" in out
    assert "wait-db" not in out            # internal hidden
    assert "python -m src.lens" not in out  # modules not in make help

def test_render_catalog_shows_all_labeled():
    out = render_catalog(CMDS)
    assert "test" in out and "wait-db" in out and "internal" in out
    assert "python -m src.lens" in out and "Lens engine." in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/cli/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.cli.render'`

- [ ] **Step 3: Implement**

```python
# tools/cli/render.py
from __future__ import annotations

from typing import List

from tools.cli.reader import Command


def render_help(commands: List[Command]) -> str:
    everyday = [c for c in commands if c.kind == "make" and c.visibility == "everyday"]
    lines = ["Usage: make [target]", ""]
    for c in sorted(everyday, key=lambda c: c.name):
        lines.append(f"  {c.name:24s} {c.description}")
    return "\n".join(lines) + "\n"


def render_catalog(commands: List[Command]) -> str:
    makes = sorted((c for c in commands if c.kind == "make"), key=lambda c: c.name)
    mods = sorted((c for c in commands if c.kind == "module"), key=lambda c: c.name)
    lines = ["# CLI surface", "", "## Make targets", "",
             "| command | visibility | description |", "| --- | --- | --- |"]
    for c in makes:
        lines.append(f"| {c.name} | {c.visibility} | {c.description} |")
    lines += ["", "## Module entry points", "", "| command | description |", "| --- | --- |"]
    for c in mods:
        lines.append(f"| {c.name} | {c.description} |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/cli/test_render.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/cli/render.py tests/cli/test_render.py
git commit -m "feat(cli): help + catalog renderers"
```

---

### Task 3: `check.py` — reconciliation guard

**Files:**
- Create: `tools/cli/check.py`
- Test: `tests/cli/test_check.py`

**Interfaces:**
- Consumes: `tools.cli.reader` (`parse_makefile`, `module_entrypoints`, `Command`), `tools.cli.render.render_catalog`
- Produces: `@dataclass Finding(message)`, `check_docs_reference_real(commands, doc_paths)`, `check_catalog_in_sync(catalog_path, commands)`, `check_undocumented(commands)`, `run_all(root=".") -> list[Finding]`

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_check.py
from tools.cli.reader import Command
from tools.cli.check import (
    check_docs_reference_real, check_catalog_in_sync, check_undocumented, Finding,
)

CMDS = [
    Command("test", "make", "Run tests", "everyday"),
    Command("mystery", "make", "", "undocumented"),
    Command("python -m src.lens", "module", "Lens engine.", "everyday"),
]

def test_docs_reference_real_flags_only_backticked_missing(tmp_path):
    doc = tmp_path / "CLAUDE.md"
    doc.write_text("Run `make test` then `make gone`. Also make sure to `python -m src.gone`.\n", encoding="utf-8")
    msgs = " ".join(f.message for f in check_docs_reference_real(CMDS, [str(doc)]))
    assert "make gone" in msgs                      # backticked, not real -> flagged
    assert "src.gone" in msgs                        # not a real entry point
    assert "make sure" not in msgs and "make test" not in msgs   # prose + real command not flagged

def test_catalog_in_sync(tmp_path):
    from tools.cli.render import render_catalog
    cat = tmp_path / "index.md"
    cat.write_text("stale\n", encoding="utf-8")
    assert check_catalog_in_sync(str(cat), CMDS)     # out of sync
    cat.write_text(render_catalog(CMDS), encoding="utf-8")
    assert check_catalog_in_sync(str(cat), CMDS) == []

def test_undocumented_informational():
    msgs = " ".join(f.message for f in check_undocumented(CMDS))
    assert "mystery" in msgs and "test" not in msgs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/cli/test_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.cli.check'`

- [ ] **Step 3: Implement**

```python
# tools/cli/check.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

from tools.cli.reader import Command, module_entrypoints, parse_makefile
from tools.cli.render import render_catalog

# command mentions are only counted inside inline code spans (leading backtick),
# so prose like "make sure" is not a false positive.
_MAKE_MENTION = re.compile(r"`make\s+([a-zA-Z][\w-]*)")
_MODULE_MENTION = re.compile(r"`python\s+-m\s+([\w.]+)")


@dataclass
class Finding:
    message: str


def check_docs_reference_real(commands: List[Command], doc_paths: List[str]) -> List[Finding]:
    real_make = {c.name for c in commands if c.kind == "make"}
    real_mod = {c.name.split()[-1] for c in commands if c.kind == "module"}
    findings: List[Finding] = []
    for dp in doc_paths:
        if not os.path.exists(dp):
            continue
        text = open(dp, encoding="utf-8").read()
        base = os.path.basename(dp)
        for m in _MAKE_MENTION.finditer(text):
            if m.group(1) not in real_make:
                findings.append(Finding(f"{base} references `make {m.group(1)}` which is not a real target"))
        for m in _MODULE_MENTION.finditer(text):
            if m.group(1) not in real_mod:
                findings.append(Finding(f"{base} references `python -m {m.group(1)}` which is not a real entry point"))
    return findings


def check_catalog_in_sync(catalog_path: str, commands: List[Command]) -> List[Finding]:
    want = render_catalog(commands)
    have = open(catalog_path, encoding="utf-8").read() if os.path.exists(catalog_path) else ""
    if want != have:
        return [Finding("docs/cli/index.md out of sync — run `make cli-index`")]
    return []


def check_undocumented(commands: List[Command]) -> List[Finding]:
    return [Finding(f"make target `{c.name}` has no ## description")
            for c in commands if c.kind == "make" and c.visibility == "undocumented"]


def run_all(root: str = ".", docs=("CLAUDE.md", "README.md"),
            catalog: str = "docs/cli/index.md") -> List[Finding]:
    commands = parse_makefile(os.path.join(root, "Makefile")) + module_entrypoints(root)
    findings: List[Finding] = []
    findings += check_docs_reference_real(commands, [os.path.join(root, d) for d in docs])
    findings += check_catalog_in_sync(os.path.join(root, catalog), commands)
    findings += check_undocumented(commands)
    return findings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/cli/test_check.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/cli/check.py tests/cli/test_check.py
git commit -m "feat(cli): docs-reconcile, catalog-sync, undocumented guard checks"
```

---

### Task 4: CLI + Makefile targets + stdlib-only guard

**Files:**
- Create: `tools/cli/__main__.py`
- Modify: `Makefile` (add `cli-index`, `cli-check` targets, each with a `##` doc)
- Test: `tests/cli/test_cli.py`

**Interfaces:**
- Produces: `python -m tools.cli {help|index|check}` (all exit 0)

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_cli.py
import glob
import subprocess
import sys

def test_cli_help_and_check_exit_zero():
    for cmd in ("help", "check"):
        proc = subprocess.run([sys.executable, "-m", "tools.cli", cmd], capture_output=True, text=True)
        assert proc.returncode == 0, (cmd, proc.stderr)

def test_tools_cli_is_stdlib_only():
    banned = ("import yaml", "from src", "import pydantic", "import requests")
    for path in glob.glob("tools/cli/*.py"):
        src = open(path, encoding="utf-8").read()
        for b in banned:
            assert b not in src, f"{path} contains banned import: {b}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/cli/test_cli.py -v`
Expected: FAIL — `No module named tools.cli.__main__` (help/check subcommands missing)

- [ ] **Step 3: Implement**

```python
# tools/cli/__main__.py
from __future__ import annotations

import argparse
import os
import sys

from tools.cli.check import run_all
from tools.cli.reader import module_entrypoints, parse_makefile
from tools.cli.render import render_catalog, render_help

CATALOG = "docs/cli/index.md"


def _commands(root: str = "."):
    return parse_makefile(os.path.join(root, "Makefile")) + module_entrypoints(root)


def cmd_help(args) -> int:
    print(render_help(_commands()), end="")
    return 0


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
    with open(CATALOG, "w", encoding="utf-8") as fh:
        fh.write(render_catalog(_commands()))
    print(f"wrote {CATALOG}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"cli-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("cli-check: clean")
    return 0  # NON-BLOCKING: always 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("help")
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"help": cmd_help, "index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

Add to `Makefile` (near `lint`), each self-documented:

```makefile
cli-index: ## Regenerate docs/cli/index.md (the CLI catalog)
	@$(PYTHON) -m tools.cli index

cli-check: ## Reconcile docs against the real CLI surface (non-blocking)
	@$(PYTHON) -m tools.cli check
```

- [ ] **Step 4: Run tests + smoke**

Run: `~/.pyenv/shims/python -m pytest tests/cli/ -v`
Expected: PASS (all cli tests)
Run: `~/.pyenv/shims/python -m tools.cli check`
Expected: prints `cli-check` output, exit 0 (findings expected until Task 5 — many undocumented targets + no catalog yet)

- [ ] **Step 5: Commit**

```bash
git add tools/cli/__main__.py Makefile tests/cli/test_cli.py
git commit -m "feat(cli): CLI (help/index/check) + make targets + stdlib-only guard"
```

---

### Task 5: Migration — self-document the Makefile, generate the catalog, wire help

Integration + content task: make the real surface self-describing and turn on the generator.

**Files:**
- Modify: `Makefile` (add `##`/`##@` docs to targets; replace the `help:` echo block with the generator)
- Create (generated): `docs/cli/index.md`

- [ ] **Step 1: Add `##` / `##@` docs to targets**

For each real target, add a trailing `## <desc>` (everyday) or `##@ <desc>` (internal). Seed the everyday descriptions from the text already in the current `help:` echo block; judge the rest (test/run/build/lint/format/ui-* = everyday; `wait-*`, `clean-*`, `*-up`/`*-down` infra, `db-test-*`, low-level helpers = internal). Every target should end with a `##` or `##@` comment.

- [ ] **Step 2: Replace the `help:` echo block with the generator**

Delete the entire hand-typed `help:` recipe (the `@echo` block) and replace with:

```makefile
help: ## Show the everyday commands
	@$(PYTHON) -m tools.cli help
```

- [ ] **Step 3: Generate the catalog + verify**

```bash
make cli-index          # writes docs/cli/index.md (all targets + entry points)
make help               # renders the everyday targets from the ## docs
make cli-check          # iterate until clean (fix docs that name a gone command; re-run cli-index)
```
`cli-check` clean means: no doc references a non-existent command, the catalog is in sync, and no target is left undocumented. (If a couple of deliberately-undocumented targets remain, that's an informational finding — acceptable, but prefer documenting them.)

- [ ] **Step 4: Commit**

```bash
git add Makefile docs/cli/index.md
git commit -m "docs(cli): self-document Makefile targets + generated help & catalog"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/cli/ -v` — all green.
- [ ] `make help` — renders the everyday commands (and now includes `adr-check`/`adr-index`, which the old hand-typed block omitted).
- [ ] `make cli-check` — clean (or only known-informational undocumented targets).
- [ ] `make cli-index` then `git status` — `docs/cli/index.md` regenerates identically (in sync).
- [ ] Confirm `docs/cli/index.md` lists all make targets (labeled) + the 7 `python -m` entry points.
