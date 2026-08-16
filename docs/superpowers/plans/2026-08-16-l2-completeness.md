# L2 — Completeness & currency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two completeness signals that make the graph trustworthy — `check_reachability` (code no intent/decision reaches, via L1's `walk`) in the graph domain, and `check_unregistered_types` (a declared `type:` nobody registered) in the corpus domain. Both non-blocking, wired into their domain's `run_all`. Completes R2.

**Architecture:** Two new check functions added to existing files — `tools/graph/check.py` (uses `tools.graph.traverse.walk`) and `tools/corpus/check.py` (reuses the L0 markdown scan). No new tools, no new Make targets, no generated-index changes.

**Tech Stack:** Python 3 (stdlib), pytest. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-16-l2-completeness-design.md`. **ADRs:** none new — realizes the program spec's L2; ADR-0016/0023 govern non-blocking.

## Global Constraints

- **Both checks are non-blocking** — they return `List[Finding]`; the domain CLIs still `return 0`.
- **No generated index changes** — these are check functions; do not regenerate or alter any `docs/**/index.md`.
- **Reuse, don't rebuild** — reachability uses `tools.graph.traverse.walk`; unregistered-type reuses `tools.corpus.reader._iter_markdown` + `parse_front_matter`. Do not reimplement traversal or markdown scanning.
- **Names verbatim:** `check_reachability(root=".")`, `check_unregistered_types(root=".")`.
- **Known registered document types = `OKF_HOMES` keys** (ADR, Capability, UseCase, CodeUnit, Term).

---

### Task 1: `check_reachability` (unexplained code)

**Files:**
- Modify: `tools/graph/check.py` (add `check_reachability`, wire into `run_all`)
- Test: `tests/graph/test_reachability.py`

**Interfaces:**
- Consumes: `tools.graph.reader.nodes`, `tools.graph.traverse.walk`, the module's `Finding`.
- Produces: `check_reachability(root=".") -> List[Finding]`.

- [ ] **Step 1: Write the failing test** — `tests/graph/test_reachability.py`:

```python
import tools.graph.check as gc
from tools.graph.check import check_reachability
from tools.graph.traverse import Subgraph, Node


def _sg(addresses):
    return Subgraph(nodes={a: Node(address=a, type="", context="") for a in addresses}, edges=[])


def test_unreached_code_unit_is_flagged(monkeypatch):
    monkeypatch.setattr(gc, "nodes", lambda root=".": {
        "Capability": {"cap"}, "UseCase": set(), "ADR": set(), "CodeUnit": {"reached", "orphan"}})
    # walk from the intents reaches only code:reached
    monkeypatch.setattr(gc, "walk",
                        lambda entry, direction="both", depth=None, root=".": _sg(["capabilities:cap", "code:reached"]))
    msgs = [f.message for f in check_reachability()]
    assert any("code:orphan" in m for m in msgs)
    assert not any("code:reached" in m for m in msgs)


def test_all_reached_is_clean(monkeypatch):
    monkeypatch.setattr(gc, "nodes", lambda root=".": {
        "Capability": {"cap"}, "UseCase": set(), "ADR": set(), "CodeUnit": {"reached"}})
    monkeypatch.setattr(gc, "walk",
                        lambda entry, direction="both", depth=None, root=".": _sg(["capabilities:cap", "code:reached"]))
    assert check_reachability() == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/graph/test_reachability.py -q --no-cov`
Expected: FAIL — `cannot import name 'check_reachability'`.

- [ ] **Step 3: Implement.** In `tools/graph/check.py` add the import near the top (with the other `tools.graph` imports):

```python
from tools.graph.traverse import walk
```

Add the function (place it beside the other `check_*` functions):

```python
def check_reachability(root: str = ".") -> List[Finding]:
    """Code the graph cannot explain: a CodeUnit reached by no Capability / UseCase / ADR.

    One multi-start walk outward from every "why" node; anything not in the reached set has no
    path from an intent, a use-case, or a decision (nor is a dependency of anything that does)."""
    ns = nodes(root)
    intents = ([f"capabilities:{i}" for i in ns.get("Capability", ())]
               + [f"use-cases:{i}" for i in ns.get("UseCase", ())]
               + [f"adr:{i}" for i in ns.get("ADR", ())])
    reached = set(walk(intents, direction="out", depth=None, root=root).nodes)
    code = {f"code:{u}" for u in ns.get("CodeUnit", ())}
    return [Finding(f"graph: code unit {a} is reached by no capability / use-case / ADR (unexplained)")
            for a in sorted(code - reached)]
```

Wire it into `run_all` (add after the existing `findings += ...` lines, before the return):

```python
    findings += check_reachability(root)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/graph/test_reachability.py -q --no-cov`
Expected: PASS (2 passed).

- [ ] **Step 5: Sanity-check on the real repo**

Run: `python -c "from tools.graph.check import check_reachability; fs=check_reachability(); print(len(fs), 'unreached'); [print(' ', f.message) for f in fs[:10]]"`
Expected: a small, sensible count (a handful of unexplained units, not all 48 and not zero). Note the count — it is advisory signal, not a failure.
Run: `python -m tools.graph check` — still exits 0; reachability findings appear as advisory warnings.

- [ ] **Step 6: Commit**

```bash
git add tools/graph/check.py tests/graph/test_reachability.py
git commit -m "feat(graph): check_reachability — flag code no capability/use-case/ADR reaches (uses walk)"
```

---

### Task 2: `check_unregistered_types` (new-domain detection, declared half)

**Files:**
- Modify: `tools/corpus/check.py` (add `check_unregistered_types`, wire into `run_all`)
- Test: `tests/corpus/test_unregistered_types.py`

**Interfaces:**
- Consumes: `tools.corpus.reader._iter_markdown` + `_IGNORE_DIRS`, `src.ingestion.front_matter.parse_front_matter`, `OKF_HOMES`, the module's `Finding`.
- Produces: `check_unregistered_types(root=".") -> List[Finding]`.

- [ ] **Step 1: Write the failing test** — `tests/corpus/test_unregistered_types.py`:

```python
import os

from tools.corpus.check import check_unregistered_types


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def test_unregistered_type_is_flagged(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "docs/policies/p.md"), "---\ntype: Policy\n---\nbody\n")
    _write(os.path.join(root, "docs/capabilities/c.md"), "---\ntype: Capability\n---\nok\n")
    msgs = [f.message for f in check_unregistered_types(root)]
    assert any("Policy" in m for m in msgs)
    assert not any("Capability" in m for m in msgs)   # registered → not flagged


def test_only_registered_types_is_clean(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "docs/adr/a.md"), "---\ntype: ADR\n---\nok\n")
    assert check_unregistered_types(root) == []


def test_body_fenced_type_is_not_flagged(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "docs/plans/p.md"), "# A plan\n\n```\ntype: Policy\n```\n")
    assert check_unregistered_types(root) == []   # top frontmatter only, not body
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/corpus/test_unregistered_types.py -q --no-cov`
Expected: FAIL — `cannot import name 'check_unregistered_types'`.

- [ ] **Step 3: Implement.** In `tools/corpus/check.py` add imports:

```python
import collections

from src.ingestion.front_matter import parse_front_matter
from tools.corpus.reader import _IGNORE_DIRS, _iter_markdown, okf_records
```

(the file already imports `okf_records` — merge the import line; do not duplicate.)

Add the function:

```python
def check_unregistered_types(root: str = ".") -> List[Finding]:
    """A `.md` whose OWN top frontmatter declares a `type:` that is not a registered document
    type. okf_records silently skips these, so a new *kind* of record is invisible until wired
    in. This surfaces it. (Declared-type detection; undeclared new domains stay the hard case.)"""
    unknown = collections.Counter()
    for path in _iter_markdown(root, _IGNORE_DIRS):
        try:
            fm, _ = parse_front_matter(open(path, encoding="utf-8", errors="ignore").read())
        except OSError:
            continue
        t = fm.get("type") if fm else None
        if t and t not in OKF_HOMES:
            unknown[t] += 1
    return [Finding(f"corpus: '{t}' is declared as a type on {n} file(s) but is not a registered "
                    f"node type — wire it in, or it stays invisible to the graph")
            for t, n in sorted(unknown.items())]
```

Wire into `run_all`:

```python
def run_all(root: str = ".") -> List[Finding]:
    return check_misfiled(okf_records(root)) + check_unregistered_types(root)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/corpus/test_unregistered_types.py -q --no-cov`
Expected: PASS (3 passed).

- [ ] **Step 5: Sanity-check on the real repo**

Run: `python -m tools.corpus check`
Expected: `corpus-check: clean` — only the 5 registered types exist repo-wide today, so no findings. (This check starts silent and fires only when a new declared kind appears.)

- [ ] **Step 6: Commit**

```bash
git add tools/corpus/check.py tests/corpus/test_unregistered_types.py
git commit -m "feat(corpus): check_unregistered_types — flag a declared type nobody registered"
```

---

## After all tasks

Run `python -m tools.corpus check` and `python -m tools.graph check` (both exit 0; reachability findings are advisory, corpus clean). Run the full unit suite (`make test-unit`) and confirm green. Confirm the freshness gate is unaffected (`make regen-derived && git diff --exit-code` → CLEAN — no index changed). No new ADR. Run the final whole-branch review on the most capable model, then use **superpowers:finishing-a-development-branch**.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-16.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| graph | yes | `check_reachability` + `run_all` (uses `traverse.walk`) | the subject |
| corpus | yes | `check_unregistered_types` + `run_all` (reuses L0 scan) | the subject |
| code / capabilities / use-cases | no (read-only) | reachability reads their node sets; logic unchanged | — |
| adr | yes | no new ADR — realizes L2; ADR-0016/0023 govern | — |
| cli | no | no new target (checks join existing `run_all`) | — |

**Verdict:** reconciled — graph + corpus gain two non-blocking completeness checks; no new tool, target, ADR, or generated index.
