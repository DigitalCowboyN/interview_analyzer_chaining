# Category-axis Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the capability category axis's reserved/defined state machine-readable and guarded, so a category used before it's defined is flagged instead of drifting silently.

**Architecture:** Turn `CATEGORIES` (in `tools/capability/reader.py`) from a list into an ordered dict `{name: definition}` (reserved = `""`) — backward-compatible with all existing membership/iteration usages. Add a cross-domain `check_category_axis` to the knowledge domain that flags any category used by a capability or use-case but not yet defined. Reconcile the README prose.

**Tech Stack:** Python 3 (stdlib), pytest, Makefile. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-07-category-axis-reconciliation-design.md`.

## Global Constraints

- **Non-blocking:** the new check returns `list[Finding]` and never raises; `knowledge` `run_all` stays exit-0.
- **Backward-compatible axis swap:** `CATEGORIES` list→dict must not change behavior of `capability/check.py`, `capability/render.py`, `usecase/check.py`, `usecase/render.py` — all use only `x in CATEGORIES` / `x not in CATEGORIES` (dict tests keys) and `for c in CATEGORIES` (iterates keys). The rendered `docs/capabilities/index.md` and `docs/use-cases/index.md` must be **byte-identical** after the swap (verify by regen).
- **Reserved = empty definition.** `category_defined(name)` is the single predicate; no other place hard-codes the reserved rule.
- **The new check is cross-domain and complementary:** it flags *used-but-reserved* (in `CATEGORIES` but undefined). It does NOT flag *used-but-unknown* (not in `CATEGORIES`) — that stays with the per-domain checks. It reads both capabilities and use-cases.
- **No new ADR** (refines ADR-0018's mechanism).
- **Names used across tasks (verbatim):** `CATEGORIES` (dict), `category_defined`, `check_category_axis`.

---

### Task 1: `CATEGORIES` carries definitions + `category_defined`

**Files:**
- Modify: `tools/capability/reader.py:12-15` (the `CATEGORIES` comment + assignment)
- Test: `tests/capability/test_reader.py` (add cases; file exists)

**Interfaces:**
- Produces: `CATEGORIES: dict[str, str]` (name → definition, `""` = reserved), `category_defined(name: str) -> bool`.

- [ ] **Step 1: Write the failing test**

Add to `tests/capability/test_reader.py`:

```python
from tools.capability.reader import CATEGORIES, category_defined


def test_categories_is_defined_axis():
    # membership + iteration still behave like the old list
    assert "product" in CATEGORIES and "nonsense" not in CATEGORIES
    assert list(CATEGORIES)[:2] == ["product", "operations"]  # order preserved for render
    # product/operations/supporting are defined; strategic is reserved ("")
    assert category_defined("product") and category_defined("operations")
    assert category_defined("supporting")
    assert not category_defined("strategic")   # reserved
    assert not category_defined("unknown")     # not in the axis
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/capability/test_reader.py::test_categories_is_defined_axis -v`
Expected: FAIL — `ImportError: cannot import name 'category_defined'` (and dict shape absent).

- [ ] **Step 3: Write minimal implementation**

Replace the comment + list at `tools/capability/reader.py:12-15` with:

```python
# The capability category axis — an open, ordered set carrying each value's definition.
# A non-empty definition = defined & in use; "" = reserved (declared; define on first use).
# Adding/promoting a value = one edit here; knowledge-check flags a used-but-undefined value.
# Kept dict-shaped so `x in CATEGORIES` / `for c in CATEGORIES` behave as before (keys).
CATEGORIES = {
    "product": "the product itself — the capability a customer directly uses",
    "operations": "running and maintaining the system — CI, infra, projections, the guarded knowledge graph",
    "supporting": "customer-facing but around the product, not the product itself — self-help, notifications, getting output out",
    "strategic": "",  # reserved: direction-setting; define on first use
}


def category_defined(name: str) -> bool:
    """True when `name` is a category with a real definition (not reserved / unknown)."""
    return bool(CATEGORIES.get(name))
```

- [ ] **Step 4: Run tests to verify pass + backward-compat**

Run: `python -m pytest tests/capability tests/usecase -v`
Expected: PASS — the new test passes AND every existing capability/use-case reader/check/render test still passes (membership + iteration parity).

- [ ] **Step 5: Verify generated indexes are byte-identical (the swap is transparent)**

Run: `make capability-index && make usecase-index && git diff --stat docs/capabilities/index.md docs/use-cases/index.md`
Expected: no diff (empty output). If either changed, the key order diverged — reorder `CATEGORIES` keys to match the old `[product, operations, strategic, supporting]` visible order and re-verify.

- [ ] **Step 6: Commit**

```bash
git add tools/capability/reader.py tests/capability/test_reader.py
git commit -m "feat(capability): category axis carries definitions (dict) + category_defined"
```

---

### Task 2: `check_category_axis` in the knowledge guard

**Files:**
- Modify: `tools/knowledge/check.py` (imports + new check + wire into `run_all`)
- Test: `tests/knowledge/test_check.py` (add cases; file exists)

**Interfaces:**
- Consumes: `tools.capability.reader.CATEGORIES`, `category_defined`, `load_capabilities`; `tools.usecase.reader.load_use_cases`.
- Produces: `check_category_axis(root=".") -> list[Finding]`, wired into `run_all`.

- [ ] **Step 1: Write the failing test**

Add to `tests/knowledge/test_check.py`:

```python
from tools.knowledge.check import check_category_axis, run_all


def _cap(tmp_path, slug, category):
    d = tmp_path / "docs" / "capabilities"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.md").write_text(
        f"---\ntype: Capability\nkind: primary\ntier: core\ncategory: {category}\n"
        f"implemented_by: []\n---\n{slug}.\n", encoding="utf-8")


def test_flags_used_but_reserved_category(tmp_path):
    _cap(tmp_path, "x", "strategic")          # strategic is reserved ("" definition)
    findings = check_category_axis(str(tmp_path))
    assert any("strategic" in f.message and "in use" in f.message for f in findings)


def test_clean_when_used_categories_are_defined(tmp_path):
    _cap(tmp_path, "x", "product")            # defined
    _cap(tmp_path, "y", "supporting")         # defined
    assert check_category_axis(str(tmp_path)) == []


def test_run_all_includes_axis_and_never_raises(tmp_path):
    assert isinstance(run_all(str(tmp_path)), list)   # empty repo: no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/knowledge/test_check.py -k category -v`
Expected: FAIL — `ImportError: cannot import name 'check_category_axis'`.

- [ ] **Step 3: Write minimal implementation**

In `tools/knowledge/check.py`, add imports below the stdlib imports:

```python
from tools.capability.reader import CATEGORIES, category_defined, load_capabilities
from tools.usecase.reader import load_use_cases
```

Add the check (after `check_cascade_covers_domains`):

```python
def check_category_axis(root: str = ".") -> List[Finding]:
    """Cross-domain: every category USED by a capability or use-case must be DEFINED,
    not a reserved placeholder. Complements the per-domain 'unknown category' checks
    (which flag values not in the axis at all)."""
    try:
        used: dict = {}
        for node in (*load_capabilities(root), *load_use_cases(root)):
            if node.category:
                used[node.category] = used.get(node.category, 0) + 1
    except Exception as exc:  # non-blocking: a guard must never raise out
        return [Finding(f"knowledge: category-axis check failed: {exc}")]
    findings: List[Finding] = []
    for cat, n in sorted(used.items()):
        if cat in CATEGORIES and not category_defined(cat):
            findings.append(Finding(
                f"knowledge: category '{cat}' is in use ({n} node(s)) but has no "
                f"definition in tools/capability/reader.py — define it before use"))
    return findings
```

Wire it into `run_all` (after the cascade check):

```python
    findings += check_cascade_covers_domains(root)
    findings += check_category_axis(root)
    findings += check_addendum_present(specs, plans)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/knowledge/test_check.py -v`
Expected: PASS (new cases + existing).

- [ ] **Step 5: Commit**

```bash
git add tools/knowledge/check.py tests/knowledge/test_check.py
git commit -m "feat(knowledge): check_category_axis — flag a category used but not defined"
```

---

### Task 3: Reconcile the README prose + full sweep

**Files:**
- Modify: `docs/capabilities/README.md` (the category section: frontmatter comment + the "Categories" bullet)

**Interfaces:** none (docs + verification).

- [ ] **Step 1: Update the README category prose**

In `docs/capabilities/README.md`, change the frontmatter-example comment from
`category: product | operations   # primaries carry category (industry axis; strategic/supporting reserved)`
to
`category: product | operations | supporting   # industry axis; definitions live in tools/capability/reader.py (strategic reserved)`

and replace the **Categories** bullet (currently "product and operations are populated; strategic and supporting are reserved") with:

```markdown
- **Categories** follow industry and are an **open, defined set** — each value's meaning
  lives in `tools/capability/reader.py` (`CATEGORIES`), and `make knowledge-check` flags a
  value used before it's defined. Current axis: **product** (the thing customers use),
  **operations** (running the system — CI, infra, the knowledge graph), **supporting**
  (customer-facing but around the product — self-help, notifications, output access).
  `strategic` (direction-setting) is reserved — define it on first use.
```

- [ ] **Step 2: Full sweep — the current drift is now cleared**

```bash
make knowledge-check     # clean — 'supporting' is now defined (was the silent drift)
make capability-check    # clean
make usecase-check       # unchanged advisories only
make health              # full sweep
python -m pytest tests/capability tests/usecase tests/knowledge -q
```

Expected: `knowledge-check: clean` (proves the reserved-but-used `supporting` drift is resolved); capability/use-case checks unchanged; all tests green.

- [ ] **Step 3: Commit**

```bash
git add docs/capabilities/README.md
git commit -m "docs(capability): reconcile category prose — supporting defined, axis is machine-guarded"
```

---

## After all tasks

No ADR (refines ADR-0018's mechanism — noted in the spec's knowledge-graph check). Run the final whole-branch review on the most capable model, then use **superpowers:finishing-a-development-branch**.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-07.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| capabilities | yes | `CATEGORIES` list→dict + `category_defined` (Task 1); README prose (Task 3) | backward-compatible; indexes byte-identical |
| knowledge | yes | new cross-domain `check_category_axis` in `run_all` (Task 2) | mirrors cascade/graph cross-domain checks |
| use-cases | yes (read-only) | the reconciler reads use-case categories; no node change | — |
| adr | yes | refines ADR-0018 (no new ADR) | noted, not captured |
| code / graph / cli / glossary / api / prompts / graph-queries / tests | no | — | unaffected |

**Verdict:** reconciled — capabilities (axis + prose) + knowledge (the new guard) are the subjects; use-cases consulted read-only; ADR-0018 refinement noted.
