# Category-axis reconciliation — design

**Status:** approved by owner 2026-08-07 (brainstorm dialogue).
**Program:** a small self-consistency fix for the guarded knowledge graph. The capability
**category axis** (`product | operations | supporting | strategic`) had a "reserved vs
in-use" distinction that lived **only as prose** in `docs/capabilities/README.md`. The
code treated all four values as equally valid, so when the tests round began *using*
`supporting` (3 use-cases), nothing flagged that the docs still called it reserved. The
system's whole purpose is catching its docs drifting from reality; on this axis it had no
eyes. This makes the reserved/defined state **machine-readable and guarded**, so the drift
is detected the moment it happens.

## The problem (concretely)

- `tools/capability/reader.py` defines `CATEGORIES = ["product", "operations", "strategic",
  "supporting"]` — a flat list. Every check just tests membership; nothing knows which
  values are "defined and in use" vs "reserved placeholder."
- `docs/capabilities/README.md` says *"`strategic` and `supporting` are reserved."* That
  claim is prose — unreconciled against actual usage.
- Reality today: `product`, `operations`, `supporting` are **used** (across capabilities +
  use-cases); `strategic` is **unused**. So the README is stale for `supporting`, and
  nothing caught it.

## Design

**1. Categories carry their own definition (one source of truth).**
Turn `CATEGORIES` from a list into an **ordered dict** `{name: definition}` in
`tools/capability/reader.py`. A non-empty definition = **defined / in-use**; an empty
string = **reserved** (declared, not yet defined). This is backward-compatible with all
five existing usages — every one is either `x in CATEGORIES` / `x not in CATEGORIES`
(dict membership tests keys, identical) or `for c in CATEGORIES` (iterates keys in
insertion order, identical). No change to `capability/check.py`, `capability/render.py`,
`usecase/check.py`, or `usecase/render.py`.

```python
# tools/capability/reader.py  — the axis + its definitions (reserved = "")
CATEGORIES = {
    "product":     "the product itself — the capability a customer directly uses",
    "operations":  "running and maintaining the system — CI, infra, projections, the guarded knowledge graph",
    "supporting":  "customer-facing but around the product, not the product itself — self-help, notifications, getting output out",
    "strategic":   "",  # reserved: direction-setting; define on first use
}
```

A tiny helper makes intent explicit and keeps callers from hard-coding the rule:

```python
def category_defined(name: str) -> bool:
    return bool(CATEGORIES.get(name))
```

**2. The guard grows eyes (cross-domain reconciliation, in the knowledge domain).**
Add `check_category_axis(root)` to `tools/knowledge/check.py` and wire it into `run_all`.
It reads **both** capabilities and use-cases (either can populate a category — same
cross-domain reach the graph guard already uses), collects the categories actually in use,
and flags any that is a **known-but-undefined** value:

- **used-but-reserved** → finding: *"category 'supporting' is in use (by N nodes) but has
  no definition in tools/capability/reader.py — define it."*

This is complementary to the existing per-domain checks, which already flag a category
that is **not in the axis at all** (`capability/check.py`'s classification check,
`usecase/check.py`'s `check_categories`). Those stay; this adds the *reserved-but-used*
case no single-domain check owns. Non-blocking, `return 0`, like every guard.

Optional (low value, include only if trivial): a soft advisory for a **defined-but-never-
used** category (dead definition). Deferred unless it falls out for free — YAGNI.

**3. Define the categories now.** Fill in `product / operations / supporting` per the
owner's model (above); leave `strategic` reserved (`""`). This promotes `supporting` out of
reserved and **clears today's drift immediately** — `knowledge-check` goes clean.

**4. Reconcile the prose.** Update `docs/capabilities/README.md` so its category section
matches: `supporting` is defined and in use (with its meaning); `strategic` remains the
only reserved value; point to `tools/capability/reader.py` as the machine source of truth
for the definitions (the README explains, the code enforces).

## Where the check lives — and why

The category axis is a **shared vocabulary** across capabilities and use-cases, so
"is every used category defined" is inherently **cross-domain** — no single per-domain
check can own it. This mirrors the established split:

- per-domain `*-check` validates that domain's own nodes (incl. "category is a known
  value");
- `graph-check` validates cross-domain **edge** integrity;
- `knowledge-check` validates cross-domain **meta** consistency (cascade covers domains,
  spec addenda present) — and now **the shared category axis**.

Putting it in `knowledge` keeps the capability and use-case domains from reaching sideways
into each other, and puts the reconciliation where the other cross-cutting reconciliations
already live.

## Non-goals

- **Renaming or restructuring categories** — `operations` and `supporting` stay distinct
  (they are: operations = running the system; supporting = customer-facing around it).
- **Making the axis closed** — it stays an open, ordered set; adding a value is still one
  edit. This only requires that a value be *defined* before it's *used*.
- **A per-category node type or its own domain** — the axis is a small shared vocabulary,
  not a domain. YAGNI.
- **Blocking** on any finding.

## Testing

- **Unit** (`tests/knowledge/`): `check_category_axis` flags a synthetic capability/use-case
  fixture that uses a reserved (empty-definition) category, and passes when every used
  category is defined; assert it never raises. `category_defined` returns False for `""`
  / unknown, True for a defined value. Confirm `capability/check.py` + `usecase/check.py` +
  both renders still pass unchanged against the dict-shaped `CATEGORIES` (membership +
  iteration parity).
- **Smoke:** `make knowledge-check` clean on the real repo after the definitions land
  (proves the current `supporting` drift is cleared); `make capability-check` /
  `make usecase-check` / `make capability-index` / `make usecase-index` unchanged (the dict
  swap is transparent); regenerate no generated files change shape.

## ADR

No new ADR. This **refines the mechanism** of ADR-0018 (which introduced the category axis
with `strategic`/`supporting` reserved) — turning "reserved" from prose into a guarded,
machine-readable state. It reverses no decision and adds no new node/edge type; capturing
it would duplicate ADR-0018. Noted here per the knowledge-graph check.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-07.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| capabilities | yes | `CATEGORIES` gains definitions (list→dict, backward-compatible); README prose reconciled | axis owner |
| knowledge | yes | new cross-domain `check_category_axis` in `run_all` | mirrors cascade/graph cross-domain checks |
| use-cases | yes (read-only) | the reconciler reads use-case categories too | no node change |
| adr | yes | refines ADR-0018 (no new ADR) | noted, not captured |
| code / graph / cli / glossary / api / prompts / graph-queries / tests | no | — | unaffected |

**Verdict:** reconciled — capabilities (axis + prose) and knowledge (the new guard) are the
subjects; use-cases consulted read-only; ADR-0018 refinement noted.
