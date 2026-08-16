# L2 — Completeness & currency (design)

**Status:** proposed (brainstorm dialogue with owner, 2026-08-16).
**Program:** Phase L2 of the first-class knowledge graph
(`docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`). Completes
**R2** (L0 intake + L1 traversal + L2 completeness = a graph you can *trust*). Non-blocking, per
ADR-0016/0023. No new ADR.

## Purpose

L0 made intake complete; L1 made it walkable. L2 verifies the graph can be **trusted** — it adds
the two completeness signals that don't exist yet, without rebuilding what already works.

What is already covered (and therefore NOT L2's job): code orphans/coverage (`code-check`'s
`check_coverage` + `check_top_level_modules`), misfiled records (L0 `corpus-check`), dangling
edges (`graph-check`'s `check_endpoints`), currency (R1 forward loop). L2 fills exactly the two
remaining gaps.

## Check 1 — Reachability (unexplained code)

**Where:** `tools/graph/check.py::check_reachability`, using L1's `walk`.

Seed a single multi-start walk with the "why" node set — every **Capability ∪ UseCase ∪ ADR** —
direction `out`, depth to exhaustion (`walk` already accepts an iterable of addresses). The
result is every node reachable *from an intent, a use-case, or a decision*. Flag every
**`CodeUnit` that is not in the reached set**: code that no capability implements, no ADR
governs, and nothing reached depends on — i.e. code the graph cannot explain.

```python
def check_reachability(root=".") -> List[Finding]:
    ns = nodes(root)
    intents = ([f"capabilities:{i}" for i in ns.get("Capability", ())]
               + [f"use-cases:{i}" for i in ns.get("UseCase", ())]
               + [f"adr:{i}" for i in ns.get("ADR", ())])
    reached = set(walk(intents, direction="out", depth=None, root=root).nodes)
    code = {f"code:{u}" for u in ns.get("CodeUnit", ())}
    return [Finding(f"graph: code unit {a} is reached by no capability / use-case / ADR (unexplained)")
            for a in sorted(code - reached)]
```

Non-blocking and advisory — an unreached unit is a prompt ("why does this exist / what intent
covers it?"), not a failure. This signal was impossible before L1.

## Check 2 — Unregistered declared type (the achievable half of new-domain detection)

**Where:** `tools/corpus/check.py::check_unregistered_types`.

The corpus already scans every `.md`'s top frontmatter, but `okf_records` **silently skips** any
file whose `type:` is not in `OKF_HOMES` — so a brand-new *kind* of record (`type: Policy`,
`type: Runbook`, …) is invisible. This check does the opposite of skipping: it collects **every**
declared top-frontmatter `type:` value repo-wide and flags any that is not a registered document
type.

```python
def check_unregistered_types(root=".") -> List[Finding]:
    # scan all .md (same ignore list as okf_records); collect top-frontmatter type: values
    # flag any value not in OKF_HOMES (the registered document node types)
    ...
    return [Finding(f"corpus: '{t}' is declared as a type on {n} file(s) but is not a registered "
                    f"node type — wire it in, or it stays invisible to the graph")
            for t, n in sorted(unknown.items())]
```

Clean today (only the 5 registered types exist repo-wide), so it starts silent and fires only
when someone introduces a new declared kind. This is the concrete, achievable form of
"detect a new derived domain as it appears": for **declared** types. (Detecting *undeclared*
things that should be nodes but carry no marker remains the hard orphan class — partly served by
Check 1's reachability and by `code-check`; not fully solved here.)

## Wiring

Each check joins its domain's `run_all` (`tools.graph.check.run_all`, `tools.corpus.check.run_all`),
which already run in `make health` and the changed-domain pre-commit. No new Make targets. Both
stay non-blocking (return findings; the CLI returns 0).

## Non-goals (this phase)

- **Undeclared new-domain detection** (things that should be nodes but carry no marker) — remains
  the hard blind spot; reachability + `code-check` partially cover it, full solution is out of scope.
- **Rebuilding code orphan/coverage, misfiled, dangling, or currency** — already covered elsewhere.
- **Making the checks blocking** — non-blocking, per ADR-0023.
- **The `# okf:` marker migration** — a separate later phase.

## Testing

- **check_reachability:** on a small fake graph (monkeypatched `walk`/`nodes`), a code unit with
  no inbound intent path is flagged; one reached via `implements`/`governs`/a `depends_on` chain
  is not. On the real repo the check runs and returns a (possibly empty) list of `Finding`s
  without error.
- **check_unregistered_types:** a temp tree with a `type: Policy` doc → flagged; a tree with only
  registered types → no findings; a fenced `type: X` inside a plan body → NOT flagged (top
  frontmatter only, reusing the L0 discipline). On the real repo → clean (only the 5 types exist).
- Both appear in their domain's `run_all`; `graph-check` / `corpus-check` still exit 0.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-16.

| domain | touched? | note |
| --- | --- | --- |
| graph | yes — `check_reachability` added to `check.py` + `run_all` (uses `traverse.walk`) | the subject |
| corpus | yes — `check_unregistered_types` added to `check.py` + `run_all` | the subject |
| code / capabilities / use-cases / adr | no (read-only) | reachability reads their nodes; logic unchanged |
| adr | yes | no new ADR — realizes the program spec's L2; ADR-0016/0023 govern non-blocking |

**Verdict:** reconciled — graph + corpus are the subjects (two new non-blocking completeness
checks); no new ADR; nothing else changes.
