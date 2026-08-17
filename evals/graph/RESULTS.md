# Graph agentic-eval results — docstring backlog lift

Run 2026-08-17. **Before** = `14f5872` (harness landed, no docstrings). **After** = `5968138`
(all 61 backlog modules documented). Reproduce: `python -m evals.graph.run --root <checkout>`.

## Deterministic lift (Layer 1)

| scenario | category | coverage (before → after) | recall | precision | overfetch |
| --- | --- | --- | --- | --- | --- |
| explore-tools-graph | exploration | **0.00 → 1.00** | 1.00 | 0.88 | 1 |
| spec-code-intake | spec | **0.00 → 1.00** | 0.70 | 0.09 | 68 |
| trace-classify-obligation | implement | — (no code gold) | 1.00 | 0.05 | 20 |

- **Context coverage** — the docstring-sensitive number — rose **0 → 100%** on both scenarios whose
  gold includes code nodes. Every `tools.graph` module (S1) and every reached code node the code-intake
  spec depended on (S2) now carries substantive context when an agent walks the graph.
- **recall / precision / overfetch are unchanged** across before/after — as expected: they measure
  graph *structure* (which nodes are reachable), which docstrings do not touch. They are reported to
  keep the coverage lift honest and to surface structural gaps (below).

## Agentic proof (Layer 2) — scenario 1, graph-context-only

Two agents were each given ONLY the `walk(code:tools.graph, out, 1)` output (no source-file access)
and asked to summarize the package and each module.

- **Before** (all node contexts empty) — the agent **declared the context insufficient and refused to
  guess**: *"every node's content is marked `<empty>` … Module names alone are suggestive but I cannot
  responsibly infer their actual behavior from names — that would be guessing."*
- **After** (modules documented) — the agent produced an **accurate module-by-module summary from the
  graph alone**: `__main__` = the CLI (index/check/neighbors/walk), `reader` = harvest nodes+edges,
  `registry` = the node/edge schema, `traverse` = `walk()` the ephemeral subgraph, `render` = the
  Markdown catalogs, `check` = non-blocking drift/reachability, `classify` = derived category/determinism
  axes. It also honestly noted the package-root context is still empty.

This is the graph functioning as an agent's "spontaneous short-term memory": before the backlog it
could not, after it could — without the agent reading a single source file.

## Findings (inputs to the future full-framework milestone)

1. **Docstring lift confirmed** — context coverage 0 → 100% (S1, S2). The backlog measurably improved
   the graph's fitness for exploration and spec-context gathering.
2. **Govern-edge gap (S2)** — `adr:19`, `adr:20`, `adr:24` are in the gold but **unreachable**: the
   knowledge-graph-tooling ADRs carry no `governs` edge to the `tools/` code they govern, so an agent
   walking for spec context never surfaces the decisions that constrain it. Docstrings do not fix this;
   it needs `governs:` frontmatter (or a derived tooling-ADR link). A real "plan-vs-ADR" eval would fail
   here today.
3. **Over-fetch / minimality (S2)** — a depth-2 both-direction walk from two code seeds pulls **68**
   nodes beyond the gold (precision 0.09). "Grab the *small* necessary context" wants a tighter
   retrieval (shallower/directional walk, or a spec-context helper), not the raw neighborhood.
4. **Bonus lift beyond the graph** — the same module docstrings populated the human-facing CLI catalog
   (`docs/cli/index.md`): every `python -m tools.<domain>` row gained a real description (was blank).
5. **Residual** — package `__init__.py` files have no docstring, so package-root nodes
   (`code:tools.graph`) still surface empty context. A follow-up increment (package docstrings) would
   close it; modules were the 61-item backlog and are done.

## Deferred (per the spec)

Full framework (all 6 categories, a scenario corpus, thresholds, a `make evals` target, a CI gate, an
automated Layer-2 judge); evals over the transcript/product graph; symbols. Findings 2 and 3 are the
concrete work the full-framework milestone should target — the graph reaches the right *neighborhood*
but not yet the right *small, decision-linked* context.
