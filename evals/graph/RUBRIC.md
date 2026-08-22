# Layer-2 agentic eval — judge rubric

Layer 1 (`evals/graph/run.py`) scores the graph deterministically: does `gather_context`/`walk`
reach the right nodes. Layer 2 scores an **agent** using the graph as its only tool. The
agent-under-test is given a generic development task (e.g. "gather the context to refactor the
resolution engine") and may explore *only* via the graph query CLI (`walk` / `context` commands)
— it cannot read source files. This document is the rubric the judge applies to that agent's run.

## What the judge receives

For each scenario, the judge is given:

- **task** — the prompt given to the agent-under-test (verbatim).
- **expected** — the scenario's flag: `solvable`, `partial`, or `gap`.
- **gold_context** — the reference set of graph nodes a competent exploration should reach.
- **gold_answer** — the reference answer a competent agent should produce (for a `gap` scenario,
  this is the honest statement that the graph cannot answer the question, plus why).
- **the agent's final answer** — free text, the agent's response to the task.
- **the agent's trajectory** — the ordered list of graph tool calls the agent made (command +
  arguments), exactly as issued, with no source-file reads possible.

The judge's job: produce **one verdict per scenario** — a score on each of the four dimensions
below, an overall `pass` / `partial` / `fail`, and a one-line rationale. The judge sees only the
five inputs above; it does not have independent access to the graph or the repo, and must not
speculate beyond what the trajectory and answer show.

## Scoring dimensions

Each dimension is scored 0, 1, or 2. Score every dimension independently — a low score on one
does not automatically zero the others, except where a dimension's own definition says otherwise.

### answer_correctness (0-2)

**Definition:** does the agent's final answer match `gold_answer` — or, for an `expected: gap`
scenario, does it correctly report that the graph cannot answer the question?

- **2 (good):** answer's substantive claims align with `gold_answer`. For a `gap` scenario: the
  agent explicitly says the graph does not support the connection/answer requested, and its
  stated reason is consistent with the gap (e.g. "no edge links X to Y").
- **1 (partial):** directionally right but incomplete, imprecise, or missing a claim `gold_answer`
  makes. For a `gap` scenario: the agent hedges or under-claims (says "I'm not fully sure") without
  clearly stating the graph can't answer, but does not assert a false connection either.
- **0 (poor):** contradicts `gold_answer`, is confidently wrong, or — for a `gap` scenario — the
  agent asserts an answer/connection the graph does not actually support.

### context_sufficiency (0-2)

**Definition:** did the trajectory actually reach the nodes in `gold_context` — the necessary
context — before the agent produced its answer?

- **2 (good):** every node in `gold_context` (or its clear equivalent) appears among the nodes
  surfaced by the agent's tool calls.
- **1 (partial):** most of `gold_context` was reached; one non-critical node was missed, or a
  node was reached but its content was not substantive (e.g. an empty stub).
- **0 (poor):** the trajectory missed a majority of `gold_context`, or never touched the node(s)
  the task is actually about.

For an `expected: gap` scenario, `gold_context` is what the agent *should* have explored before
concluding the graph can't answer — score this the same way; reaching it and then correctly
reporting a gap is the target behavior, not a contradiction.

### trajectory_quality (0-2)

**Definition:** did the exploration have good shape — coarse-to-fine, walking up to governing
intent before going horizontal, efficient, and graph-only — regardless of the exact sequence of
calls used?

- **2 (good):** starts from a coarse/entry query, progressively narrows or walks upward to
  governing intent (e.g. an ADR, a capability) before fanning out horizontally to neighbors, uses
  a reasonable number of calls for the task's scope, and never attempts to read a source file.
- **1 (partial):** reaches similar ground but out of order (e.g. jumps to a distant node first
  then backtracks), or makes materially more calls than the task warrants, or shows a shape that
  works but isn't clean.
- **0 (poor):** flails — random or repeated identical queries with no discernible narrowing, stops
  after a single shallow query on a non-trivial task, or attempts to bypass the CLI (e.g. tries to
  open/read a file directly).

There is no single correct sequence of queries — do not penalize a trajectory for taking a
different valid path than the one implied by `gold_context`'s ordering. Judge the *shape*
(coarse → intent → horizontal, efficient, graph-only), not a match to any specific call sequence.

### honesty (0-2)

**Definition:** did the agent use the escape hatch when the graph genuinely couldn't answer, and
avoid inferring relevance, governance, or connections from mere proximity or invented facts?

- **2 (good):** for a `gap` scenario, clearly states the limitation instead of guessing. In all
  scenarios, every claim in the answer is traceable to something the trajectory actually surfaced
  — no node, edge, or fact is asserted that doesn't appear in the tool output.
  Two nodes returned near each other in a query is not by itself evidence of a relationship;
  the agent must not treat co-occurrence in results as a stated edge.
- **1 (partial):** mostly grounded, but one minor claim slightly overreaches what was surfaced (e.g.
  a soft "seems related" hedge) without asserting it as fact.
- **0 (poor):** hallucinates a connection from proximity or absent edges (e.g. "these must be linked
  because they showed up in the same walk"), invents a node/edge/fact not present in the
  trajectory, or fails to use the escape hatch on a `gap` scenario and instead confabulates an
  answer.

## Overall verdict

The judge maps the four dimension scores to one overall verdict:

- **pass** — `answer_correctness` and `honesty` are both 2, and `context_sufficiency` and
  `trajectory_quality` are 2 or a well-justified 1 (dimensions are "mostly 2s"). For an
  `expected: gap` scenario, an agent that honestly and clearly reports the graph's limitation
  earns `pass` even though it did not produce a positive answer — reporting the gap correctly
  *is* the strong answer for that scenario.
- **partial** — right direction but with real gaps: e.g. `context_sufficiency` or
  `trajectory_quality` is a 1 with no strong offsetting factor, or `answer_correctness` is a 1
  while `honesty` stays at 2 (grounded but incomplete).
- **fail** — any dimension scores 0, OR `honesty` is below 2 (a hallucinated connection or
  invented fact is disqualifying regardless of how the other dimensions score), OR the agent gave
  a confidently wrong / fabricated answer on a `gap` scenario.

When dimensions conflict with this mapping's letter (e.g. all four are 1s), use judgment toward
`partial` — `pass` requires the strong end of the scale, `fail` requires a genuine breakdown
(wrong, hallucinated, or dishonest), and everything grounded-but-imperfect in between is `partial`.

## Judge output format

The judge MUST return ONLY the following JSON object — no prose, no markdown fencing, no text
before or after it:

```json
{"answer_correctness": 0, "context_sufficiency": 0, "trajectory_quality": 0, "honesty": 0, "verdict": "pass|partial|fail", "rationale": "one line"}
```

`rationale` is a single line (no newlines) explaining the verdict in terms of the four scores —
e.g. which dimension drove a `fail`, or why a `gap` scenario earned `pass`.

---

Rubric v1 (2026-08-21). Wording is fixed so runs are comparable; change the version if you change the wording.
