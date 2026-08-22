# Layer-2 agentic eval — Mode B routine (subscription, session-driven)

`evals/graph/agentic.py` runs the Layer-2 harness against the Claude Code **subscription**,
never an API key. It has two modes:

- **Mode A** — fully headless. `python -m evals.graph.agentic` shells out to `claude -p` for
  both the agent-under-test and the judge. Only available where a headless `claude` binary can
  authenticate against the subscription non-interactively (`probe()` checks this).
- **Mode B** — this document. Where Mode A's `probe()` returns `False` (headless subscription
  auth unavailable, e.g. this sandbox), a live Claude Code **session** stands in for the headless
  subprocess: it dispatches a subagent per scenario, that subagent does the graph-only
  exploration, and the session writes the result to disk in the same shape `run_agent()` would
  have produced. `agentic.py --mode-b` then reads those files, judges them (if a judge subscription
  path is available), and aggregates — reusing `build_agent_prompt`, `build_judge_prompt`, and
  `aggregate` unchanged.

Run this routine as part of pre-commit verification whenever the graph traversal surface
(`tools/graph/traverse.py`, `tools/graph`, or a domain bundle it reads) changes, or when asked to
produce a Layer-2 baseline.

## Prerequisites

- You are running as a Claude Code session (interactive or a session-driven pre-commit hook) —
  not a headless subprocess. This routine depends on the session's own subagent dispatch, which
  is exactly what Mode A's headless path cannot do.
- `evals/graph/scenarios/*.json` exist and load via `evals.graph.run.load_scenarios()`.
- `evals/graph/.runs/` exists (created on first write below) and is **git-ignored** — it is
  scratch output for a single Mode B run, never committed. Treat any file already in it as stale
  unless you just wrote it in this routine.

## Procedure

For each scenario returned by `evals.graph.run.load_scenarios()`:

1. **Build the prompt.** Compute `evals.graph.agentic.build_agent_prompt(scenario)`. This is the
   exact prompt the headless agent-under-test would receive in Mode A — do not add or remove
   instructions; the harness is only comparable across modes if the prompt is identical.

2. **Dispatch a subagent** (the `general-purpose` agent type, or an equivalent fresh, tool-using
   agent with no memory of this session) with that prompt. Scope its tools to exactly the WALK /
   CONTEXT graph CLI — nothing else:
   - `python -m tools.graph walk <entry> --dir out|in|both --depth N|full --level module|symbol`
   - `python -m tools.graph context <entry>`
   The subagent must not read source files or repo docs directly; it may only see what these two
   commands return. This mirrors `_ALLOWED` / `_AGENT_SYSTEM` in `agentic.py` — the subagent
   should be told (in its own system framing) the same constraints `_AGENT_SYSTEM` encodes:
   investigate graph-only, start coarse and expand progressively toward governing intent, then
   outward, and say explicitly when the graph can't answer rather than guessing.

3. **Let it loop to a conclusion.** The subagent decides its own strategy — issue a `walk` or
   `context` call, read the result, decide whether to narrow/expand/stop, repeat — until it
   produces a clear final answer or explicitly reports the graph cannot answer the task. Do not
   hand it a fixed depth or direction; that would leak the recipe the prompt deliberately omits
   (see `test_agent_prompt_is_generic_and_tool_scoped`).

4. **Record the trajectory.** As the subagent runs its tool loop, capture each graph CLI
   invocation it made, in order (command + arguments), as a list of strings — this is the
   `trajectory` field `run_judge`/`build_judge_prompt` expects (the same shape `run_agent()`
   builds from `tool_use` events in Mode A's `stream-json` output).

5. **Write the result.** Once the subagent concludes, write:

   ```json
   {"id": "<scenario id>", "answer": "<subagent's final answer text>",
    "trajectory": ["<call 1>", "<call 2>", "..."]}
   ```

   to `evals/graph/.runs/<scenario id>.json` (create the `.runs/` directory if it doesn't exist
   yet). One file per scenario; overwrite if re-running.

6. Repeat for every scenario before moving to judging — do not judge one-by-one interleaved with
   dispatch, so a partial run is easy to spot (missing `.runs/<id>.json` files) before scoring.

## Judge + aggregate

Once every scenario has a `.runs/<id>.json`, run:

```bash
python -m evals.graph.agentic --mode-b
```

This reads each `evals/graph/.runs/<id>.json`, and:

- if a judge subscription path is available (`probe()` returns `True`), calls
  `run_judge(scenario, answer, trajectory)` per scenario — which builds the judge prompt via
  `build_judge_prompt` (rubric + task + gold context/answer + the agent's answer and trajectory,
  see `evals/graph/RUBRIC.md`) and shells out to `claude -p` for the verdict;
- otherwise, falls back to any `verdict` already present in the run file, or
  `{"verdict": "fail", "rationale": "no judge"}` if none.

It then prints `aggregate(records)` — `{"n", "pass", "partial", "fail"}` — as the run summary.

If the judge subscription path is also unavailable in this environment, judge the runs manually
using `evals/graph/RUBRIC.md` (the same rubric `build_judge_prompt` embeds) and record the verdict
by hand into each run file. `aggregate()` reads `record["verdict"]["verdict"]`, so the verdict must be
the **nested rubric object**, not a bare string — e.g.:

```json
{"id": "...", "answer": "...", "trajectory": ["..."],
 "verdict": {"answer_correctness": 2, "context_sufficiency": 2, "trajectory_quality": 2,
             "honesty": 2, "verdict": "pass", "rationale": "one line"}}
```

## Cleanup

`evals/graph/.runs/` is scratch — it is listed in `.gitignore` and must never be committed. Clear
it (or just let the next run overwrite it) once you've captured the aggregate result you need
(e.g. pasted into `evals/graph/RESULTS.md` or a spec's baseline section).

## Optional: run Layer 1 as a pre-commit check (local, non-blocking)

The deterministic Layer-1 scorecard is fast and CI-safe; wire it as a local pre-commit if you want
regressions surfaced before each commit. Install a git hook (opt-in — the repo does not force one):

```bash
cat > .git/hooks/pre-commit <<'HOOK'
#!/usr/bin/env bash
# Non-blocking: print the graph agentic-fitness scorecard; never fail the commit.
make eval-graph || true
HOOK
chmod +x .git/hooks/pre-commit
```

Layer 2 (`make eval-graph-agentic` / this routine) stays **manual** — it spawns real agents on the
subscription and is not suited to run on every commit.
