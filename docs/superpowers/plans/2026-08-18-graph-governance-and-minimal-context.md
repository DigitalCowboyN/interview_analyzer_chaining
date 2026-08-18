# Graph self-governance + minimal-context retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the top eval-found gaps — author `governs` edges so the KG tooling's own ADRs link to their code, add a `gather_context` minimal-context retrieval, expose `--level` + a `context` command on the CLI, and trim phantom call edges.

**Architecture:** Four independent, small changes. `governs` is authored frontmatter on 5 ADRs (resolves by path). `gather_context` is a new function in `tools/graph/traverse.py` composed of bounded `walk` calls. The CLI gains `--level` + a `context` subcommand. `calls_of` gets a builtin-method denylist. Continues on branch `feat/graph-self-complete-and-fixes` (Phase 1 intent nodes already committed).

**Tech Stack:** Python 3 (stdlib), pytest. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-18-graph-governance-and-minimal-context-design.md`. **ADRs:** none new; `governs` edges realize decisions ADR-0020–0027 already implied.

## Global Constraints

- **`governs` paths are directory/module style, no `.py`** — verified: `tools/graph/traverse` resolves to `code:tools.graph.traverse`; `…/traverse.py` resolves to nothing.
- **Only true `governs` edges** — each added edge must genuinely reflect the ADR constraining that code; every new edge must resolve (no dangling, per `graph-check`).
- **Module-grain `walk` is unchanged** — `gather_context` is additive; the harvest-equivalence regression stays green.
- **Names verbatim:** `gather_context(entry, root=".", level="module", max_up=6)`, CLI `context` subcommand, `walk --level`.

---

### Task 1: Author `governs` edges for the KG-tooling ADRs

**Files:**
- Modify: `docs/adr/0020-*.md`, `docs/adr/0024-*.md`, `docs/adr/0025-*.md`, `docs/adr/0026-*.md`, `docs/adr/0027-*.md` (add/replace the `governs:` frontmatter list)
- Test: `tests/graph/test_tooling_governs.py`

- [ ] **Step 1: Add the `governs:` frontmatter** to each ADR's frontmatter block (between the existing keys, before the closing `---`). Set exactly:

```yaml
# docs/adr/0020-*.md
governs:
  - tools/graph/
```
```yaml
# docs/adr/0024-*.md
governs:
  - tools/corpus/
```
```yaml
# docs/adr/0025-*.md
governs:
  - tools/graph/traverse
  - tools/graph/neighbors
```
```yaml
# docs/adr/0026-*.md
governs:
  - tools/code/
```
```yaml
# docs/adr/0027-*.md
governs:
  - tools/graph/traverse
  - tools/graph/neighbors
  - tools/code/reader
```

If an ADR already has a `governs:` key (e.g. `governs: []`), replace it with the block above. Preserve all other frontmatter.

- [ ] **Step 2: Write the test** — `tests/graph/test_tooling_governs.py`:

```python
from tools.graph.reader import harvest
from tools.graph.check import check_endpoints
from tools.graph.reader import nodes


def _governs(root="."):
    return {(e.src, e.dst) for e in harvest(root) if e.type == "governs"}


def test_tooling_adrs_now_govern_their_code():
    g = _governs()
    assert ("adr:27", "code:tools.graph.traverse") in g       # lazy walk governs traversal
    assert ("adr:27", "code:tools.code.reader") in g          # + symbol derivation
    assert ("adr:25", "code:tools.graph.neighbors") in g      # ephemeral substrate
    assert ("adr:26", "code:tools.code.reader") in g          # code intake (tools/code/ dir)
    assert ("adr:20", "code:tools.graph.traverse") in g       # graph model (tools/graph/ dir)


def test_no_dangling_after_governs():
    assert check_endpoints(harvest("."), nodes(".")) == []    # every governs endpoint resolves
```

- [ ] **Step 3: Run to verify it fails, then passes** — `python -m pytest tests/graph/test_tooling_governs.py -q --no-cov`. Before the frontmatter edits it FAILS (edges absent); after, PASS (2 passed).

- [ ] **Step 4: Real-repo check** — `python -m tools.graph check` → no `does not resolve` finding. Confirm an agent walking up now reaches the ADR:

```bash
python -c "from tools.graph.traverse import walk; sg=walk('code:tools.graph.traverse','in',3,level='module'); print(sorted(a for a in sg.nodes if a.startswith('adr:')))"
```
Expected: includes `adr:25`, `adr:27`, `adr:20` (was empty before).

- [ ] **Step 5: Regenerate + commit**

```bash
make regen-derived            # graph/adr indexes pick up the new governs edges
python -m flake8 tests/graph/test_tooling_governs.py
git add docs/adr tests/graph/test_tooling_governs.py docs/graph/index.md docs/graph/graph.md docs/adr/index.md docs/adr/by-code.md docs/adr/log.md
git commit -m "feat(adr): author governs edges for the KG-tooling ADRs (0020/0024/0025/0026/0027)"
```

(Also `git add` any other file `make regen-derived` changed, then confirm `make regen-derived && git diff --exit-code` is clean.)

---

### Task 2: `gather_context` — minimal task-context retrieval

**Files:**
- Modify: `tools/graph/traverse.py` (add `gather_context`)
- Test: `tests/graph/test_gather_context.py`

**Interfaces:**
- Produces: `gather_context(entry, root=".", level="module", max_up=6) -> Subgraph`.

- [ ] **Step 1: Write the failing test** — `tests/graph/test_gather_context.py`:

```python
from tools.graph.traverse import gather_context, walk


def test_gather_context_is_small_and_reaches_intent():
    # entry: the traversal module. Full closure is ~644 nodes; gather_context must be far smaller
    # and must include the governing ADR (post-Task-1) and the entry's own out-neighbors.
    sg = gather_context("code:tools.graph.traverse", level="module")
    full = walk("code:tools.graph.traverse", direction="both", depth=None, level="module")
    assert len(sg.nodes) < len(full.nodes)                     # minimal, not the whole closure
    assert "code:tools.graph.traverse" in sg.nodes             # the entry
    # reached the nearest governing intent by walking up (post-Task-1, adr:27 governs it directly)
    assert any(a.partition(":")[0] in ("adr", "capabilities", "use-cases") for a in sg.nodes)


def test_gather_context_stops_at_first_intent_layer(monkeypatch):
    # if intent appears at depth 1 in, we should not climb further than needed
    import tools.graph.traverse as tr
    calls = []
    real = tr.walk

    def spy(entry, direction="both", depth=None, root=".", level="module"):
        calls.append((direction, depth))
        return real(entry, direction=direction, depth=depth, root=root, level=level)

    monkeypatch.setattr(tr, "walk", spy)
    gather_context("code:tools.graph.traverse", level="module")
    in_depths = [d for (dr, d) in calls if dr == "in"]
    # climbed progressively from depth 1 and stopped once intent was found (did not go to max_up=6)
    assert in_depths == sorted(in_depths) and max(in_depths) < 6
```

- [ ] **Step 2: Run to verify it fails** — `cannot import name 'gather_context'`.

- [ ] **Step 3: Implement** in `tools/graph/traverse.py` (after `walk`):

```python
_INTENT_SLUGS = ("capabilities", "use-cases", "adr")


def gather_context(entry, root: str = ".", level: str = "module", max_up: int = 6) -> Subgraph:
    """The minimal necessary context for a task targeting `entry`: walk UP progressively (following
    `in` edges) until the nearest governing intent (capability / use-case / ADR) appears — the
    shortest path up — unioned with the entry's direct OUT-neighbors (its deps / calls / contained
    symbols). Far smaller than the full closure; the agent-facing 'give me the right small context'."""
    up = walk(entry, direction="in", depth=1, root=root, level=level)
    for d in range(2, max_up + 1):
        if any(a.partition(":")[0] in _INTENT_SLUGS for a in up.nodes):
            break
        up = walk(entry, direction="in", depth=d, root=root, level=level)
    out = walk(entry, direction="out", depth=1, root=root, level=level)

    nodes = dict(up.nodes)
    nodes.update(out.nodes)
    seen, edges = set(), []
    for e in list(up.edges) + list(out.edges):
        k = (e.src, e.dst, e.type)
        if k not in seen:
            seen.add(k)
            edges.append(e)
    return Subgraph(nodes=nodes, edges=edges)
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/graph/test_gather_context.py -q --no-cov` → PASS.

- [ ] **Step 5: Real-repo sanity**

```bash
python -c "from tools.graph.traverse import gather_context; sg=gather_context('code:tools.graph.traverse'); print(len(sg.nodes),'nodes; adrs:',sorted(a for a in sg.nodes if a.startswith('adr:')))"
```
Expected: a small node count (≪ 644) that includes `adr:25`/`adr:27`.

- [ ] **Step 6: Commit**

```bash
python -m flake8 tools/graph/traverse.py tests/graph/test_gather_context.py
git add tools/graph/traverse.py tests/graph/test_gather_context.py
git commit -m "feat(graph): gather_context — progressive walk-up to intent + bounded local (minimal task context)"
```

---

### Task 3: CLI — `walk --level` and a `context` command

**Files:**
- Modify: `tools/graph/__main__.py`
- Test: `tests/graph/test_walk_cli.py` (extend) or a new `tests/graph/test_context_cli.py`

- [ ] **Step 1: Implement.** In `tools/graph/__main__.py`:

Add `--level` to the `walk` subparser and pass it through:

```python
    wp = sub.add_parser("walk")
    wp.add_argument("entry")
    wp.add_argument("--dir", default="both", choices=["out", "in", "both"])
    wp.add_argument("--depth", default="full")
    wp.add_argument("--level", default="module", choices=["module", "symbol"])
```

In `cmd_walk`, pass `level`:

```python
def cmd_walk(args) -> int:
    from tools.graph.traverse import walk
    depth = None if args.depth == "full" else int(args.depth)
    sg = walk(args.entry, direction=args.dir, depth=depth, level=args.level)
    _print_subgraph(args.entry, sg, f"dir={args.dir}, depth={args.depth}, level={args.level}")
    return 0
```

Factor the printing into a shared helper and add the `context` command:

```python
def _print_subgraph(entry, sg, meta) -> None:
    print(f"subgraph from {entry} ({meta}): {len(sg.nodes)} nodes, {len(sg.edges)} edges")
    for addr in sorted(sg.nodes):
        n = sg.nodes[addr]
        head = n.context.splitlines()[0] if n.context else ""
        print(f"  {addr}  [{n.type}]  {head[:80]}")
    for e in sg.edges:
        print(f"    {e.src} --{e.type}--> {e.dst}")


def cmd_context(args) -> int:
    from tools.graph.traverse import gather_context
    sg = gather_context(args.entry, level=args.level)
    _print_subgraph(args.entry, sg, f"minimal context, level={args.level}")
    return 0
```

Register the subcommand in `main` (both the parser and the dispatch dict):

```python
    cp = sub.add_parser("context")
    cp.add_argument("entry")
    cp.add_argument("--level", default="module", choices=["module", "symbol"])
    ...
    return {"index": cmd_index, "check": cmd_check, "neighbors": cmd_neighbors,
            "walk": cmd_walk, "context": cmd_context}[args.cmd](args)
```

- [ ] **Step 2: Test** — add to the CLI tests: `python -m tools.graph walk code:tools.graph.traverse --level symbol --depth 1` exits 0 and prints a symbol node (`code:tools.graph.traverse.walk`); `python -m tools.graph context code:tools.graph.traverse` exits 0 and prints a subgraph including an `adr:` line. Use `subprocess` like the existing CLI tests.

```python
import subprocess, sys

def test_walk_cli_level_symbol():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "walk",
                        "code:tools.graph.traverse", "--level", "symbol", "--depth", "1"],
                       capture_output=True, text=True)
    assert p.returncode == 0
    assert "code:tools.graph.traverse.walk" in p.stdout

def test_context_cli():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "context",
                        "code:tools.graph.traverse"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "adr:" in p.stdout            # minimal context reached the governing ADR
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/graph -k "cli or context" -q --no-cov
python -m flake8 tools/graph/__main__.py
git add tools/graph/__main__.py tests/graph/
git commit -m "feat(cli): tools.graph walk --level + context command (symbols + minimal context)"
```

---

### Task 4: Trim phantom call edges

**Files:**
- Modify: `tools/code/reader.py` (`calls_of`)
- Test: `tests/code/test_calls.py` (extend)

- [ ] **Step 1: Write the failing test** — add to `tests/code/test_calls.py`:

```python
def test_builtin_method_calls_are_not_edges(tmp_path):
    _w(str(tmp_path / "src/svc/__init__.py"), "")
    _w(str(tmp_path / "src/svc/render.py"), "def harvest():\n    return 1\n")
    _w(str(tmp_path / "src/svc/m.py"),
       "from src.svc import render\n\nMAP = {}\n\n"
       "def run():\n    MAP.get('x')\n    render.harvest()\n")
    by_id = {s.id: s for s in symbols_of("svc.m", str(tmp_path))}
    calls = set(by_id["svc.m.run"].calls)
    assert "svc.render.harvest" in calls          # real submodule call kept
    assert not any(c.endswith(".get") for c in calls)   # MAP.get() (builtin) dropped
```

- [ ] **Step 2: Run to verify it fails** (`MAP.get` currently emits a `.get` edge).

- [ ] **Step 3: Implement** in `tools/code/reader.py`. Add the denylist near `_CALLS_MARKER`:

```python
# builtin container/str/obj method names — an `x.get()` / `x.append()` call is not a graph edge
_BUILTIN_METHODS = frozenset({
    "get", "keys", "values", "items", "pop", "setdefault", "update", "copy", "clear",
    "append", "extend", "insert", "add", "discard", "remove", "sort", "reverse", "index",
    "count", "join", "split", "strip", "lstrip", "rstrip", "format", "replace", "startswith",
    "endswith", "lower", "upper", "encode", "decode", "read", "write", "close",
})
```

In `calls_of`, guard the `Attribute` branch:

```python
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                    and f.value.id in name_index and f.attr not in _BUILTIN_METHODS:
                out.add(f"{name_index[f.value.id]}.{f.attr}")
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/code/test_calls.py -q --no-cov` → PASS.

- [ ] **Step 5: Real-repo check** — the phantom is gone:

```bash
python -c "from tools.code.reader import symbols_of; c=[s for s in symbols_of('tools.graph.reader') if s.id.endswith('.harvest')][0].calls; print('NODE_DOMAINS.get' in ' '.join(c), '<- should be False')"
```

- [ ] **Step 6: Commit**

```bash
python -m flake8 tools/code/reader.py tests/code/test_calls.py
git add tools/code/reader.py tests/code/test_calls.py
git commit -m "fix(code): drop phantom call edges to builtin container methods (.get/.append/...)"
```

---

### Task 5: Full gate + final review

- [ ] **Step 1: Full gate** — `make regen-derived && git diff --exit-code` CLEAN; `make test-unit` green; `python -m tools.graph check` no dangling; `python -m tools.adr check` (the new `governs` edges may add staleness *advisories* for module-precise paths — non-blocking; note any).
- [ ] **Step 2: Final whole-branch review** on the most capable model with a review package (`scripts/review-package "$(git merge-base main HEAD)" HEAD`) — this branch includes Phase 1 (intent nodes) + Phase 2 (these fixes). Then use **superpowers:finishing-a-development-branch**.

## After all tasks

- An agent walking `in` from `code:tools.graph.traverse` reaches ADR-0025/0027 (govern gap closed).
- `gather_context` returns a small subgraph (≪ 644) with the entry, its 1-hop neighbors, and the nearest governing intent; exposed as `python -m tools.graph context`.
- `walk --level symbol` reaches symbols from the CLI.
- Phantom `.get`/`.append` call edges are gone.
- Full suite green; freshness clean; module-grain equivalence intact.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-18.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| adr | yes | `governs:` on 0020/0024/0025/0026/0027 | the govern-gap fix |
| graph | yes | `gather_context` + CLI `--level`/`context` | minimality + agent access |
| code | yes | `calls_of` builtin-method denylist | edge quality |
| capabilities / use-cases | yes (Phase 1, committed) | walk-the-graph-for-context + gather-context-with-the-graph | intent capture |

**Verdict:** reconciled — the tooling's ADRs now govern their code, an agent can retrieve minimal context and reach symbols from the CLI, and phantom edges are trimmed. No new ADR. Flow-nodes, eval suite, and symbol backlog remain deferred.
