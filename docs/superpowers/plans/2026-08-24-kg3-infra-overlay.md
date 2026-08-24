# KG-3 Infra / Deployment Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the deployment topology walkable — derive `Service`/`EnvVar` nodes and four edges from `docker-compose.yml` so the graph answers "what must be up for X" and "what does service Y need," closing the two `deployment` gap scenarios.

**Architecture:** A new `tools/infra/` domain parses `docker-compose.yml` (PyYAML) into `Service` + `EnvVar` node data and four edge-pair sets. `tools/graph/reader.py` wraps that data into graph nodes/edges via new `_ADAPTERS` and `_DERIVED` entries; `tools/graph/registry.py` gains two `NODE_DOMAINS` rows and four `EdgeType` rows. Derived, never authored — the KG-2 pattern (an authoritative source + a `# talks-to:` marker fallback for the one fuzzy edge).

**Tech Stack:** Python 3.10, PyYAML 6.0.3 (already in `requirements.txt`), `ast` for import scanning, pytest.

## Global Constraints

- **Non-blocking checks:** every check function returns `List[Finding]`; the CLI prints and returns 0. Guard any check wired into a cross-domain `run_all` in `try/except`.
- **Derived, not authored:** `Service`/`EnvVar` are graph-derived (like `CodeUnit`, ADR-0026) — NOT OKF corpus docs, NOT added to `OKF_HOMES`.
- **Import layering (hard):** `tools/infra/reader.py` must NOT import from `tools/graph/reader.py` (which imports it) — it imports only `tools.code.reader`, `yaml`, stdlib. The `_derived_*` edge wrappers live in `tools/graph/reader.py` and import infra data functions **lazily inside the function body** (mirrors `_derived_writes` → `tools.graph.flow`).
- **Edge-name uniqueness:** the service→service edge verb is `requires` (NOT a second `depends_on`) — `tools/graph/render.py` keys the catalog by `et.name`, so a duplicate name double-counts/double-lists. `requires`, `runs`, `talks_to`, `configured_by` are all new verbs; none collide.
- **Secret-safety:** `EnvVar` nodes come from inline compose `environment:` ONLY. `.env` / `env_file` contents are never read or enumerated.
- **Freshness:** after adding node/edge types and a new domain, `make regen-derived && git diff --exit-code` must be clean — regenerate `docs/graph/{index,graph}.md`, `docs/infra/index.md`, `docs/cli/index.md`, `docs/code/index.md`, `docs/tests/index.md` and fold into the commit.
- DRY, YAGNI, TDD, frequent commits.

---

## File Structure

- **Create** `tools/infra/__init__.py`, `tools/infra/reader.py` (compose parse → `Service`/`EnvVar` + edge-pair functions), `tools/infra/render.py` (renders `docs/infra/index.md`), `tools/infra/check.py` (`check_infra` findings), `tools/infra/__main__.py` (`index`/`check`/`list` CLI).
- **Modify** `tools/graph/registry.py` (2 `NODE_DOMAINS` rows, 4 `EdgeType` rows), `tools/graph/reader.py` (2 `_ADAPTERS` rows, 4 `_DERIVED` builders), `Makefile` (`infra-index`, `infra-check`, add `infra-index` to `regen-derived`).
- **Create** `docs/infra/index.md` (generated), tests under `tests/infra/`.

---

### Task 1: Compose reader — `Service` + `EnvVar` nodes

**Files:**
- Create: `tools/infra/__init__.py` (empty), `tools/infra/reader.py`
- Modify: `tools/graph/registry.py` (NODE_DOMAINS), `tools/graph/reader.py` (_ADAPTERS)
- Test: `tests/infra/__init__.py` (empty), `tests/infra/test_reader.py`

**Interfaces:**
- Produces: `tools.infra.reader.Service(id, kind, image, command, ports, requires, env, loads_env_file)`, `EnvVar(name)`, `load_services(root=".") -> List[Service]`, `load_env_vars(root=".") -> List[EnvVar]`. `Service.id`/`EnvVar.name` are the node ids.
- Consumes: `docker-compose.yml`; nothing from `tools.graph`.

- [ ] **Step 1: Write the failing test**

```python
# tests/infra/test_reader.py
from tools.infra.reader import load_services, load_env_vars


def test_services_have_kind_axis():
    by = {s.id: s for s in load_services(".")}
    # every real compose service is present
    assert {"app", "worker", "projection-service", "redis", "neo4j", "eventstore"} <= set(by)
    # code services build our image + carry a command; backing services are image-only
    assert by["app"].kind == "code" and by["app"].command
    assert by["projection-service"].kind == "code"
    assert by["neo4j"].kind == "backing" and by["neo4j"].image.startswith("neo4j")
    assert by["eventstore"].kind == "backing"


def test_env_vars_are_inline_only_never_dotenv():
    names = {e.name for e in load_env_vars(".")}
    # inline `environment:` vars are modeled
    assert {"PROJECTION_LANE_COUNT", "ESDB_CONNECTION_STRING", "ENABLE_PROJECTION_SERVICE"} <= names
    # a .env-only secret must NOT appear (env_file contents are never read)
    assert not any(n in names for n in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEO4J_PASSWORD"))
    # services record that they load .env as an opaque boolean, not its contents
    assert {s.id: s.loads_env_file for s in load_services(".")}["app"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/infra/test_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: tools.infra.reader`

- [ ] **Step 3: Write `tools/infra/reader.py`**

```python
# tools/infra/reader.py
"""KG-3 infra overlay: Service/EnvVar node data and the four infra edge-pair sets, all derived
from docker-compose.yml (+ a client-lib map and `# talks-to:` markers). Consumed by
tools.graph.reader via _ADAPTERS + _DERIVED. Must not import tools.graph.reader (layering)."""
# governed-by: ADR-0029
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List

import yaml

COMPOSE = "docker-compose.yml"

# backing-service client library (top-level import) -> the compose service it connects to
_CLIENT_LIBS = {"neo4j": "neo4j", "esdbclient": "eventstore", "celery": "redis"}
_TALKS_MARKER = re.compile(r"#\s*talks-to:\s*([\w-]+)")
_SRC_TOKEN = re.compile(r"^src\.([\w.]+?)(?::\w+)?$")   # "src.main:app" -> "main"


@dataclass
class Service:
    id: str
    kind: str                                  # "code" | "backing"
    image: str = ""
    command: List[str] = field(default_factory=list)
    ports: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)   # service names (compose depends_on)
    env: List[str] = field(default_factory=list)         # inline environment var names
    loads_env_file: bool = False


@dataclass
class EnvVar:
    name: str


def _compose(root: str) -> dict:
    with open(os.path.join(root, COMPOSE), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _depends(spec: dict) -> List[str]:
    d = spec.get("depends_on")
    if isinstance(d, dict):
        return list(d)
    return list(d or [])


def _env_names(spec: dict) -> List[str]:
    e = spec.get("environment")
    if isinstance(e, dict):
        return list(e)
    return [str(x).split("=", 1)[0] for x in (e or [])]


def load_services(root: str = ".") -> List[Service]:
    out: List[Service] = []
    for name, spec in (_compose(root).get("services") or {}).items():
        spec = spec or {}
        out.append(Service(
            id=name,
            kind="code" if "build" in spec else "backing",
            image=spec.get("image", "") or "",
            command=list(spec.get("command") or []),
            ports=[str(p) for p in (spec.get("ports") or [])],
            requires=_depends(spec),
            env=_env_names(spec),
            loads_env_file=bool(spec.get("env_file")),
        ))
    return out


def load_env_vars(root: str = ".") -> List[EnvVar]:
    names = sorted({v for s in load_services(root) for v in s.env})
    return [EnvVar(n) for n in names]
```

- [ ] **Step 4: Register the node types**

In `tools/graph/registry.py`, add to `NODE_DOMAINS` (keep alphabetical/grouped with the others):

```python
    "Service": "service",
    "EnvVar": "env",
```

In `tools/graph/reader.py`, add a top-level import near the other domain loaders:

```python
from tools.infra.reader import load_services, load_env_vars
```

and add to `_ADAPTERS`:

```python
    "Service": (load_services, "id"),
    "EnvVar": (load_env_vars, "name"),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/infra/test_reader.py -v`
Expected: PASS (2 passed)

Also verify the nodes register and no import cycle:
Run: `PYTHONPATH=. python -c "from tools.graph.reader import nodes; n=nodes('.'); print(sorted(n['Service']), len(n['EnvVar']))"`
Expected: prints the 7 service ids and a nonzero EnvVar count.

- [ ] **Step 6: Commit**

```bash
git add tools/infra/__init__.py tools/infra/reader.py tools/graph/registry.py tools/graph/reader.py tests/infra/__init__.py tests/infra/test_reader.py
git commit -m "feat(infra): Service + EnvVar nodes derived from docker-compose (kind axis; inline env only)"
```

---

### Task 2: `requires` + `configured_by` edges (pure-compose)

**Files:**
- Modify: `tools/infra/reader.py` (add `requires_pairs`, `configured_by_pairs`), `tools/graph/registry.py` (2 EdgeType rows), `tools/graph/reader.py` (2 `_DERIVED` builders)
- Test: `tests/infra/test_edges_compose.py`

**Interfaces:**
- Consumes: `Service` from Task 1.
- Produces: `requires_pairs(root) -> List[Tuple[str, str]]` (service id → service id); `configured_by_pairs(root) -> List[Tuple[str, str]]` (service id → env var name). Graph edges `requires`/`required_by` (Service→Service), `configured_by`/`configures` (Service→EnvVar).

- [ ] **Step 1: Write the failing test**

```python
# tests/infra/test_edges_compose.py
from tools.graph.traverse import walk


def test_app_requires_backing_services():
    sg = walk("service:app", direction="out", depth=1, level="module")
    reqs = {e.dst for e in sg.edges if e.type == "requires"}
    assert {"service:neo4j", "service:eventstore", "service:redis"} <= reqs


def test_projection_service_configured_by_lane_count():
    sg = walk("service:projection-service", direction="out", depth=1, level="module")
    cfg = {e.dst for e in sg.edges if e.type == "configured_by"}
    assert "env:PROJECTION_LANE_COUNT" in cfg
    assert "env:ESDB_CONNECTION_STRING" in cfg


def test_backing_service_reached_inbound_from_requirer():
    # required_by is discoverable inbound: who needs neo4j?
    sg = walk("service:neo4j", direction="in", depth=1, level="module")
    requirers = {e.src for e in sg.edges if e.type == "requires"}
    assert "service:app" in requirers and "service:projection-service" in requirers
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/infra/test_edges_compose.py -v`
Expected: FAIL — no `requires`/`configured_by` edges yet.

- [ ] **Step 3: Add the pair functions to `tools/infra/reader.py`**

First extend the typing import: `from typing import List, Tuple`.

```python
def requires_pairs(root: str = ".") -> List[Tuple[str, str]]:
    ids = {s.id for s in load_services(root)}
    return [(s.id, dep) for s in load_services(root) for dep in s.requires if dep in ids]


def configured_by_pairs(root: str = ".") -> List[Tuple[str, str]]:
    return [(s.id, var) for s in load_services(root) for var in s.env]
```

- [ ] **Step 4: Register the edge types**

In `tools/graph/registry.py`, add to the `EDGES` list (near the other derived rows):

```python
    EdgeType("requires", "required_by", "Service", "Service", "derived",
             field="requires_edges", resolve="id",
             description="A compose service must be up before this one (compose depends_on)."),
    EdgeType("configured_by", "configures", "Service", "EnvVar", "derived",
             field="configured_by_edges", resolve="id",
             description="A compose service is configured by an inline environment variable."),
```

- [ ] **Step 5: Add the derived builders to `tools/graph/reader.py`**

Add near `_derived_writes` (import infra lazily to preserve layering):

```python
def _derived_requires(edge: EdgeType, root: str) -> List[Edge]:
    from tools.infra.reader import requires_pairs
    return [Edge(edge.name, _addr("Service", a), _addr("Service", b))
            for a, b in requires_pairs(root)]


def _derived_configured_by(edge: EdgeType, root: str) -> List[Edge]:
    from tools.infra.reader import configured_by_pairs
    return [Edge(edge.name, _addr("Service", s), _addr("EnvVar", v))
            for s, v in configured_by_pairs(root)]
```

and add to `_DERIVED`:

```python
    "requires_edges": _derived_requires,
    "configured_by_edges": _derived_configured_by,
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/infra/test_edges_compose.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add tools/infra/reader.py tools/graph/registry.py tools/graph/reader.py tests/infra/test_edges_compose.py
git commit -m "feat(infra): requires (service->service) + configured_by (service->env) edges"
```

---

### Task 3: `runs` edge (Service → CodeUnit entrypoint)

**Files:**
- Modify: `tools/infra/reader.py` (add `runs_pairs` + `_entrypoint_module`), `tools/graph/registry.py` (1 EdgeType), `tools/graph/reader.py` (1 `_DERIVED` builder)
- Test: `tests/infra/test_runs.py`

**Interfaces:**
- Consumes: `Service.command` (Task 1), `tools.code.reader.load_units`.
- Produces: `runs_pairs(root) -> List[Tuple[str, str]]` (service id → code unit id). Graph edge `runs`/`run_by` (Service→CodeUnit). A `command:` naming no resolvable `src.*` module yields no pair (flagged later by `check_infra`).

- [ ] **Step 1: Write the failing test**

```python
# tests/infra/test_runs.py
from tools.infra.reader import runs_pairs, _entrypoint_module
from tools.graph.traverse import walk


def test_entrypoint_resolver_extracts_src_module():
    code_ids = {"main", "run_projection_service", "celery_app"}
    assert _entrypoint_module(["uvicorn", "src.main:app", "--reload"], code_ids) == "main"
    assert _entrypoint_module(["python", "-m", "src.run_projection_service"], code_ids) == "run_projection_service"
    assert _entrypoint_module(["celery", "-A", "src.celery_app", "worker"], code_ids) == "celery_app"
    # a command that names no known src.* module resolves to None (no edge, flagged elsewhere)
    assert _entrypoint_module(["bash", "start.sh"], code_ids) is None


def test_services_run_their_entrypoint_modules():
    pairs = dict(runs_pairs("."))
    assert pairs["app"] == "main"
    assert pairs["projection-service"] == "run_projection_service"
    assert pairs["worker"] == "celery_app"


def test_runs_edge_reaches_code_and_is_inbound_discoverable():
    sg = walk("service:projection-service", direction="out", depth=1, level="module")
    assert any(e.type == "runs" and e.dst == "code:run_projection_service" for e in sg.edges)
    # run_by is inbound-discoverable: which service runs this code?
    back = walk("code:main", direction="in", depth=1, level="module")
    assert any(e.type == "runs" and e.src == "service:app" for e in back.edges)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/infra/test_runs.py -v`
Expected: FAIL — `runs_pairs`/`_entrypoint_module` not defined.

- [ ] **Step 3: Add the resolver + pair function to `tools/infra/reader.py`**

First extend the imports: `from typing import List, Optional, Set, Tuple` and add `from tools.code.reader import load_units`.

```python
def _entrypoint_module(command: List[str], code_ids: Set[str]) -> Optional[str]:
    for tok in command:
        m = _SRC_TOKEN.match(str(tok))
        if m and m.group(1) in code_ids:
            return m.group(1)
    return None


def runs_pairs(root: str = ".") -> List[Tuple[str, str]]:
    code_ids = {u.unit for u in load_units(root)}
    out: List[Tuple[str, str]] = []
    for s in load_services(root):
        mod = _entrypoint_module(s.command, code_ids)
        if mod:
            out.append((s.id, mod))
    return out
```

- [ ] **Step 4: Register the edge type**

In `tools/graph/registry.py` `EDGES`:

```python
    EdgeType("runs", "run_by", "Service", "CodeUnit", "derived",
             field="runs_edges", resolve="id",
             description="A code service launches a code module (its compose command entrypoint)."),
```

- [ ] **Step 5: Add the derived builder to `tools/graph/reader.py`**

```python
def _derived_runs(edge: EdgeType, root: str) -> List[Edge]:
    from tools.infra.reader import runs_pairs
    return [Edge(edge.name, _addr("Service", s), _addr("CodeUnit", mod))
            for s, mod in runs_pairs(root)]
```

and to `_DERIVED`: `"runs_edges": _derived_runs,`

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/infra/test_runs.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add tools/infra/reader.py tools/graph/registry.py tools/graph/reader.py tests/infra/test_runs.py
git commit -m "feat(infra): runs edge (service -> code entrypoint) from compose command"
```

---

### Task 4: `talks_to` edge (CodeUnit → Service) — import + marker

**Files:**
- Modify: `tools/infra/reader.py` (add `talks_to_pairs` + `_module_imports`), `tools/graph/registry.py` (1 EdgeType), `tools/graph/reader.py` (1 `_DERIVED` builder)
- Test: `tests/infra/test_talks_to.py`

**Interfaces:**
- Consumes: `tools.code.reader.load_units` (module `.path`), `_CLIENT_LIBS`, `_TALKS_MARKER` (Task 1).
- Produces: `talks_to_pairs(root) -> List[Tuple[str, str]]` (code unit id → service id). Graph edge `talks_to`/`talked_to_by` (CodeUnit→Service). Derived from a module's client-lib import; `# talks-to: <service>` marker adds an explicit edge.

- [ ] **Step 1: Write the failing test**

```python
# tests/infra/test_talks_to.py
import os
from tools.infra.reader import talks_to_pairs
from tools.graph.traverse import walk


def test_client_lib_imports_derive_talks_to():
    pairs = set(talks_to_pairs("."))
    assert ("utils.neo4j_driver", "neo4j") in pairs       # imports neo4j driver
    assert ("events.store", "eventstore") in pairs         # imports esdbclient
    assert ("celery_app", "redis") in pairs                # imports celery


def test_marker_adds_talks_to(tmp_path):
    # a synthetic module with only a `# talks-to:` marker (no client-lib import) still links
    os.makedirs(tmp_path / "src" / "svc", exist_ok=True)
    open(tmp_path / "docker-compose.yml", "w").write(
        "services:\n  neo4j:\n    image: neo4j:5\n")
    open(tmp_path / "src" / "svc" / "__init__.py", "w").close()
    open(tmp_path / "src" / "svc" / "m.py", "w").write("# talks-to: neo4j\ndef f():\n    pass\n")
    assert ("svc.m", "neo4j") in set(talks_to_pairs(str(tmp_path)))


def test_schema_topology_from_a_backing_service():
    # from a backing service, walk inbound: who runs toward it (requires) AND who talks to it
    sg = walk("service:neo4j", direction="in", depth=1, level="module")
    talkers = {e.src for e in sg.edges if e.type == "talks_to"}
    assert "code:utils.neo4j_driver" in talkers
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/infra/test_talks_to.py -v`
Expected: FAIL — `talks_to_pairs` not defined.

- [ ] **Step 3: Add the derivation to `tools/infra/reader.py`**

First add `import ast` (at the top, alphabetically before `os`). `Set`/`Tuple`/`load_units` are already imported from Task 3.

```python
def _module_imports(src: str) -> Set[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    libs: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                libs.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            libs.add(n.module.split(".")[0])
    return libs


def talks_to_pairs(root: str = ".") -> List[Tuple[str, str]]:
    service_ids = {s.id for s in load_services(root)}
    out: List[Tuple[str, str]] = []
    for u in load_units(root):
        if getattr(u, "level", "") != "module" or not str(u.path).endswith(".py"):
            continue
        try:
            src = open(u.path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        targets: Set[str] = set()
        for lib in _module_imports(src):                       # derived: client-lib import
            svc = _CLIENT_LIBS.get(lib)
            if svc in service_ids:
                targets.add(svc)
        for svc in _TALKS_MARKER.findall(src):                 # marker fallback
            if svc in service_ids:
                targets.add(svc)
        out += [(u.unit, svc) for svc in sorted(targets)]
    return out
```

- [ ] **Step 4: Register the edge type**

In `tools/graph/registry.py` `EDGES`:

```python
    EdgeType("talks_to", "talked_to_by", "CodeUnit", "Service", "derived",
             field="talks_to_edges", resolve="id",
             description="A code module connects to a backing service (client-lib import or # talks-to: marker)."),
```

- [ ] **Step 5: Add the derived builder to `tools/graph/reader.py`**

```python
def _derived_talks_to(edge: EdgeType, root: str) -> List[Edge]:
    from tools.infra.reader import talks_to_pairs
    return [Edge(edge.name, _addr("CodeUnit", u), _addr("Service", svc))
            for u, svc in talks_to_pairs(root)]
```

and to `_DERIVED`: `"talks_to_edges": _derived_talks_to,`

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/infra/test_talks_to.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Bridge the code-centric gold (markers only where needed)**

Verify whether the two deployment scenarios' gold nodes reach a service. Run:
`PYTHONPATH=. python -m tools.graph walk service:neo4j --dir in --depth 2 --level module`
and confirm `code:persistence` and `code:projections.subscription_manager` are reachable (they should be, via `talks_to code:utils.neo4j_driver`/`events.store` then existing `depends_on` code edges). ONLY if a gold node is genuinely unreachable, add a single `# talks-to: <service>` marker comment to that module's top (e.g. `src/persistence/__init__.py`) and re-run — never add a marker speculatively. If markers were added, re-run `tests/infra/test_talks_to.py` and note the additions in the commit body.

- [ ] **Step 8: Commit**

```bash
git add tools/infra/reader.py tools/graph/registry.py tools/graph/reader.py tests/infra/test_talks_to.py
git commit -m "feat(infra): talks_to edge (code -> backing service) from client-lib import + marker"
```

---

### Task 5: Infra domain — render, check, CLI, Makefile, catalog

**Files:**
- Create: `tools/infra/render.py`, `tools/infra/check.py`, `tools/infra/__main__.py`, `docs/infra/index.md` (generated)
- Modify: `Makefile` (`infra-index`, `infra-check`, add `infra-index` to `regen-derived`)
- Test: `tests/infra/test_check.py`, `tests/infra/test_render.py`

**Interfaces:**
- Consumes: `load_services`, `load_env_vars`, and the four pair functions (Tasks 1–4).
- Produces: `tools.infra.check.check_infra(root=".") -> List[Finding]` (dataclass with `.message`), `tools.infra.check.run_all(root=".") -> List[Finding]`; `tools.infra.render.render_index(...) -> str`; CLI `python -m tools.infra index|check|list`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/infra/test_check.py
import os
from tools.infra.check import check_infra


def test_real_repo_infra_clean():
    # every code-service command resolves to a code node; every talks-to marker names a real service
    assert check_infra(".") == []


def test_unresolvable_command_is_flagged(tmp_path):
    open(tmp_path / "docker-compose.yml", "w").write(
        "services:\n"
        "  app:\n"
        "    build: .\n"
        "    command: [\"bash\", \"start.sh\"]\n")   # code service, no resolvable src.* module
    msgs = [f.message for f in check_infra(str(tmp_path))]
    assert any("app" in m and "command" in m for m in msgs)
```

```python
# tests/infra/test_render.py
from tools.infra.render import render_index
from tools.infra.reader import load_services, load_env_vars


def test_index_lists_services_by_kind():
    out = render_index(load_services("."), load_env_vars("."),
                       runs=[], talks_to=[], requires=[], configured_by=[])
    assert "app" in out and "neo4j" in out
    assert "code" in out and "backing" in out
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/infra/test_check.py tests/infra/test_render.py -v`
Expected: FAIL — modules not defined.

- [ ] **Step 3: Write `tools/infra/check.py`**

```python
# tools/infra/check.py
"""Non-blocking drift checks for the infra overlay: a code-service whose compose command resolves
to no code module (so `runs` silently drops it), and a `# talks-to:` marker naming an unknown
service. Returns List[Finding]; the CLI returns 0."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from tools.code.reader import load_units
from tools.infra.reader import _TALKS_MARKER, _entrypoint_module, load_services


@dataclass
class Finding:
    message: str


def check_infra(root: str = ".") -> List[Finding]:
    findings: List[Finding] = []
    code_ids = {u.unit for u in load_units(root)}
    for s in load_services(root):
        if s.kind == "code" and s.command and _entrypoint_module(s.command, code_ids) is None:
            findings.append(Finding(
                f"infra: service {s.id} command {s.command!r} resolves to no code module "
                f"— `runs` will drop it (use exec-form src.* or a marker)"))
    service_ids = {s.id for s in load_services(root)}
    for u in load_units(root):
        if getattr(u, "level", "") != "module" or not str(u.path).endswith(".py"):
            continue
        try:
            src = open(u.path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for svc in _TALKS_MARKER.findall(src):
            if svc not in service_ids:
                findings.append(Finding(
                    f"infra: {u.unit} has `# talks-to: {svc}` but no compose service named {svc}"))
    return findings


def run_all(root: str = ".") -> List[Finding]:
    return check_infra(root)
```

Note: `check.py` imports `load_units` directly from `tools.code.reader` (not re-exported through `infra.reader`), and only the three names it actually uses from `infra.reader` — keeps the import list flake8-clean.

- [ ] **Step 4: Write `tools/infra/render.py`**

```python
# tools/infra/render.py
"""Renders docs/infra/index.md — the deployment topology catalog: services by kind, the requires
DAG, each code service's entrypoint + talks_to, and the inline env vars. Pure function of the
compose-derived data (freshness-gated)."""
from __future__ import annotations

from typing import List, Tuple

from tools.infra.reader import EnvVar, Service


def render_index(services: List[Service], env_vars: List[EnvVar],
                 runs: List[Tuple[str, str]], talks_to: List[Tuple[str, str]],
                 requires: List[Tuple[str, str]], configured_by: List[Tuple[str, str]]) -> str:
    lines = ["# Infra / deployment topology",
             "",
             "> Generated from `docker-compose.yml` by `make infra-index`. Do not edit by hand.",
             "",
             "## Services", ""]
    for s in sorted(services, key=lambda x: (x.kind, x.id)):
        detail = f"image `{s.image}`" if s.kind == "backing" else f"runs `{' '.join(s.command)}`"
        reqs = ", ".join(sorted(d for a, d in requires if a == s.id)) or "—"
        lines.append(f"- **{s.id}** ({s.kind}) — {detail}; requires: {reqs}")
    lines += ["", "## Code → backing service (`talks_to`)", ""]
    for code, svc in sorted(talks_to):
        lines.append(f"- `{code}` → **{svc}**")
    lines += ["", "## Environment variables (inline; `.env` excluded)", ""]
    for v in sorted(env_vars, key=lambda x: x.name):
        svcs = ", ".join(sorted(s for s, name in configured_by if name == v.name))
        lines.append(f"- `{v.name}` — {svcs}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 5: Write `tools/infra/__main__.py`**

```python
# tools/infra/__main__.py
"""CLI for the tools.infra domain: `index` renders docs/infra/index.md; `check` runs the
non-blocking findings; `list` prints the services."""
from __future__ import annotations

import argparse
import os
import sys

from tools.infra.check import run_all
from tools.infra.reader import (
    configured_by_pairs, load_env_vars, load_services, requires_pairs, runs_pairs, talks_to_pairs,
)
from tools.infra.render import render_index

INFRA_DIR = "docs/infra"
INDEX = f"{INFRA_DIR}/index.md"


def cmd_index(args) -> int:
    os.makedirs(INFRA_DIR, exist_ok=True)
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_index(load_services(), load_env_vars(), runs_pairs(),
                              talks_to_pairs(), requires_pairs(), configured_by_pairs()))
    print(f"wrote {INDEX}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"infra-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("infra-check: clean")
    return 0  # NON-BLOCKING


def cmd_list(args) -> int:
    for s in load_services():
        print(f"{s.kind:8} {s.id}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.infra")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for c in ("index", "check", "list"):
        sub.add_parser(c)
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Wire the Makefile**

Add targets (mirror `code-index`/`code-check`), and add `infra-index` to the `regen-derived` list:

```makefile
infra-index: ## Regenerate docs/infra/index.md (deployment topology)
	@$(PYTHON) -m tools.infra index

infra-check: ## Reconcile the infra overlay vs docker-compose (non-blocking)
	@$(PYTHON) -m tools.infra check
```

In `regen-derived`, append `infra-index` to the `$(MAKE) ...` list.

- [ ] **Step 7: Generate the catalog + run tests**

Run: `PYTHONPATH=. python -m tools.infra index` (creates `docs/infra/index.md`)
Run: `PYTHONPATH=. python -m pytest tests/infra/ -v`
Expected: all infra tests PASS; `PYTHONPATH=. python -m tools.infra check` prints `infra-check: clean`.

- [ ] **Step 8: Freshness + commit**

Run: `make regen-derived && git diff --exit-code` — regenerate until clean, staging any drifted `docs/graph/{index,graph}.md`, `docs/cli/index.md`, `docs/code/index.md`, `docs/tests/index.md`, `docs/infra/index.md`.

```bash
git add tools/infra/ docs/infra/index.md Makefile tests/infra/test_check.py tests/infra/test_render.py docs/graph docs/cli docs/code docs/tests
git commit -m "feat(infra): domain render/check/CLI + docs/infra catalog + make targets"
```

---

### Task 6: Close-out — end-to-end walk, eval re-run, ADR-0029, review

**Files:**
- Create: `tests/infra/test_topology_walk.py`, `docs/adr/0029-*.md`
- Modify: `evals/graph/RESULTS.md`, `docs/superpowers/kg-program-roadmap.md`

**Interfaces:** Consumes the full overlay (Tasks 1–5). No new production code (markers only if Task 4 Step 7 required them).

- [ ] **Step 1: Write the end-to-end topology test (the two eval questions, as graph walks)**

```python
# tests/infra/test_topology_walk.py
from tools.graph.traverse import walk


def test_backing_services_for_api_reads():
    # "which backing services must be up for the API?" — app runs code:main and requires the stores
    app = walk("service:app", direction="both", depth=2, level="module")
    kinds = {e.dst for e in app.edges if e.type == "requires"}
    assert {"service:neo4j", "service:eventstore"} <= kinds
    assert any(e.type == "runs" and e.dst == "code:main" for e in app.edges)


def test_projection_service_needs():
    # "what does projection-service need?" — deps + entrypoint + config all reachable in one walk
    sg = walk("service:projection-service", direction="out", depth=2, level="module")
    assert any(e.type == "requires" and e.dst == "service:eventstore" for e in sg.edges)
    assert any(e.type == "runs" and e.dst == "code:run_projection_service" for e in sg.edges)
    assert any(e.type == "configured_by" and e.dst == "env:PROJECTION_LANE_COUNT" for e in sg.edges)
```

Run: `PYTHONPATH=. python -m pytest tests/infra/test_topology_walk.py -v` — expected PASS.

- [ ] **Step 2: Full gate**

Run: `make regen-derived && git diff --exit-code` (clean); `make test-unit` (green); `PYTHONPATH=. python -m tools.graph check` (no dangling infra endpoints — `check_endpoints` validates every harvested edge, including the four new ones); `PYTHONPATH=. python -m tools.graph walk service:app --dir both --depth 2` (eyeball the topology).

- [ ] **Step 3: Eval re-run (the measurement)**

Re-run the two deployment scenarios agentically (Mode-B autonomous subagent driving the graph CLI + judge, exactly like the KG-2 re-run): `deploy-service-topology`, `deploy-projection-service`. Record the new verdicts + the lift in `evals/graph/RESULTS.md` under a "KG-3 re-run" section. These were `expected: gap` on Layer 1; note that they are now traversable (the Layer-1 `expected` labels are re-scored separately by `make eval-graph`).

- [ ] **Step 4: ADR-0029**

`python -m tools.adr new "Infra and deployment overlay is derived from docker-compose"`, fill it in: decision = two derived node types (`Service` with a `kind` axis, `EnvVar` inline-only) + four derived edges (`requires`/`runs`/`talks_to`/`configured_by`) from `docker-compose.yml` + a client-lib map + `# talks-to:` marker. Extends ADR-0020; consistent with 0025/0026/0028. Record the ceilings (single compose file; command must name `src.*`; talks_to import ceiling; EnvVar excludes `.env`; requires reflects declarations not runtime). Set `source:` to this spec, add the reciprocal `# governed-by: ADR-0029` marker on `tools/infra/reader.py` (already in the Task 1 file header), then `make adr-index`.

- [ ] **Step 5: Roadmap + freshness + commit**

Update `docs/superpowers/kg-program-roadmap.md`: mark KG-3 done, note the eval lift. Run `make regen-derived && git diff --exit-code` (fold drifted catalogs). Commit:

```bash
git add tests/infra/test_topology_walk.py docs/adr evals/graph/RESULTS.md docs/superpowers/kg-program-roadmap.md docs/graph docs/code docs/tests docs/cli tools/infra/reader.py
git commit -m "docs(kg3): ADR-0029 + eval re-run — deployment topology is traversable"
```

- [ ] **Step 6: Final whole-branch review**

Dispatch the final whole-branch review (most capable model) with a review package (`scripts/review-package $(git merge-base main HEAD) HEAD`). Focus: correctness of the four derivations (no false edges — especially `talks_to` and any markers added; `requires` not double-listed in the catalog), the import-layering constraint (infra.reader must not import graph.reader), check non-blocking, tests real, secret-safety (no `.env` var leaked into `EnvVar`), ADR/RESULTS honesty. Fix Critical/Important findings, then finish via `superpowers:finishing-a-development-branch`.

---

## Self-Review notes

- **Spec coverage:** node types (Task 1), all four edges (Tasks 2–4), the `kind` axis + secret-safe `EnvVar` (Task 1), the `tools/infra/` domain + catalog + check + Makefile + freshness (Task 5), the eval re-run + ADR-0029 + review (Task 6). All spec §Deliverables map to a task.
- **Type consistency:** `Service.id`/`EnvVar.name` are the node ids everywhere; pair functions return `List[Tuple[str, str]]`; `_derived_*` wrap with `_addr("Service"/"EnvVar"/"CodeUnit", …)`; edge verbs `requires`/`runs`/`talks_to`/`configured_by` used identically in registry rows, builders, and tests.
- **Import layering** enforced in every task that touches `tools/infra/reader.py` (no `tools.graph` import) and every `_derived_*` (lazy infra import).
