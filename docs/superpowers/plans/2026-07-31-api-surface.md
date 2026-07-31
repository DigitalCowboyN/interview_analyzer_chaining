# API-Surface Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Catalog the HTTP API from the live FastAPI app and guard the committed `frontend/openapi.json` against drift, plus catalog-sync and docs-reference checks — all non-blocking.

**Architecture:** New `tools/api/` package (reader → render → check → CLI). The reader **imports `src.main:app`** (setting placeholder API keys so it loads headlessly) and reads the live routes + `app.openapi()`. Runs via `make api-check` in the project env — not a git hook.

**Tech Stack:** Python 3 (stdlib + FastAPI app import), pytest, Make.

## Global Constraints

- **Non-blocking, always.** Checks return `list[Finding]`; none raises; `make api-check` / the CLI exit 0. If the app import fails, `run_all` returns one Finding and still exits 0.
- **Definition = live app.** Catalog from `app.routes` (APIRoutes whose endpoint module starts with `src`). Freshness compares committed `frontend/openapi.json` ↔ live `app.openapi()` (like-for-like OpenAPI).
- **Endpoint-set granularity** — `(method, path)` pairs; not field-level schema.
- **`load_app()` sets placeholder `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`** (only via `setdefault`) so the import works without real keys, and disables logging.
- `Endpoint` / `Finding` local to `tools/api`. Tooling in `tools/api/`; tests in `tests/api_surface/` (avoid clashing with the existing `tests/api/` router tests).
- Run tests with `~/.pyenv/shims/python -m pytest <path> -v`.

---

### Task 1: `reader.py` — app import + route/openapi extraction

**Files:**
- Create: `tools/api/__init__.py` (empty), `tools/api/reader.py`
- Test: `tests/api_surface/__init__.py` (empty), `tests/api_surface/test_reader.py`

**Interfaces:**
- Produces:
  - `@dataclass Endpoint(method: str, path: str, router: str, summary: str)`
  - `load_app()` — sets placeholder keys, imports and returns `src.main:app`
  - `live_endpoints(app) -> list[Endpoint]`
  - `live_openapi_pairs(app) -> set[tuple[str,str]]`
  - `committed_pairs(openapi_path) -> set[tuple[str,str]] | None` (None if file missing)

- [ ] **Step 1: Write the failing test**

```python
# tests/api_surface/test_reader.py
import json
import pytest
from tools.api.reader import Endpoint, committed_pairs, load_app, live_endpoints

def test_committed_pairs_parses_openapi(tmp_path):
    f = tmp_path / "openapi.json"
    f.write_text(json.dumps({"paths": {"/x": {"get": {}, "post": {}}, "/y": {"get": {}}}}), encoding="utf-8")
    assert committed_pairs(str(f)) == {("GET", "/x"), ("POST", "/x"), ("GET", "/y")}

def test_committed_pairs_missing_is_none(tmp_path):
    assert committed_pairs(str(tmp_path / "nope.json")) is None

@pytest.mark.integration
def test_live_endpoints_against_real_app():
    try:
        app = load_app()
    except Exception as e:
        pytest.skip(f"app import unavailable: {e}")
    eps = live_endpoints(app)
    assert len(eps) > 20
    assert all(isinstance(e, Endpoint) for e in eps)
    assert any(e.path == "/exports/{interview_id}/{lens_name}" and e.method == "GET" for e in eps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/api_surface/test_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.api'`

- [ ] **Step 3: Implement**

```python
# tools/api/reader.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")


@dataclass
class Endpoint:
    method: str
    path: str
    router: str
    summary: str


def load_app():
    import logging
    os.environ.setdefault("ANTHROPIC_API_KEY", "api-surface-placeholder")
    os.environ.setdefault("OPENAI_API_KEY", "api-surface-placeholder")
    logging.disable(logging.CRITICAL)
    from src.main import app
    return app


def live_endpoints(app) -> List[Endpoint]:
    from fastapi.routing import APIRoute
    eps: List[Endpoint] = []
    for r in app.routes:
        if isinstance(r, APIRoute) and r.endpoint.__module__.startswith("src"):
            for m in sorted(r.methods):
                if m in _METHODS:
                    eps.append(Endpoint(m, r.path, r.endpoint.__module__, r.summary or r.name or ""))
    return sorted(eps, key=lambda e: (e.router, e.path, e.method))


def live_openapi_pairs(app) -> Set[Tuple[str, str]]:
    return {(m.upper(), p) for p, ops in app.openapi()["paths"].items() for m in ops}


def committed_pairs(openapi_path: str) -> Optional[Set[Tuple[str, str]]]:
    if not os.path.exists(openapi_path):
        return None
    data = json.load(open(openapi_path, encoding="utf-8"))
    return {(m.upper(), p) for p, ops in data.get("paths", {}).items() for m in ops}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/api_surface/test_reader.py -v`
Expected: PASS (the 2 unit tests; the integration test either passes or skips)

- [ ] **Step 5: Commit**

```bash
git add tools/api/__init__.py tools/api/reader.py tests/api_surface/__init__.py tests/api_surface/test_reader.py
git commit -m "feat(api): live-app route + openapi reader"
```

---

### Task 2: `render.py` — catalog rendering

**Files:**
- Create: `tools/api/render.py`
- Test: `tests/api_surface/test_render.py`

**Interfaces:**
- Consumes: `tools.api.reader.Endpoint`
- Produces: `render_catalog(endpoints) -> str` (grouped by router)

- [ ] **Step 1: Write the failing test**

```python
# tests/api_surface/test_render.py
from tools.api.reader import Endpoint
from tools.api.render import render_catalog

EPS = [
    Endpoint("GET", "/exports/{interview_id}/{lens_name}", "src.api.routers.exports", "Export bundle"),
    Endpoint("GET", "/files/", "src.api.routers.files", "List files"),
    Endpoint("POST", "/files/", "src.api.routers.files", "Ingest a transcript"),
]

def test_render_catalog_groups_by_router():
    out = render_catalog(EPS)
    assert "## src.api.routers.exports" in out
    assert "`GET /exports/{interview_id}/{lens_name}` — Export bundle" in out
    assert "## src.api.routers.files" in out
    assert "`POST /files/` — Ingest a transcript" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/api_surface/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.api.render'`

- [ ] **Step 3: Implement**

```python
# tools/api/render.py
from __future__ import annotations

from typing import List

from tools.api.reader import Endpoint


def render_catalog(endpoints: List[Endpoint]) -> str:
    by_router: dict = {}
    for e in endpoints:
        by_router.setdefault(e.router, []).append(e)
    lines = ["# API surface", ""]
    for router in sorted(by_router):
        lines.append(f"## {router}")
        lines.append("")
        for e in sorted(by_router[router], key=lambda e: (e.path, e.method)):
            summ = f" — {e.summary}" if e.summary else ""
            lines.append(f"- `{e.method} {e.path}`{summ}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/api_surface/test_render.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/api/render.py tests/api_surface/test_render.py
git commit -m "feat(api): catalog renderer (grouped by router)"
```

---

### Task 3: `check.py` — the reconciliation guard

**Files:**
- Create: `tools/api/check.py`
- Test: `tests/api_surface/test_check.py`

**Interfaces:**
- Consumes: `tools.api.reader` (`Endpoint`, `load_app`, `live_endpoints`, `live_openapi_pairs`, `committed_pairs`), `tools.api.render.render_catalog`
- Produces: `@dataclass Finding`, `check_openapi_fresh(committed, live, openapi_path=...)`, `check_docs_reference_real(endpoints, doc_paths)`, `check_catalog_in_sync(catalog_path, endpoints)`, `run_all(root=".") -> list[Finding]`

- [ ] **Step 1: Write the failing test**

```python
# tests/api_surface/test_check.py
from tools.api.reader import Endpoint
from tools.api.check import (
    check_openapi_fresh, check_docs_reference_real, check_catalog_in_sync, Finding,
)

EPS = [Endpoint("GET", "/exports/{interview_id}/{lens_name}", "src.api.routers.exports", "Export")]

def test_openapi_fresh_flags_added_and_missing():
    live = {("GET", "/a"), ("GET", "/b")}
    committed = {("GET", "/a")}
    msgs = " ".join(f.message for f in check_openapi_fresh(committed, live))
    assert "GET /b exists in the app but not in" in msgs   # stale: app has it, contract doesn't
    # in-sync -> no findings
    assert check_openapi_fresh(live, live) == []

def test_openapi_fresh_missing_file():
    msgs = " ".join(f.message for f in check_openapi_fresh(None, {("GET", "/a")}))
    assert "missing" in msgs

def test_docs_reference_real_normalizes_path_params(tmp_path):
    doc = tmp_path / "CLAUDE.md"
    doc.write_text("Use `GET /exports/{id}/{lens}` and `POST /gone`. Prose /exports/x ignored.\n", encoding="utf-8")
    msgs = " ".join(f.message for f in check_docs_reference_real(EPS, [str(doc)]))
    assert "POST /gone" in msgs                       # not a real endpoint
    assert "/exports/{id}/{lens}" not in msgs          # param-name difference tolerated -> real

def test_catalog_in_sync(tmp_path):
    from tools.api.render import render_catalog
    cat = tmp_path / "index.md"
    cat.write_text("stale\n", encoding="utf-8")
    assert check_catalog_in_sync(str(cat), EPS)
    cat.write_text(render_catalog(EPS), encoding="utf-8")
    assert check_catalog_in_sync(str(cat), EPS) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/api_surface/test_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.api.check'`

- [ ] **Step 3: Implement**

```python
# tools/api/check.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from tools.api.reader import (
    Endpoint, committed_pairs, live_endpoints, live_openapi_pairs, load_app,
)
from tools.api.render import render_catalog

OPENAPI = "frontend/openapi.json"
_ENDPOINT_MENTION = re.compile(r"`(GET|POST|PUT|DELETE|PATCH)\s+(/[\w{}./-]*)`")


@dataclass
class Finding:
    message: str


def _shape(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def check_openapi_fresh(committed: Optional[Set[Tuple[str, str]]],
                        live: Set[Tuple[str, str]],
                        openapi_path: str = OPENAPI) -> List[Finding]:
    if committed is None:
        return [Finding(f"{openapi_path} is missing — run make ui-typegen")]
    findings: List[Finding] = []
    for m, p in sorted(live - committed):
        findings.append(Finding(f"{m} {p} exists in the app but not in {openapi_path} — run make ui-typegen"))
    for m, p in sorted(committed - live):
        findings.append(Finding(f"{m} {p} is in {openapi_path} but not in the app — run make ui-typegen"))
    return findings


def check_docs_reference_real(endpoints: List[Endpoint], doc_paths: List[str]) -> List[Finding]:
    real = {(e.method, _shape(e.path)) for e in endpoints}
    findings: List[Finding] = []
    for dp in doc_paths:
        if not os.path.exists(dp):
            continue
        text = open(dp, encoding="utf-8").read()
        base = os.path.basename(dp)
        for mo in _ENDPOINT_MENTION.finditer(text):
            method, path = mo.group(1), mo.group(2)
            if (method, _shape(path)) not in real:
                findings.append(Finding(f"{base} references `{method} {path}` which is not a real endpoint"))
    return findings


def check_catalog_in_sync(catalog_path: str, endpoints: List[Endpoint]) -> List[Finding]:
    want = render_catalog(endpoints)
    have = open(catalog_path, encoding="utf-8").read() if os.path.exists(catalog_path) else ""
    if want != have:
        return [Finding("docs/api/index.md out of sync — run make api-index")]
    return []


def run_all(root: str = ".") -> List[Finding]:
    try:
        app = load_app()
    except Exception as e:  # non-blocking: degrade to a warning
        return [Finding(f"could not import src.main:app ({e}) — api-check skipped")]
    endpoints = live_endpoints(app)
    findings: List[Finding] = []
    findings += check_openapi_fresh(committed_pairs(os.path.join(root, OPENAPI)), live_openapi_pairs(app),
                                    os.path.join(root, OPENAPI))
    findings += check_catalog_in_sync(os.path.join(root, "docs/api/index.md"), endpoints)
    findings += check_docs_reference_real(endpoints, [os.path.join(root, d) for d in ("CLAUDE.md", "README.md")])
    return findings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/api_surface/test_check.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/api/check.py tests/api_surface/test_check.py
git commit -m "feat(api): openapi-freshness, catalog-sync, docs-reference guard checks"
```

---

### Task 4: CLI + Makefile targets

**Files:**
- Create: `tools/api/__main__.py`
- Modify: `Makefile` (add `api-index`, `api-check`, each with a `##` doc)
- Test: `tests/api_surface/test_cli.py`

**Interfaces:**
- Produces: `python -m tools.api {index|check}` (both exit 0)

- [ ] **Step 1: Write the failing test**

```python
# tests/api_surface/test_cli.py
import subprocess
import sys

def test_cli_check_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.api", "check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "api-check" in proc.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/api_surface/test_cli.py -v`
Expected: FAIL — `No module named tools.api.__main__`

- [ ] **Step 3: Implement**

```python
# tools/api/__main__.py
from __future__ import annotations

import argparse
import os
import sys

from tools.api.check import run_all
from tools.api.reader import live_endpoints, load_app
from tools.api.render import render_catalog

CATALOG = "docs/api/index.md"


def cmd_index(args) -> int:
    app = load_app()
    os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
    with open(CATALOG, "w", encoding="utf-8") as fh:
        fh.write(render_catalog(live_endpoints(app)))
    print(f"wrote {CATALOG}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"api-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("api-check: clean")
    return 0  # NON-BLOCKING: always 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.api")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

Add to `Makefile` (near `cli-check`), each self-documented:

```makefile
.PHONY: api-index
api-index: ## Regenerate docs/api/index.md (the API catalog)
	@$(PYTHON) -m tools.api index

.PHONY: api-check
api-check: ## Reconcile the API surface + openapi.json freshness (non-blocking)
	@$(PYTHON) -m tools.api check
```

- [ ] **Step 4: Run tests + smoke**

Run: `~/.pyenv/shims/python -m pytest tests/api_surface/test_cli.py -v`
Expected: PASS
Run: `~/.pyenv/shims/python -m tools.api check`
Expected: exit 0; prints the openapi-fresh finding for `GET /ui/streams/events` (real pre-existing drift) + a catalog-out-of-sync finding (until Task 5)

- [ ] **Step 5: Commit**

```bash
git add tools/api/__main__.py Makefile tests/api_surface/test_cli.py
git commit -m "feat(api): CLI (index/check) + make targets"
```

---

### Task 5: Generate the catalog + smoke

**Files:**
- Create (generated): `docs/api/index.md`

- [ ] **Step 1: Generate the catalog**

```bash
make api-index          # writes docs/api/index.md from the live app (35 endpoints)
```

- [ ] **Step 2: Reconcile**

```bash
make api-check
```
Expected result: the catalog-sync finding is gone; **one openapi-fresh finding remains** —
`GET /ui/streams/events exists in the app but not in frontend/openapi.json`. This is a
**real, pre-existing drift** the tool correctly caught (the SSE endpoint was added
without re-running `ui-typegen`). Leave it as a reported finding — the fix is the
owner running `make ui-typegen` (a frontend-toolchain action, out of this domain's
scope). Note it in the commit message / PR so it isn't mistaken for a tool bug.

- [ ] **Step 3: Commit**

```bash
git add docs/api/index.md
git commit -m "docs(api): generated API catalog (api-check surfaces one real openapi.json drift: GET /ui/streams/events)"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/api_surface/ -v` — unit tests green (the integration reader test passes or skips).
- [ ] `make api-index` then `git status` — `docs/api/index.md` regenerates identically (in sync).
- [ ] `make api-check` — exits 0; the only finding is the known `GET /ui/streams/events` openapi drift (or clean, if `make ui-typegen` was run to fix it).
- [ ] Confirm `docs/api/index.md` groups all 35 endpoints under their routers.
