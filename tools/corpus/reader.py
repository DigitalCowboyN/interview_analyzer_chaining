"""Type-primary, repo-wide discovery of OKF documents: walks every `.md` file (skipping
vendored/build/VCS directories), reads each file's own top-of-file `type:` frontmatter,
and returns a `Record` for every one whose type is registered in `OKF_HOMES` — the corpus
substrate's single intake point, consumed by `tools.corpus.check` and `__main__`."""

from __future__ import annotations

import os
from typing import Iterable, List

from src.ingestion.front_matter import parse_front_matter
from tools.corpus.model import OKF_HOMES, Record

# Directories never scanned for records (vendored, build, VCS, caches, worktrees).
_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".worktrees", "htmlcov",
                ".pytest_cache", ".mypy_cache", "venv", ".venv", "build", "dist", ".next"}


def _iter_markdown(root: str, ignore) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore]   # prune in place
        for fn in filenames:
            if fn.endswith(".md"):
                yield os.path.join(dirpath, fn)


def okf_records(root: str = ".", ignore=_IGNORE_DIRS) -> List[Record]:
    """Every OKF document in the repo, discovered by its OWN top-of-file `type:` frontmatter
    (never a body match) and classified by type. This is the type-primary, repo-wide intake:
    a record is found by what it IS, anywhere; its home folder is only used later to judge
    whether it is misfiled."""
    out: List[Record] = []
    for path in sorted(_iter_markdown(root, ignore)):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        fm, offset = parse_front_matter(text)
        if not fm or fm.get("type") not in OKF_HOMES:
            continue
        out.append(Record(
            type=fm["type"],
            id=os.path.splitext(os.path.basename(path))[0],
            path=os.path.relpath(path, root),
            frontmatter=fm,
            body=text[offset:],
        ))
    return out
