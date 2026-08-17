"""Scans `.py` and `README.md` files under given subdirs for `governed-by: ADR-####` markers
and extracts the referenced ADR ids, giving `tools.adr.check` the code-side half of the
governs/governed-by agreement check."""
from __future__ import annotations

import os
import re
from typing import Dict, List

_MARKER = re.compile(r"governed-by:\s*(.+)", re.IGNORECASE)
_ADR_ID = re.compile(r"ADR[-\s]?(\d{1,4})", re.IGNORECASE)


def extract_refs(text: str) -> List[str]:
    refs: List[str] = []
    for m in _MARKER.finditer(text):
        refs += [tok for tok in re.split(r"[,\s]+", m.group(1).strip()) if tok]
    return refs


def adr_ids_from_refs(refs: List[str]) -> List[int]:
    ids: List[int] = []
    for ref in refs:
        m = _ADR_ID.search(ref)
        if m:
            ids.append(int(m.group(1)))
    return ids


def scan_markers(root: str, subdirs=("src",)) -> Dict[str, List[str]]:
    """Map repo-relative path -> raw governed-by ref tokens. File markers key by
    the file path; __init__.py / README.md markers key by the directory (trailing /)."""
    result: Dict[str, List[str]] = {}
    for base in subdirs:
        start = os.path.join(root, base)
        if not os.path.isdir(start):
            continue
        for dirpath, _dirs, filenames in os.walk(start):
            for fn in filenames:
                if not (fn.endswith(".py") or fn == "README.md"):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    text = open(full, encoding="utf-8").read()
                except Exception:
                    continue
                refs = extract_refs(text)
                if not refs:
                    continue
                if fn == "__init__.py" or fn == "README.md":
                    key = os.path.relpath(dirpath, root).replace(os.sep, "/") + "/"
                else:
                    key = os.path.relpath(full, root).replace(os.sep, "/")
                result.setdefault(key, []).extend(refs)
    return result
