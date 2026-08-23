"""KG-2 flow derivations parsed from source: the event->handler registry map and each handler's
written Neo4j labels. Consumed by tools.graph.neighbors at level='symbol' (memoized per walk)."""
from __future__ import annotations

import os
import re
from typing import Dict, List

from tools.code.reader import load_units, symbols_of

_REGISTER = re.compile(r'register\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\(')


def _class_index(root: str, pkg_prefix: str) -> Dict[str, str]:
    """class name -> dotted symbol id, over modules under a package prefix (e.g. 'events', 'projections.handlers')."""
    idx: Dict[str, str] = {}
    for u in load_units(root):
        if u.level == "module" and u.unit.startswith(pkg_prefix):
            for s in symbols_of(u.unit, root):
                if s.kind == "class":
                    idx[s.id.split(".")[-1]] = s.id
    return idx


def register_map(root: str = ".") -> Dict[str, str]:
    # e.g. 'InterviewCreatedData' -> events.interview_events.InterviewCreatedData
    events = _class_index(root, "events")
    handlers = _class_index(root, "projections.handlers")
    path = os.path.join(root, "src", "projections", "bootstrap.py")
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return {}
    out: Dict[str, str] = {}
    for m in _REGISTER.finditer(text):
        etype, handler = m.group(1), m.group(2)
        ev = events.get(etype + "Data")              # convention: <Type> -> <Type>Data
        hid = handlers.get(handler)
        if ev and hid:
            out[ev] = hid
    return out
