"""KG-2 flow derivations parsed from source: the event->handler registry map and each handler's
written Neo4j labels. Consumed by tools.graph.neighbors at level='symbol' (memoized per walk)."""
from __future__ import annotations

import os
import re
from typing import Dict, List

from tools.code.reader import load_units, symbols_of

_REGISTER = re.compile(r'register\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\(')
_MERGE_LABEL = re.compile(r"(?:MERGE|CREATE)\s*\(\s*\w*\s*:\s*(\w+)")


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


def _module_file(module_id: str, root: str) -> str:
    from tools.code.reader import _module_path
    return _module_path(module_id, root)


def handler_labels(handler_id: str, root: str = ".") -> List[str]:
    module_id = handler_id.rsplit(".", 1)[0]         # handler class -> its module
    from tools.glossary.model import load_glossary
    terms = {t.term for t in load_glossary(os.path.join(root, "docs/glossary"))}
    try:
        text = open(_module_file(module_id, root), encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    return sorted({lbl for lbl in _MERGE_LABEL.findall(text) if lbl in terms})


def writes_edges(root: str = ".") -> Dict[str, List[str]]:
    """handler-MODULE id -> the glossary-term labels it writes (Cypher MERGE). Module-grain so the
    schema blast-radius is traversable INBOUND from a label (symmetric with `reads`); the handler
    modules are exactly those hosting a registered handler class."""
    modules = sorted({hid.rsplit(".", 1)[0] for hid in register_map(root).values()})
    return {m: handler_labels(m + ".X", root) for m in modules}
