from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

_TARGET = re.compile(r"^([a-zA-Z][\w-]*):")


@dataclass
class Command:
    name: str
    kind: str          # "make" | "module"
    description: str
    visibility: str    # "everyday" | "internal" | "undocumented"


def parse_makefile(path: str) -> List[Command]:
    out: dict = {}
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("\t"):
            continue
        if ":=" in line.split("##", 1)[0]:
            continue  # variable assignment, not a target
        m = _TARGET.match(line)
        if not m:
            continue
        name = m.group(1)
        if "##@" in line:
            desc, vis = line.split("##@", 1)[1].strip(), "internal"
        elif "##" in line:
            desc, vis = line.split("##", 1)[1].strip(), "everyday"
        else:
            desc, vis = "", "undocumented"
        # prefer a documented entry if the name recurs
        if name not in out or (not out[name].description and desc):
            out[name] = Command(name, "make", desc, vis)
    return list(out.values())


def _package_doc(dirpath: str) -> str:
    for fn in ("__init__.py", "__main__.py"):
        p = os.path.join(dirpath, fn)
        if not os.path.exists(p):
            continue
        m = re.search(r'"""(.*?)"""', open(p, encoding="utf-8").read(), re.DOTALL)
        if m and m.group(1).strip():
            first = m.group(1).strip().splitlines()[0].strip()
            if first:
                return first
    return ""


def module_entrypoints(root: str, subdirs=("src", "tools")) -> List[Command]:
    cmds: List[Command] = []
    for base in subdirs:
        start = os.path.join(root, base)
        if not os.path.isdir(start):
            continue
        for dirpath, _dirs, files in os.walk(start):
            if "__main__.py" in files:
                dotted = os.path.relpath(dirpath, root).replace(os.sep, ".")
                cmds.append(Command(f"python -m {dotted}", "module", _package_doc(dirpath), "everyday"))
    return sorted(cmds, key=lambda c: c.name)
