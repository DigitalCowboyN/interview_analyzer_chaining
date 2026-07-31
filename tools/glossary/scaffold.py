from __future__ import annotations

import os

from tools.glossary.reader import code_dimensions, code_enums


def new_term(name: str, kind: str, root: str = ".") -> str:
    ct = (code_enums(root) if kind == "enum" else code_dimensions(root)).get(name)
    values = ct.values if ct else []
    source = ct.source if ct else ""
    slug = name.lower().replace("_", "-")
    path = os.path.join(root, "docs/glossary", f"{slug}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vals = "[" + ", ".join(values) + "]"
    content = (f"---\ntype: Term\nterm: {name}\nkind: {kind}\nsource: {source}\n"
               f"values: {vals}\n---\nTODO: define {name}.\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
