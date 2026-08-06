from __future__ import annotations

import argparse
import os
import sys

from tools.graph.check import run_all
from tools.graph.reader import harvest, nodes
from tools.graph.render import render_catalog, render_graph

GRAPH_DIR = "docs/graph"
INDEX = f"{GRAPH_DIR}/index.md"
GRAPH = f"{GRAPH_DIR}/graph.md"


def cmd_index(args) -> int:
    os.makedirs(GRAPH_DIR, exist_ok=True)
    edges = harvest()
    node_ids = nodes()
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_catalog(edges, node_ids))
    with open(GRAPH, "w", encoding="utf-8") as fh:
        fh.write(render_graph(edges))
    print(f"wrote {INDEX} + {GRAPH}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"graph-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("graph-check: clean")
    return 0  # NON-BLOCKING


def cmd_neighbors(args) -> int:
    addr = args.address
    edges = harvest()
    inbound = [e for e in edges if e.dst == addr]
    outbound = [e for e in edges if e.src == addr]
    print(f"neighbors of {addr}:")
    print(f"  inbound ({len(inbound)}):")
    for e in inbound:
        print(f"    {e.src} --{e.type}--> {addr}")
    print(f"  outbound ({len(outbound)}):")
    for e in outbound:
        print(f"    {addr} --{e.type}--> {e.dst}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.graph")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    neighbors_parser = sub.add_parser("neighbors")
    neighbors_parser.add_argument("address")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check, "neighbors": cmd_neighbors}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
