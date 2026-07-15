"""Resolve canonical entities and person identities for a project.

Usage: python -m src.resolution <project_id> [--force]
"""

import argparse
import asyncio

from src.resolution.engine import ResolutionEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Layer 4 resolution for a project")
    parser.add_argument("project_id")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reserved. Runs are full recomputes; skips are driven by aggregate "
            "state (locked canonicals, blocked speaker pairs), not run markers."
        ),
    )
    args = parser.parse_args()

    engine = ResolutionEngine()
    result = asyncio.run(engine.apply(args.project_id, force=args.force))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
