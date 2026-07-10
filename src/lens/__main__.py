"""Apply a lens to an ingested interview from the command line.

Usage: python -m src.lens <interview_id> <lens_name> [--force]
"""

import argparse
import asyncio

from src.lens.engine import LensEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a lens to an interview (Layer 3)")
    parser.add_argument("interview_id")
    parser.add_argument("lens_name", help="Lens profile name (lenses/<name>.yaml)")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-emit LensApplied even for an already-applied version. Note: a forced "
            "same-version run re-extracts only NOVEL items — existing unlocked items "
            "are kept (their ids already exist); bump the lens version to refresh results."
        ),
    )
    args = parser.parse_args()

    engine = LensEngine()
    result = asyncio.run(engine.apply(args.interview_id, args.lens_name, force=args.force))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
