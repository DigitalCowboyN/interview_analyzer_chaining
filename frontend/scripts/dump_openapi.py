"""
dump_openapi.py

Dumps the FastAPI app's OpenAPI schema to a JSON file WITHOUT starting a
server — imports `src.main:app` directly and calls `app.openapi()`. This
lets `npm run typegen` regenerate frontend types offline.

Usage: python scripts/dump_openapi.py <output-path>
Must be run from the repo root with the backend's env vars sourced
(`set -a; source .env; set +a`) since importing src.main initializes the
agent factory singleton, which requires ANTHROPIC_API_KEY.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/dump_openapi.py <output-path>", file=sys.stderr)
        sys.exit(1)

    from src.main import app  # noqa: E402 (deferred import — needs repo root on sys.path)

    out_path = Path(sys.argv[1])
    with out_path.open("w") as f:
        json.dump(app.openapi(), f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
