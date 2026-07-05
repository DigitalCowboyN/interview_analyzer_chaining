"""Run Layer 1 ingestion from the command line.

Usage: python -m src.ingestion <file> [--project-id ID] [--map-dir DIR]
"""

import argparse
import asyncio
from pathlib import Path

from src.ingestion.orchestrator import IngestionOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a transcript (Layer 1)")
    parser.add_argument("file", type=Path)
    parser.add_argument("--project-id", default="default-project")
    parser.add_argument("--map-dir", type=Path, default=Path("data/maps"))
    args = parser.parse_args()

    orchestrator = IngestionOrchestrator(project_id=args.project_id, map_dir=args.map_dir)
    result = asyncio.run(orchestrator.ingest_file(args.file))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
