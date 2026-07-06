"""Run Layer 1 ingestion from the command line.

Usage: python -m src.ingestion <file> [--project-id ID] [--map-dir DIR] [--enrich]
"""

import argparse
import asyncio
from pathlib import Path

from src.ingestion.orchestrator import IngestionOrchestrator


async def _run(args) -> None:
    orchestrator = IngestionOrchestrator(project_id=args.project_id, map_dir=args.map_dir)
    result = await orchestrator.ingest_file(args.file)
    print(result.model_dump_json(indent=2))

    if args.enrich:
        # Chain Layer 2 enrichment on the freshly ingested interview.
        from src.enrichment.orchestrator import EnrichmentOrchestrator

        enrich_result = await EnrichmentOrchestrator().enrich_interview(result.interview_id)
        print(enrich_result.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a transcript (Layer 1)")
    parser.add_argument("file", type=Path)
    parser.add_argument("--project-id", default="default-project")
    parser.add_argument("--map-dir", type=Path, default=Path("data/maps"))
    parser.add_argument(
        "--enrich", action="store_true", help="Chain Layer 2 enrichment after ingestion"
    )
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
