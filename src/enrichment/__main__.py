"""Run Layer 2 enrichment from the command line.

Usage: python -m src.enrichment <interview_id> [--force]
"""

import argparse
import asyncio

from src.enrichment.orchestrator import EnrichmentOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich an ingested interview (Layer 2)")
    parser.add_argument("interview_id")
    parser.add_argument("--force", action="store_true", help="Re-enrich already-analyzed fragments")
    args = parser.parse_args()

    orchestrator = EnrichmentOrchestrator()
    result = asyncio.run(orchestrator.enrich_interview(args.interview_id, force=args.force))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
