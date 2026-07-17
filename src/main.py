"""
main.py

Entry point for the Interview Analyzer. Serves the FastAPI app (uvicorn) and
provides a batch CLI that ingests + enriches every transcript in the input
directory through the event-sourced Layer 1/Layer 2 path.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI

from src.api.routers import analysis as analysis_router
from src.api.routers import ask as ask_router
from src.api.routers import edits as edits_router
from src.api.routers import exports as exports_router
from src.api.routers import files as files_router
from src.api.routers import lenses as lenses_router
from src.api.routers import queries as queries_router
from src.api.routers import resolution as resolution_router
from src.api.routers import segments as segments_router
from src.api.routers import speakers as speakers_router
from src.api.routers import ui as ui_router
from src.config import config
from src.utils.logger import get_logger
from src.utils.metrics import metrics_tracker

logger = get_logger()

app = FastAPI(
    title="Interview Analyzer API",
    description="API for analyzing interview transcripts.",
    version="0.1.0",
)

app.include_router(files_router.router)
app.include_router(analysis_router.router)
app.include_router(ask_router.router)
app.include_router(edits_router.router)
app.include_router(speakers_router.router)
app.include_router(segments_router.router)
app.include_router(resolution_router.router)
app.include_router(lenses_router.router)
app.include_router(exports_router.router)
app.include_router(queries_router.router)
app.include_router(ui_router.router)


@app.get("/", tags=["Health Check"])
async def read_root():
    """Basic health check endpoint."""
    return {"status": "ok"}


async def _batch_ingest_enrich(input_dir: Path, map_dir: Path, project_id: str) -> None:
    """Ingest + enrich every .txt transcript in input_dir."""
    from src.enrichment.orchestrator import EnrichmentOrchestrator
    from src.ingestion.orchestrator import IngestionOrchestrator

    ingest = IngestionOrchestrator(project_id=project_id, map_dir=map_dir)
    enrich = EnrichmentOrchestrator()
    files = sorted(input_dir.glob("*.txt"))
    logger.info(f"Batch processing {len(files)} transcript(s) from {input_dir}")
    failures = []
    for file_path in files:
        try:
            result = await ingest.ingest_file(file_path)
            await enrich.enrich_interview(result.interview_id)
            logger.info(f"Processed {file_path.name} -> interview {result.interview_id}")
        except Exception as exc:
            logger.error(f"Failed to process {file_path.name}: {exc}", exc_info=True)
            failures.append(file_path.name)
    if failures:
        raise RuntimeError(f"{len(failures)} file(s) failed: {failures}")


def main():
    """Batch CLI: ingest + enrich every transcript in the input directory."""
    parser = argparse.ArgumentParser(description="Interview Analyzer batch ingest + enrich")
    parser.add_argument(
        "--input_dir", type=Path, default=Path(config["paths"]["input_dir"]),
        help="Directory of transcript files to process",
    )
    parser.add_argument(
        "--map_dir", type=Path, default=Path(config["paths"].get("map_dir", "data/maps")),
        help="Directory for map files (.jsonl)",
    )
    parser.add_argument("--project-id", default="default-project")
    args = parser.parse_args()

    metrics_tracker.reset()
    metrics_tracker.start_pipeline_timer()
    logger.info("Starting batch ingest + enrich")
    failed = False
    try:
        asyncio.run(_batch_ingest_enrich(args.input_dir, args.map_dir, args.project_id))
        logger.info("Batch processing completed.")
    except Exception as e:
        logger.critical(f"Batch processing failed: {e}", exc_info=True)
        metrics_tracker.increment_errors()
        failed = True
    finally:
        metrics_tracker.stop_pipeline_timer()
        summary = metrics_tracker.get_summary()
        logger.info(f"Execution Summary: {json.dumps(summary, indent=2)}")
    if failed:
        sys.exit(1)  # non-zero exit so CI/scripts detect batch failure


if __name__ == "__main__":
    main()
