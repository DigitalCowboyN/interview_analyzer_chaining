"""
tests/test_tasks.py

Tests for the Celery task, which now runs Layer 1 ingestion + Layer 2
enrichment (the task name is preserved for queue compatibility).
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.tasks import _run_pipeline_for_file_core


def _patch_orchestrators():
    """Build patchable ingest + enrich instances."""
    ingest_instance = MagicMock()
    ingest_instance.ingest_file = AsyncMock(return_value=MagicMock(interview_id="iv-1"))
    enrich_instance = MagicMock()
    enrich_instance.enrich_interview = AsyncMock(return_value=MagicMock(fragments_enriched=3))
    return ingest_instance, enrich_instance


class TestRunPipelineForFileTask:
    def test_core_runs_ingest_then_enrich(self):
        ingest_instance, enrich_instance = _patch_orchestrators()
        with patch(
            "src.ingestion.orchestrator.IngestionOrchestrator", return_value=ingest_instance
        ), patch(
            "src.enrichment.orchestrator.EnrichmentOrchestrator", return_value=enrich_instance
        ):
            result = _run_pipeline_for_file_core(
                input_file_path_str="/data/input/interview.txt",
                output_dir_str="/data/output",
                map_dir_str="/data/maps",
                config_dict={"project_id": "proj-x"},
                task_id="t-1",
            )

        assert result["status"] == "Success"
        assert result["file"] == "/data/input/interview.txt"
        assert result["interview_id"] == "iv-1"
        ingest_instance.ingest_file.assert_awaited_once()
        enrich_instance.enrich_interview.assert_awaited_once_with("iv-1")

    def test_core_reraises_on_failure(self):
        ingest_instance = MagicMock()
        ingest_instance.ingest_file = AsyncMock(side_effect=FileNotFoundError("missing"))
        with patch(
            "src.ingestion.orchestrator.IngestionOrchestrator", return_value=ingest_instance
        ):
            try:
                _run_pipeline_for_file_core(
                    input_file_path_str="/data/input/nope.txt",
                    output_dir_str="/data/output",
                    map_dir_str="/data/maps",
                    config_dict={},
                    task_id="t-2",
                )
                assert False, "expected FileNotFoundError"
            except FileNotFoundError:
                pass
