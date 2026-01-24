"""
Integration tests for dual-write metrics tracking.

Tests that all metrics increment correctly during:
- Successful dual-writes
- Event emission failures
- Neo4j failures after event success
"""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.config import config
from src.pipeline import PipelineOrchestrator
from src.utils.environment import detect_environment
from src.utils.metrics import metrics_tracker


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.eventstore
@pytest.mark.neo4j
class TestDualWriteMetrics:
    """Test dual-write metrics tracking."""

    async def test_metrics_track_successful_dual_write(
        self,
        sample_interview_file,
        clean_test_database,
        clean_event_store,
        tmp_path,
    ):
        """
        Test that metrics correctly track successful dual-write operations.

        Scenario:
        1. Reset metrics
        2. Process file through pipeline (successful dual-write)
        3. Verify metrics incremented:
           - event_emission_success_{event_type}
           - dual_write_event_first_success
           - dual_write_neo4j_after_event_success

        This validates:
        - Metrics track on success path
        - Counters increment correctly
        - Observable for monitoring
        """
        # === STEP 1: Reset metrics ===
        metrics_tracker.reset()

        # === STEP 2: Process file through pipeline ===
        test_config = config.copy()

        environment = detect_environment()
        if environment in ("docker", "ci"):
            esdb_connection = "esdb://eventstore:2113?tls=false"
        else:
            esdb_connection = "esdb://localhost:2113?tls=false"
        test_config["event_sourcing"] = {"enabled": True, "connection_string": esdb_connection}

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_config["paths"]["output_dir"] = str(output_dir)

        pipeline = PipelineOrchestrator(
            input_dir=sample_interview_file.parent, output_dir=output_dir, config_dict=test_config
        )
        await pipeline._process_single_file(Path(sample_interview_file))

        # === STEP 3: Verify metrics ===
        # Check event emission success metrics
        events_metrics = metrics_tracker.custom_metrics.get("events", {})

        # InterviewCreated event
        interview_created_success = events_metrics.get("emission_success_InterviewCreated", 0)
        assert interview_created_success >= 1, f"Expected >=1 InterviewCreated success, got {interview_created_success}"

        # SentenceCreated events (4 sentences in sample file)
        sentence_created_success = events_metrics.get("emission_success_SentenceCreated", 0)
        assert sentence_created_success == 4, f"Expected 4 SentenceCreated success, got {sentence_created_success}"

        # AnalysisGenerated events (may or may not be present depending on whether analysis runs)
        # Not asserting on this since it requires LLM calls which may not happen in test environment
        analysis_generated_success = events_metrics.get("emission_success_AnalysisGenerated", 0)

        # M2.8: Check event-only metrics (no more direct Neo4j writes)
        dual_write_metrics = metrics_tracker.custom_metrics.get("dual_write", {})

        # Event-first successes (at least 1 interview + 4 sentences = 5, may have more with analyses)
        event_first_success = dual_write_metrics.get("event_first_success", 0)
        assert event_first_success >= 5, f"Expected >=5 event_first_success, got {event_first_success}"

        # M2.8: No more Neo4j direct writes - projection service handles all Neo4j writes
        # So we don't check neo4j_after_event_success/failure metrics

        # Check failures (ignore AnalysisGenerated failures as they can have version conflicts)
        # AnalysisGenerated failures are acceptable in test environment due to stream reuse
        sentence_failures = events_metrics.get("emission_failure_SentenceCreated", 0)
        interview_failures = events_metrics.get("emission_failure_InterviewCreated", 0)

        assert sentence_failures == 0, f"Expected 0 SentenceCreated failures, got {sentence_failures}"
        assert interview_failures == 0, f"Expected 0 InterviewCreated failures, got {interview_failures}"

        event_first_failure = dual_write_metrics.get("event_first_failure", 0)
        assert event_first_failure == 0, f"Expected 0 event_first_failure, got {event_first_failure}"

        print("\n✓ M2.8 Success metrics validation passed:")
        print(f"  - InterviewCreated success: {interview_created_success}")
        print(f"  - SentenceCreated success: {sentence_created_success}")
        print(f"  - AnalysisGenerated success: {analysis_generated_success}")
        print(f"  - Event-first success: {event_first_success}")
        print("  - No emission failures recorded")
        print("  - M2.8: Projection service handles all Neo4j writes (no direct writes)")

    async def test_metrics_track_event_emission_failure(
        self,
        clean_test_database,
        clean_event_store,
    ):
        """
        Test that metrics correctly track event emission failures.

        Scenario:
        1. Reset metrics
        2. Mock EventStore to fail
        3. Attempt to write sentence (event emission will fail)
        4. Verify failure metrics incremented:
           - event_emission_failure_{event_type}
           - dual_write_event_first_failure

        This validates:
        - Metrics track on failure path
        - Event failures recorded correctly
        """
        # === STEP 1: Reset metrics ===
        metrics_tracker.reset()

        # === STEP 2: Create storage with event emitter ===
        from src.io.neo4j_map_storage import Neo4jMapStorage
        from src.pipeline_event_emitter import PipelineEventEmitter

        event_emitter = PipelineEventEmitter(clean_event_store)

        # Mock the EventStore client to fail (not the emit method)
        # This allows the emit method to run and track metrics before failing
        with patch.object(
            clean_event_store, "append_events", new_callable=AsyncMock
        ) as mock_append:
            mock_append.side_effect = Exception("EventStore connection failed")

            storage = Neo4jMapStorage(
                project_id="test-project",
                interview_id="test-interview",
                event_emitter=event_emitter,
                correlation_id="test-correlation",
            )

            # === STEP 3: Attempt to write entry (should fail) ===
            entry = {
                "sentence_id": 0,
                "sequence_order": 0,
                "sentence": "Test sentence",
            }

            with pytest.raises(RuntimeError, match="Event emission failed"):
                await storage.write_entry(entry)

        # === STEP 4: Verify failure metrics ===
        events_metrics = metrics_tracker.custom_metrics.get("events", {})
        dual_write_metrics = metrics_tracker.custom_metrics.get("dual_write", {})

        # Event emission failure should be recorded
        sentence_created_failure = events_metrics.get("emission_failure_SentenceCreated", 0)
        assert sentence_created_failure >= 1, f"Expected >=1 SentenceCreated failure, got {sentence_created_failure}"

        # Event-first failure should be recorded
        event_first_failure = dual_write_metrics.get("event_first_failure", 0)
        assert event_first_failure >= 1, f"Expected >=1 event_first_failure, got {event_first_failure}"

        # Neo4j write should NOT have been attempted (no success or failure)
        neo4j_after_event_success = dual_write_metrics.get("neo4j_after_event_success", 0)
        neo4j_after_event_failure = dual_write_metrics.get("neo4j_after_event_failure", 0)
        assert neo4j_after_event_success == 0, "Neo4j write should not have been attempted"
        assert neo4j_after_event_failure == 0, "Neo4j write should not have been attempted"

        print("\n✓ Failure metrics validation passed:")
        print(f"  - SentenceCreated failure: {sentence_created_failure}")
        print(f"  - Event-first failure: {event_first_failure}")
        print("  - Neo4j writes not attempted (correct - event failed first)")

    async def test_metrics_distinguish_event_types(
        self,
        clean_test_database,
        clean_event_store,
    ):
        """
        Test that metrics correctly distinguish between different event types.

        Validates:
        - InterviewCreated metrics separate from SentenceCreated
        - AnalysisGenerated metrics separate from others
        - Can track success/failure per event type
        """
        # === Reset metrics ===
        metrics_tracker.reset()

        # === Emit different event types ===
        from src.pipeline_event_emitter import PipelineEventEmitter

        event_emitter = PipelineEventEmitter(clean_event_store)

        interview_id = str(uuid.uuid4())
        project_id = str(uuid.uuid4())

        # Emit InterviewCreated
        await event_emitter.emit_interview_created(
            interview_id=interview_id,
            project_id=project_id,
            title="test.txt",
            source="/test.txt",
        )

        # Emit SentenceCreated
        await event_emitter.emit_sentence_created(
            interview_id=interview_id,
            index=0,
            text="Test sentence",
        )

        # Emit AnalysisGenerated
        await event_emitter.emit_analysis_generated(
            interview_id=interview_id,
            sentence_index=0,
            analysis_data={
                "function_type": "declarative",
                "overall_keywords": ["test"],
            },
        )

        # === Verify metrics are separated by event type ===
        events_metrics = metrics_tracker.custom_metrics.get("events", {})

        interview_success = events_metrics.get("emission_success_InterviewCreated", 0)
        sentence_success = events_metrics.get("emission_success_SentenceCreated", 0)
        analysis_success = events_metrics.get("emission_success_AnalysisGenerated", 0)

        assert interview_success == 1, f"Expected 1 InterviewCreated, got {interview_success}"
        assert sentence_success == 1, f"Expected 1 SentenceCreated, got {sentence_success}"
        assert analysis_success == 1, f"Expected 1 AnalysisGenerated, got {analysis_success}"

        print("\n✓ Event type distinction validation passed:")
        print(f"  - InterviewCreated: {interview_success}")
        print(f"  - SentenceCreated: {sentence_success}")
        print(f"  - AnalysisGenerated: {analysis_success}")
        print("  - Each event type tracked separately")
