"""
Integration tests for projection service dual-write validation.

Tests that projection service correctly handles dual-write scenario:
- No duplicate nodes when both pipeline and projection write
- Version tracking works correctly
- MERGE logic prevents conflicts
"""

import uuid
from pathlib import Path

import pytest

from src.config import config
from src.pipeline import PipelineOrchestrator
from src.utils.environment import detect_environment


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.eventstore
@pytest.mark.neo4j
@pytest.mark.skip(reason="Projection service integration tests - requires M2.8 transition and proper service setup")
class TestProjectionDualWriteValidation:
    """Test projection service behavior during dual-write phase."""

    async def test_projection_service_deduplicates_during_dual_write(
        self,
        sample_interview_file,
        clean_test_database,
        clean_event_store,
        tmp_path,
    ):
        """
        Test that projection service correctly handles dual-write scenario.

        Scenario:
        1. Pipeline processes file → creates events + direct writes to Neo4j
        2. Verify direct write exists (event_version=0, source=pipeline_direct)
        3. Process events through projection service handlers
        4. Verify: Only 1 node per sentence (no duplicates)
        5. Verify: event_version updated from 0 to N
        6. Verify: source updated to projection_service

        This validates:
        - MERGE logic prevents duplicates
        - Version checking works correctly
        - Projection service can coexist with direct writes
        """
        # === STEP 1: Process file through pipeline (dual-write mode) ===
        test_config = config.copy()

        # Enable event sourcing
        environment = detect_environment()
        if environment in ("docker", "ci"):
            esdb_connection = "esdb://eventstore:2113?tls=false"
        else:
            esdb_connection = "esdb://localhost:2113?tls=false"
        test_config["event_sourcing"] = {"enabled": True, "connection_string": esdb_connection}

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_config["paths"]["output_dir"] = str(output_dir)

        # Process file
        pipeline = PipelineOrchestrator(
            input_dir=sample_interview_file.parent, output_dir=output_dir, config_dict=test_config
        )
        await pipeline._process_single_file(Path(sample_interview_file))

        # === STEP 2: Verify direct writes exist in Neo4j ===
        from src.utils.neo4j_driver import Neo4jConnectionManager

        driver = await Neo4jConnectionManager.get_driver(test_mode=True)

        # Calculate interview_id (deterministic)
        interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{sample_interview_file.name}"))

        # Query for sentences written by pipeline (direct writes)
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN s.sentence_id as sentence_id,
                       s.text as text,
                       s.event_version as event_version,
                       s.source as source,
                       count(s) as node_count
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )
            direct_writes = []
            async for record in result:
                direct_writes.append(
                    {
                        "sentence_id": record["sentence_id"],
                        "text": record["text"],
                        "event_version": record["event_version"],
                        "source": record["source"],
                        "node_count": record["node_count"],
                    }
                )

        # Verify direct writes exist
        assert len(direct_writes) == 4, f"Expected 4 sentences, got {len(direct_writes)}"

        # Verify direct write properties
        for sentence_data in direct_writes:
            # Direct writes don't currently set event_version or source
            # (this is OK - projection service will set them)
            assert sentence_data["node_count"] == 1, "Should have exactly 1 node per sentence"

        print(f"\n✓ Verified {len(direct_writes)} sentences written by pipeline (direct writes)")

        # === STEP 3: Process events through projection service ===
        from src.projections.handlers.interview_handlers import InterviewCreatedHandler
        from src.projections.handlers.sentence_handlers import AnalysisGeneratedHandler, SentenceCreatedHandler

        # Read events from EventStore
        interview_stream = f"Interview-{interview_id}"
        interview_events = await clean_event_store.read_stream(interview_stream)

        # Process InterviewCreated event
        interview_handler = InterviewCreatedHandler()
        for event in interview_events:
            if event.event_type == "InterviewCreated":
                await interview_handler.handle(event)
                print(f"✓ Processed InterviewCreated event (version {event.version})")

        # Process SentenceCreated events for each sentence
        sentence_handler = SentenceCreatedHandler()
        analysis_handler = AnalysisGeneratedHandler()

        for i in range(4):  # 4 sentences in sample file
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            sentence_stream = f"Sentence-{sentence_id}"
            sentence_events = await clean_event_store.read_stream(sentence_stream)

            for event in sentence_events:
                if event.event_type == "SentenceCreated":
                    await sentence_handler.handle(event)
                    print(f"✓ Processed SentenceCreated event for sentence {i} (version {event.version})")
                elif event.event_type == "AnalysisGenerated":
                    await analysis_handler.handle(event)
                    print(f"✓ Processed AnalysisGenerated event for sentence {i} (version {event.version})")

        # === STEP 4: Verify no duplicates and version tracking ===
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN s.sentence_id as sentence_id,
                       s.text as text,
                       s.event_version as event_version,
                       s.source as source,
                       count(s) as node_count
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )
            final_state = []
            async for record in result:
                final_state.append(
                    {
                        "sentence_id": record["sentence_id"],
                        "text": record["text"],
                        "event_version": record["event_version"],
                        "source": record["source"],
                        "node_count": record["node_count"],
                    }
                )

        # Verify results
        assert len(final_state) == 4, f"Expected 4 sentences after projection, got {len(final_state)}"

        for i, sentence_data in enumerate(final_state):
            # No duplicates
            assert sentence_data["node_count"] == 1, f"Sentence {i} has duplicates: {sentence_data['node_count']}"

            # Version tracking (projection handlers update version)
            # SentenceCreated is version 0, so event_version should be 0 after projection
            assert (
                sentence_data["event_version"] == 0
            ), f"Sentence {i} event_version should be 0, got {sentence_data['event_version']}"

            # Source tracking (projection handlers update source)
            # Note: Direct writes from pipeline don't set source field,
            # so it will be None until projection service processes event
            # After projection, it should be "projection_service"
            assert (
                sentence_data["source"] == "projection_service"
            ), f"Sentence {i} source should be 'projection_service', got {sentence_data['source']}"

        print("\n✓ Deduplication validation passed:")
        print("  - No duplicate nodes created")
        print("  - Version tracking works correctly")
        print("  - Source field updated by projection service")
        print("  - MERGE logic prevents conflicts")

    async def test_projection_service_handles_analysis_events(
        self,
        sample_interview_file,
        clean_test_database,
        clean_event_store,
        tmp_path,
    ):
        """
        Test that projection service correctly handles AnalysisGenerated events.

        Scenario:
        1. Pipeline processes file → creates events (SentenceCreated + AnalysisGenerated)
        2. Pipeline writes analyses to Neo4j (direct writes)
        3. Process AnalysisGenerated events through projection service
        4. Verify analyses exist (no duplicates)
        """
        # === STEP 1: Process file through pipeline ===
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

        # === STEP 2: Verify analyses in Neo4j (from pipeline) ===
        from src.utils.neo4j_driver import Neo4jConnectionManager

        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{sample_interview_file.name}"))

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Sentence)<-[:HAS_SENTENCE]-(i:Interview {interview_id: $interview_id})
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN count(a) as analysis_count
                """,
                interview_id=interview_id,
            )
            record = await result.single()
            analysis_count_before = record["analysis_count"] if record else 0

        print(f"\n✓ Verified {analysis_count_before} analyses written by pipeline")

        # === STEP 3: Process AnalysisGenerated events through projection service ===
        from src.projections.handlers.sentence_handlers import AnalysisGeneratedHandler

        analysis_handler = AnalysisGeneratedHandler()

        for i in range(4):  # 4 sentences
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            sentence_stream = f"Sentence-{sentence_id}"
            sentence_events = await clean_event_store.read_stream(sentence_stream)

            for event in sentence_events:
                if event.event_type == "AnalysisGenerated":
                    await analysis_handler.handle(event)
                    print(f"✓ Processed AnalysisGenerated event for sentence {i}")

        # === STEP 4: Verify analyses still exist (no duplicates) ===
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Sentence)<-[:HAS_SENTENCE]-(i:Interview {interview_id: $interview_id})
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN count(a) as analysis_count
                """,
                interview_id=interview_id,
            )
            record = await result.single()
            analysis_count_after = record["analysis_count"] if record else 0

        # Should have same number of analyses (projection service uses MERGE)
        assert (
            analysis_count_after == analysis_count_before
        ), f"Analysis count changed: {analysis_count_before} → {analysis_count_after}"

        print("\n✓ Analysis event handling validation passed:")
        print(f"  - {analysis_count_after} analyses exist (no duplicates)")
        print("  - Projection service MERGE logic works correctly")
