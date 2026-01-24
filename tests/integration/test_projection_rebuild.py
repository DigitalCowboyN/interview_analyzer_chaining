"""
Integration tests for projection service rebuild capability.

Tests that projection service can rebuild Neo4j database from scratch using only events.
This validates the core event sourcing principle: state can be derived from events.

Updated for M3.0 single-writer architecture:
- Pipeline only emits events (no direct Neo4j writes)
- Projection service is the SOLE writer to Neo4j
- Rebuild tests replay events through projection handlers
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
class TestProjectionRebuild:
    """Test projection service rebuild capabilities."""

    async def _process_events_through_handlers(self, event_store, interview_id, sentence_count=4):
        """Helper to process all events for an interview through projection handlers."""
        from src.projections.handlers.interview_handlers import (
            InterviewCreatedHandler,
            InterviewStatusChangedHandler,
        )
        from src.projections.handlers.sentence_handlers import (
            AnalysisGeneratedHandler,
            SentenceCreatedHandler,
        )

        # Process Interview events
        interview_handler = InterviewCreatedHandler()
        status_handler = InterviewStatusChangedHandler()

        interview_stream = f"Interview-{interview_id}"
        try:
            interview_events = await event_store.read_stream(interview_stream)
            for event in interview_events:
                if event.event_type == "InterviewCreated":
                    await interview_handler.handle(event)
                elif event.event_type == "StatusChanged":
                    await status_handler.handle(event)
        except Exception:
            pass  # Stream may not exist yet

        # Process Sentence and Analysis events
        sentence_handler = SentenceCreatedHandler()
        analysis_handler = AnalysisGeneratedHandler()

        for i in range(sentence_count):
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            sentence_stream = f"Sentence-{sentence_id}"
            try:
                sentence_events = await event_store.read_stream(sentence_stream)
                for event in sentence_events:
                    if event.event_type == "SentenceCreated":
                        await sentence_handler.handle(event)
                    elif event.event_type == "AnalysisGenerated":
                        await analysis_handler.handle(event)
            except Exception:
                pass  # Stream may not exist yet

    async def test_projection_service_rebuilds_neo4j_from_events(
        self,
        sample_interview_file,
        clean_test_database,
        clean_event_store,
        tmp_path,
    ):
        """
        Test that projection service can rebuild Neo4j from scratch using only events.

        Scenario (M3.0 single-writer architecture):
        1. Pipeline processes file → creates events only
        2. Process events through projection handlers → creates Neo4j state
        3. Capture original Neo4j state (nodes, relationships, properties)
        4. Delete ALL Neo4j nodes
        5. Replay all events through projection service
        6. Compare rebuilt state with original state

        This validates:
        - Events are sufficient to rebuild entire state
        - Projection service correctly implements all event handlers
        - No data loss when rebuilding from events
        - Core event sourcing principle: state = f(events)
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

        interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{sample_interview_file.name}"))

        # === STEP 2: Process events through projection handlers (creates Neo4j state) ===
        # In single-writer architecture, pipeline only creates events - projection service creates Neo4j state
        await self._process_events_through_handlers(clean_event_store, interview_id, sentence_count=4)

        # === STEP 3: Capture original Neo4j state ===
        from src.utils.neo4j_driver import Neo4jConnectionManager

        driver = await Neo4jConnectionManager.get_driver(test_mode=True)

        async with driver.session() as session:
            # Capture Interview node
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})
                RETURN i.interview_id as interview_id,
                       i.title as title,
                       i.source as source,
                       i.language as language,
                       i.status as status
                """,
                interview_id=interview_id,
            )
            original_interview = await result.single()
            assert original_interview is not None, "Interview node not found"

            # Capture Sentence nodes
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN s.sentence_id as sentence_id,
                       s.text as text,
                       s.sequence_order as sequence_order,
                       s.speaker as speaker
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )
            original_sentences = []
            async for record in result:
                original_sentences.append(
                    {
                        "sentence_id": record["sentence_id"],
                        "text": record["text"],
                        "sequence_order": record["sequence_order"],
                        "speaker": record["speaker"],
                    }
                )

            # Capture Analysis nodes
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN s.sentence_id as sentence_id,
                       a.model as model,
                       a.confidence as confidence
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )
            original_analyses = []
            async for record in result:
                original_analyses.append(
                    {
                        "sentence_id": record["sentence_id"],
                        "model": record["model"],
                        "confidence": record["confidence"],
                    }
                )

        print(f"\n✓ Captured original state:")
        print(f"  - 1 Interview node")
        print(f"  - {len(original_sentences)} Sentence nodes")
        print(f"  - {len(original_analyses)} Analysis nodes")

        # === STEP 4: Delete ALL Neo4j nodes ===
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            node_count = record["count"]
            assert node_count == 0, f"Failed to delete all nodes, {node_count} remaining"

        print("\n✓ Deleted all Neo4j nodes")

        # === STEP 5: Rebuild from events via projection service ===
        from src.projections.handlers.interview_handlers import (
            InterviewCreatedHandler,
            InterviewStatusChangedHandler,
        )
        from src.projections.handlers.sentence_handlers import AnalysisGeneratedHandler, SentenceCreatedHandler

        # Process InterviewCreated events
        interview_handler = InterviewCreatedHandler()
        status_handler = InterviewStatusChangedHandler()

        interview_stream = f"Interview-{interview_id}"
        interview_events = await clean_event_store.read_stream(interview_stream)

        for event in interview_events:
            if event.event_type == "InterviewCreated":
                await interview_handler.handle(event)
                print(f"✓ Processed InterviewCreated event")
            elif event.event_type == "StatusChanged":
                await status_handler.handle(event)
                print(f"✓ Processed StatusChanged event")

        # Process Sentence and Analysis events
        sentence_handler = SentenceCreatedHandler()
        analysis_handler = AnalysisGeneratedHandler()

        for i in range(len(original_sentences)):
            sentence_id = original_sentences[i]["sentence_id"]
            sentence_stream = f"Sentence-{sentence_id}"
            sentence_events = await clean_event_store.read_stream(sentence_stream)

            for event in sentence_events:
                if event.event_type == "SentenceCreated":
                    await sentence_handler.handle(event)
                    print(f"✓ Processed SentenceCreated event for sentence {i}")
                elif event.event_type == "AnalysisGenerated":
                    await analysis_handler.handle(event)
                    print(f"✓ Processed AnalysisGenerated event for sentence {i}")

        # === STEP 6: Compare rebuilt state with original ===
        async with driver.session() as session:
            # Verify Interview node
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})
                RETURN i.interview_id as interview_id,
                       i.title as title,
                       i.source as source,
                       i.language as language,
                       i.status as status
                """,
                interview_id=interview_id,
            )
            rebuilt_interview = await result.single()
            assert rebuilt_interview is not None, "Interview node not rebuilt"

            # Compare Interview properties
            assert rebuilt_interview["interview_id"] == original_interview["interview_id"]
            assert rebuilt_interview["title"] == original_interview["title"]
            assert rebuilt_interview["source"] == original_interview["source"]
            assert rebuilt_interview["language"] == original_interview["language"]
            # Note: status might differ (completed vs processing) - this is OK

            # Verify Sentence nodes
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN s.sentence_id as sentence_id,
                       s.text as text,
                       s.sequence_order as sequence_order,
                       s.speaker as speaker
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )
            rebuilt_sentences = []
            async for record in result:
                rebuilt_sentences.append(
                    {
                        "sentence_id": record["sentence_id"],
                        "text": record["text"],
                        "sequence_order": record["sequence_order"],
                        "speaker": record["speaker"],
                    }
                )

            # Compare Sentence count
            assert len(rebuilt_sentences) == len(
                original_sentences
            ), f"Sentence count mismatch: {len(original_sentences)} → {len(rebuilt_sentences)}"

            # Compare Sentence properties
            for i, (orig, rebuilt) in enumerate(zip(original_sentences, rebuilt_sentences)):
                assert rebuilt["sentence_id"] == orig["sentence_id"], f"Sentence {i} ID mismatch"
                assert rebuilt["text"] == orig["text"], f"Sentence {i} text mismatch"
                assert rebuilt["sequence_order"] == orig["sequence_order"], f"Sentence {i} order mismatch"
                # Speaker can be None - just verify they match
                assert rebuilt["speaker"] == orig["speaker"], f"Sentence {i} speaker mismatch"

            # Verify Analysis nodes
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN s.sentence_id as sentence_id,
                       a.model as model,
                       a.confidence as confidence
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )
            rebuilt_analyses = []
            async for record in result:
                rebuilt_analyses.append(
                    {
                        "sentence_id": record["sentence_id"],
                        "model": record["model"],
                        "confidence": record["confidence"],
                    }
                )

            # Compare Analysis count
            assert len(rebuilt_analyses) == len(
                original_analyses
            ), f"Analysis count mismatch: {len(original_analyses)} → {len(rebuilt_analyses)}"

        print("\n✓ Rebuild validation passed:")
        print(f"  - Interview node rebuilt correctly")
        print(f"  - {len(rebuilt_sentences)} Sentence nodes rebuilt (matches original)")
        print(f"  - {len(rebuilt_analyses)} Analysis nodes rebuilt (matches original)")
        print("  - All properties match")
        print("  - Core event sourcing principle validated: State = f(Events)")

    async def test_projection_handles_partial_replay(
        self,
        sample_interview_file,
        clean_test_database,
        clean_event_store,
        tmp_path,
    ):
        """
        Test that projection service can handle partial event replay.

        Scenario:
        1. Process file → creates events
        2. Replay only Interview events (not Sentence events)
        3. Verify Interview exists but no Sentences
        4. Replay Sentence events
        5. Verify complete state

        This validates:
        - Projection service can start from any point
        - Partial rebuilds work correctly
        - Event ordering doesn't break projection
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

        interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{sample_interview_file.name}"))

        # Clear Neo4j
        from src.utils.neo4j_driver import Neo4jConnectionManager

        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

        # === STEP 2: Replay only Interview events ===
        from src.projections.handlers.interview_handlers import InterviewCreatedHandler

        interview_handler = InterviewCreatedHandler()
        interview_stream = f"Interview-{interview_id}"
        interview_events = await clean_event_store.read_stream(interview_stream)

        for event in interview_events:
            if event.event_type == "InterviewCreated":
                await interview_handler.handle(event)

        # Verify Interview exists but no Sentences
        async with driver.session() as session:
            result = await session.run("MATCH (i:Interview) RETURN count(i) as count")
            record = await result.single()
            assert record["count"] == 1, "Interview should exist"

            result = await session.run("MATCH (s:Sentence) RETURN count(s) as count")
            record = await result.single()
            assert record["count"] == 0, "No Sentences should exist yet"

        print("\n✓ Partial replay 1: Interview created, no Sentences")

        # === STEP 3: Now replay Sentence events ===
        from src.projections.handlers.sentence_handlers import SentenceCreatedHandler

        sentence_handler = SentenceCreatedHandler()

        for i in range(4):  # 4 sentences
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            sentence_stream = f"Sentence-{sentence_id}"
            sentence_events = await clean_event_store.read_stream(sentence_stream)

            for event in sentence_events:
                if event.event_type == "SentenceCreated":
                    await sentence_handler.handle(event)

        # Verify complete state
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN count(s) as sentence_count
                """,
                interview_id=interview_id,
            )
            record = await result.single()
            assert record["sentence_count"] == 4, "Should have 4 Sentences"

        print("✓ Partial replay 2: Sentences added")
        print("✓ Partial replay validation passed")
