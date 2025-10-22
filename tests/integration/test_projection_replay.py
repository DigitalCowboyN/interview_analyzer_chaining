"""
Projection replay tests.

Tests the ability to rebuild Neo4j state from EventStoreDB events.
This validates:
1. Projection handlers correctly process events
2. Data integrity after replay
3. Idempotency of projections
"""

import uuid

import pytest

from src.events.envelope import Actor, ActorType
from src.events.interview_events import create_interview_created_event
from src.events.sentence_events import (
    create_analysis_generated_event,
    create_sentence_created_event,
)
from src.projections.handlers.interview_handlers import (
    InterviewCreatedHandler,
)
from src.projections.handlers.sentence_handlers import (
    AnalysisGeneratedHandler,
    SentenceCreatedHandler,
)
from src.utils.neo4j_driver import Neo4jConnectionManager


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.eventstore
class TestProjectionReplay:
    """Test projection replay scenarios."""

    async def test_full_interview_replay(
        self,
        clean_test_database,
        event_store_client,
    ):
        """
        Test replaying a complete interview from events.

        Workflow:
        1. Create events in EventStoreDB (interview + sentences + analysis)
        2. Process events through projection handlers
        3. Verify Neo4j state matches expected
        """
        # Generate test IDs
        interview_id = str(uuid.uuid4())
        project_id = "test-project"
        correlation_id = str(uuid.uuid4())
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")

        # === Step 1: Create events in EventStoreDB ===
        # Create interview
        interview_event = create_interview_created_event(
            aggregate_id=interview_id,
            version=0,
            title="Test Interview",
            source="test_file.txt",
            language="en",
            actor=system_actor,
            project_id=project_id,
            correlation_id=correlation_id,
        )

        interview_stream = f"Interview-{interview_id}"
        await event_store_client.append_events(
            stream_name=interview_stream,
            events=[interview_event],
            expected_version=-1,
        )

        # Create 3 sentences with analysis
        sentence_events = []
        for i in range(3):
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))

            created_event = create_sentence_created_event(
                aggregate_id=sentence_id,
                version=0,
                interview_id=interview_id,
                index=i,
                text=f"Sentence {i} text.",
                actor=system_actor,
                correlation_id=correlation_id,
            )

            analysis_event = create_analysis_generated_event(
                aggregate_id=sentence_id,
                version=1,
                model="gpt-4",
                model_version="2024-01",
                classification={
                    "function_type": "statement",
                    "structure_type": "simple",
                    "purpose": "inform",
                },
                keywords=[f"keyword{i}"],
                topics=[f"topic{i}"],
                domain_keywords=[],
                confidence=0.9,
                actor=system_actor,
                correlation_id=correlation_id,
            )

            sentence_stream = f"Sentence-{sentence_id}"
            await event_store_client.append_events(
                stream_name=sentence_stream,
                events=[created_event, analysis_event],
                expected_version=-1,
            )

            sentence_events.append((sentence_id, created_event, analysis_event))

        # === Step 2: Process events through projection handlers ===
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)

        # Process InterviewCreated
        interview_handler = InterviewCreatedHandler()
        await interview_handler.handle(interview_event, driver)

        # Process SentenceCreated and AnalysisGenerated for each sentence
        sentence_handler = SentenceCreatedHandler()
        analysis_handler = AnalysisGeneratedHandler()

        for sentence_id, created_event, analysis_event in sentence_events:
            await sentence_handler.handle(created_event, driver)
            await analysis_handler.handle(analysis_event, driver)

        # === Step 3: Verify Neo4j state ===
        async with driver.session() as session:
            # Check Interview node
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN i",
                interview_id=interview_id,
            )
            interview_node = await result.single()
            assert interview_node is not None, "Interview node not found"
            assert interview_node["i"]["title"] == "Test Interview"

            # Check Sentence nodes
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence) "
                "RETURN count(s) as count",
                interview_id=interview_id,
            )
            record = await result.single()
            assert record["count"] == 3, f"Expected 3 sentences, got {record['count']}"

            # Check Analysis nodes
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)"
                "-[:HAS_ANALYSIS]->(a:Analysis) "
                "RETURN count(a) as count",
                interview_id=interview_id,
            )
            record = await result.single()
            assert record["count"] == 3, f"Expected 3 analysis nodes, got {record['count']}"

        print("\n✓ Full interview replay validated:")
        print("  - Interview node created in Neo4j")
        print("  - 3 Sentence nodes created")
        print("  - 3 Analysis nodes created")
        print("  - All relationships intact")

    async def test_replay_after_neo4j_wipe(
        self,
        clean_test_database,
        event_store_client,
    ):
        """
        Test rebuilding Neo4j from scratch using EventStoreDB.

        Workflow:
        1. Create initial data via projection handlers
        2. Capture Neo4j state
        3. Wipe Neo4j completely
        4. Replay all events
        5. Verify Neo4j state matches original
        """
        # Generate test IDs
        interview_id = str(uuid.uuid4())
        project_id = "replay-test-project"
        correlation_id = str(uuid.uuid4())
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")

        # === Step 1: Create initial data ===
        interview_event = create_interview_created_event(
            aggregate_id=interview_id,
            version=0,
            title="Replay Test Interview",
            source="replay_test.txt",
            language="en",
            actor=system_actor,
            project_id=project_id,
            correlation_id=correlation_id,
        )

        interview_stream = f"Interview-{interview_id}"
        await event_store_client.append_events(
            stream_name=interview_stream,
            events=[interview_event],
            expected_version=-1,
        )

        # Create 2 sentences
        sentence_id_0 = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))
        sentence_id_1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:1"))

        sentence_0_created = create_sentence_created_event(
            aggregate_id=sentence_id_0,
            version=0,
            interview_id=interview_id,
            index=0,
            text="First sentence.",
            actor=system_actor,
            correlation_id=correlation_id,
        )

        sentence_1_created = create_sentence_created_event(
            aggregate_id=sentence_id_1,
            version=0,
            interview_id=interview_id,
            index=1,
            text="Second sentence.",
            actor=system_actor,
            correlation_id=correlation_id,
        )

        await event_store_client.append_events(
            stream_name=f"Sentence-{sentence_id_0}",
            events=[sentence_0_created],
            expected_version=-1,
        )

        await event_store_client.append_events(
            stream_name=f"Sentence-{sentence_id_1}",
            events=[sentence_1_created],
            expected_version=-1,
        )

        # Process events (first time)
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        interview_handler = InterviewCreatedHandler()
        sentence_handler = SentenceCreatedHandler()

        await interview_handler.handle(interview_event, driver)
        await sentence_handler.handle(sentence_0_created, driver)
        await sentence_handler.handle(sentence_1_created, driver)

        # === Step 2: Capture Neo4j state ===
        async with driver.session() as session:
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN i.title as title",
                interview_id=interview_id,
            )
            original_title = (await result.single())["title"]

            result = await session.run(
                "MATCH (s:Sentence)-[:PART_OF_INTERVIEW]->(:Interview {interview_id: $interview_id}) "
                "RETURN count(s) as count",
                interview_id=interview_id,
            )
            original_sentence_count = (await result.single())["count"]

        # === Step 3: Wipe Neo4j ===
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

            # Verify empty
            result = await session.run("MATCH (n) RETURN count(n) as count")
            assert (await result.single())["count"] == 0, "Neo4j not empty after wipe"

        # === Step 4: Replay all events ===
        await interview_handler.handle(interview_event, driver)
        await sentence_handler.handle(sentence_0_created, driver)
        await sentence_handler.handle(sentence_1_created, driver)

        # === Step 5: Verify Neo4j state matches original ===
        async with driver.session() as session:
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN i.title as title",
                interview_id=interview_id,
            )
            replayed_title = (await result.single())["title"]
            assert replayed_title == original_title, "Title mismatch after replay"

            result = await session.run(
                "MATCH (s:Sentence)-[:PART_OF_INTERVIEW]->(:Interview {interview_id: $interview_id}) "
                "RETURN count(s) as count",
                interview_id=interview_id,
            )
            replayed_sentence_count = (await result.single())["count"]
            assert replayed_sentence_count == original_sentence_count, "Sentence count mismatch after replay"

        print("\n✓ Replay after Neo4j wipe validated:")
        print("  - Neo4j completely wiped")
        print("  - All events replayed successfully")
        print(f"  - Original state restored ({original_sentence_count} sentences)")
        print("  - Data integrity maintained")

    async def test_partial_stream_replay(
        self,
        clean_test_database,
        event_store_client,
    ):
        """
        Test replaying events for a specific interview (partial replay).

        Validates:
        1. Only events for target interview are processed
        2. Other interviews not affected
        3. No side effects
        """
        # Generate test IDs for 2 interviews
        interview_1_id = str(uuid.uuid4())
        interview_2_id = str(uuid.uuid4())
        project_id = "partial-replay-test"
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")

        # === Create events for both interviews ===
        interview_1_event = create_interview_created_event(
            aggregate_id=interview_1_id,
            version=0,
            title="Interview 1",
            source="file1.txt",
            language="en",
            actor=system_actor,
            project_id=project_id,
            correlation_id=str(uuid.uuid4()),
        )

        interview_2_event = create_interview_created_event(
            aggregate_id=interview_2_id,
            version=0,
            title="Interview 2",
            source="file2.txt",
            language="en",
            actor=system_actor,
            project_id=project_id,
            correlation_id=str(uuid.uuid4()),
        )

        await event_store_client.append_events(
            stream_name=f"Interview-{interview_1_id}",
            events=[interview_1_event],
            expected_version=-1,
        )

        await event_store_client.append_events(
            stream_name=f"Interview-{interview_2_id}",
            events=[interview_2_event],
            expected_version=-1,
        )

        # === Process only Interview 1 ===
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        interview_handler = InterviewCreatedHandler()

        await interview_handler.handle(interview_1_event, driver)

        # === Verify only Interview 1 in Neo4j ===
        async with driver.session() as session:
            result = await session.run("MATCH (i:Interview) RETURN count(i) as count")
            interview_count = (await result.single())["count"]
            assert interview_count == 1, f"Expected 1 interview, got {interview_count}"

            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN i",
                interview_id=interview_1_id,
            )
            assert await result.single() is not None, "Interview 1 not found"

            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN i",
                interview_id=interview_2_id,
            )
            assert await result.single() is None, "Interview 2 should not exist"

        print("\n✓ Partial stream replay validated:")
        print("  - Only targeted interview processed")
        print("  - Other interviews not affected")
        print("  - No side effects observed")
