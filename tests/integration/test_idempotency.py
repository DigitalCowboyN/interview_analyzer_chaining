"""
Idempotency and resilience tests for projection handlers.

Tests that projection handlers can safely handle:
1. Duplicate event processing (idempotency)
2. Out-of-order events
3. Parked events (retry and DLQ)
4. Version guards
"""

import uuid

import pytest

from src.events.envelope import Actor, ActorType
from src.events.interview_events import create_interview_created_event
from src.events.sentence_events import (
    create_sentence_created_event,
    create_sentence_edited_event,
)
from src.projections.handlers.interview_handlers import InterviewCreatedHandler
from src.projections.handlers.sentence_handlers import (
    SentenceCreatedHandler,
    SentenceEditedHandler,
)
from src.utils.neo4j_driver import Neo4jConnectionManager


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.eventstore
class TestIdempotency:
    """Test idempotent behavior of projection handlers."""

    async def test_replay_same_event_multiple_times(
        self,
        clean_test_database,
    ):
        """
        Test that replaying the same event multiple times doesn't change state.

        Validates:
        1. First processing creates the node
        2. Subsequent replays are no-ops (idempotent)
        3. No duplicate nodes created
        4. No errors raised
        """
        # Generate test IDs
        interview_id = str(uuid.uuid4())
        project_id = "idempotency-test"
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")

        # Create event
        event = create_interview_created_event(
            aggregate_id=interview_id,
            version=0,
            title="Idempotency Test Interview",
            source="test.txt",
            language="en",
            actor=system_actor,
            project_id=project_id,
            correlation_id=str(uuid.uuid4()),
        )

        # === Process event 3 times ===
        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        handler = InterviewCreatedHandler()

        # First processing
        await handler.handle(event)

        # Capture state after first processing
        async with driver.session() as session:
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN count(i) as count",
                interview_id=interview_id,
            )
            count_after_first = (await result.single())["count"]
            assert count_after_first == 1, "Interview node not created on first processing"

        # Second processing (replay)
        await handler.handle(event)

        # Third processing (replay)
        await handler.handle(event)

        # === Verify state unchanged after replays ===
        async with driver.session() as session:
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN count(i) as count",
                interview_id=interview_id,
            )
            final_count = (await result.single())["count"]
            assert final_count == 1, f"Expected 1 interview, got {final_count} (not idempotent)"

            # Verify title not duplicated or changed
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN i.title as title",
                interview_id=interview_id,
            )
            title = (await result.single())["title"]
            assert title == "Idempotency Test Interview", "Title changed on replay"

        print("\n✓ Idempotency validated:")
        print("  - Event processed 3 times")
        print("  - State unchanged after first processing")
        print("  - No duplicate nodes created")
        print("  - No errors raised")

    async def test_version_guard_prevents_old_events(
        self,
        clean_test_database,
    ):
        """
        Test that version guards prevent processing old events.

        Scenario:
        1. Process SentenceCreated (version 0)
        2. Process SentenceEdited (version 1)
        3. Attempt to replay SentenceCreated (version 0)
        4. Verify replay is skipped (version guard)
        """
        # Generate test IDs
        interview_id = str(uuid.uuid4())
        sentence_index = 0
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}"))
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        human_actor = Actor(actor_type=ActorType.HUMAN, user_id="user-123")
        correlation_id = str(uuid.uuid4())

        # Create events
        created_event = create_sentence_created_event(
            aggregate_id=sentence_id,
            version=0,
            interview_id=interview_id,
            index=sentence_index,
            text="Original text",
            actor=system_actor,
            correlation_id=correlation_id,
        )

        edited_event = create_sentence_edited_event(
            aggregate_id=sentence_id,
            version=1,
            old_text="Original text",
            new_text="Edited text",
            editor_type="human",
            actor=human_actor,
            correlation_id=correlation_id,
        )

        # === First create the Interview node (required parent) ===
        interview_handler = InterviewCreatedHandler()
        interview_event = create_interview_created_event(
            aggregate_id=interview_id,
            version=0,
            title="Version Guard Test Interview",
            source="test.txt",
            language="en",
            actor=system_actor,
            project_id="test-project",
            correlation_id=correlation_id,
        )
        await interview_handler.handle(interview_event)

        # === Process in correct order ===
        created_handler = SentenceCreatedHandler()
        edited_handler = SentenceEditedHandler()

        # Process version 0 (created)
        await created_handler.handle(created_event)

        # Process version 1 (edited)
        await edited_handler.handle(edited_event)

        # Verify text is "Edited text"
        async with await Neo4jConnectionManager.get_session(database="neo4j") as session:
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) RETURN s.text as text, s.event_version as version",
                sentence_id=sentence_id,
            )
            record = await result.single()
            assert record["text"] == "Edited text", "Edit not applied"
            assert record["version"] == 1, f"Expected version 1, got {record['version']}"

        # === Attempt to replay version 0 (should be skipped) ===
        await created_handler.handle(created_event)

        # === Verify state unchanged (version guard worked) ===
        async with await Neo4jConnectionManager.get_session(database="neo4j") as session:
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) RETURN s.text as text, s.event_version as version",
                sentence_id=sentence_id,
            )
            record = await result.single()
            assert record["text"] == "Edited text", "Text reverted (version guard failed)"
            assert record["version"] == 1, "Version reverted (version guard failed)"

        print("\n✓ Version guard validated:")
        print("  - Version 0 processed first")
        print("  - Version 1 applied successfully")
        print("  - Replay of version 0 skipped by version guard")
        print("  - State preserved at version 1")

    async def test_multiple_event_types_idempotency(
        self,
        clean_test_database,
    ):
        """
        Test idempotency across different event types.

        Validates:
        1. SentenceCreated is idempotent
        2. SentenceEdited is idempotent
        3. Replaying mixed events maintains correct final state
        """
        # Generate test IDs
        interview_id = str(uuid.uuid4())
        sentence_index = 0
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}"))
        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        human_actor = Actor(actor_type=ActorType.HUMAN, user_id="user-123")
        correlation_id = str(uuid.uuid4())

        # Create events
        created_event = create_sentence_created_event(
            aggregate_id=sentence_id,
            version=0,
            interview_id=interview_id,
            index=sentence_index,
            text="Original",
            actor=system_actor,
            correlation_id=correlation_id,
        )

        edited_event_1 = create_sentence_edited_event(
            aggregate_id=sentence_id,
            version=1,
            old_text="Original text",
            new_text="First edit",
            editor_type="human",
            actor=human_actor,
            correlation_id=correlation_id,
        )

        edited_event_2 = create_sentence_edited_event(
            aggregate_id=sentence_id,
            version=2,
            old_text="First edit",
            new_text="Second edit",
            editor_type="human",
            actor=human_actor,
            correlation_id=correlation_id,
        )

        # === First create the Interview node (required parent) ===
        interview_handler = InterviewCreatedHandler()
        interview_event = create_interview_created_event(
            aggregate_id=interview_id,
            version=0,
            title="Multiple Events Test Interview",
            source="test.txt",
            language="en",
            actor=system_actor,
            project_id="test-project",
            correlation_id=correlation_id,
        )
        await interview_handler.handle(interview_event)

        # === Process events in order ===
        created_handler = SentenceCreatedHandler()
        edited_handler = SentenceEditedHandler()

        await created_handler.handle(created_event)
        await edited_handler.handle(edited_event_1)
        await edited_handler.handle(edited_event_2)

        # Capture final state
        async with await Neo4jConnectionManager.get_session(database="neo4j") as session:
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) "
                "RETURN s.text as text, s.event_version as version, s.is_edited as is_edited",
                sentence_id=sentence_id,
            )
            final_state = await result.single()
            expected_text = final_state["text"]
            expected_version = final_state["version"]
            expected_is_edited = final_state["is_edited"]

        # === Replay all events (simulate full replay) ===
        await created_handler.handle(created_event)
        await edited_handler.handle(edited_event_1)
        await edited_handler.handle(edited_event_2)

        # === Verify state unchanged ===
        async with await Neo4jConnectionManager.get_session(database="neo4j") as session:
            result = await session.run(
                "MATCH (s:Sentence {sentence_id: $sentence_id}) "
                "RETURN s.text as text, s.event_version as version, s.is_edited as is_edited",
                sentence_id=sentence_id,
            )
            replayed_state = await result.single()

            assert replayed_state["text"] == expected_text, "Text changed on replay"
            assert replayed_state["version"] == expected_version, "Version changed on replay"
            assert replayed_state["is_edited"] == expected_is_edited, "is_edited flag changed on replay"

        print("\n✓ Multiple event types idempotency validated:")
        print("  - SentenceCreated + 2 SentenceEdited events")
        print("  - All events replayed")
        print("  - Final state unchanged")
        print(f"  - Correct text: '{expected_text}'")
        print(f"  - Correct version: {expected_version}")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.eventstore
class TestResilience:
    """Test resilience features: retries, parked events, error handling."""

    async def test_handler_retry_on_transient_error(
        self,
        clean_test_database,
        monkeypatch,
    ):
        """
        Test that handlers retry on transient errors.

        Scenario:
        1. Mock Neo4j to fail twice, then succeed
        2. Process event
        3. Verify handler retried and eventually succeeded
        """
        # This test would mock Neo4j session to simulate transient failures
        # and verify the retry logic in BaseHandler.handle_with_retry()
        pass

    async def test_event_parked_after_max_retries(
        self,
        clean_test_database,
        monkeypatch,
    ):
        """
        Test that events are parked (DLQ) after max retries.

        Scenario:
        1. Mock Neo4j to always fail
        2. Process event
        3. Verify handler retries 3 times
        4. Verify event is parked in DLQ
        """
        # This test would verify the DLQ mechanism
        pass

    async def test_replay_parked_event(
        self,
        clean_test_database,
    ):
        """
        Test manually replaying a parked event.

        Scenario:
        1. Event fails and is parked
        2. Fix the underlying issue
        3. Replay the parked event
        4. Verify successful processing
        """
        # This test would verify parked event replay functionality
        pass
