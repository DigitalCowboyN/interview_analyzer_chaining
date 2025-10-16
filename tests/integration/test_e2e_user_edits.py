"""
End-to-end user edit workflow tests.

Tests the complete user edit workflow:
1. File upload → Pipeline creates initial data
2. User edits sentence via API → Event emitted
3. User overrides analysis via API → Event emitted
4. Events accessible via history API

Note: Projection service updates to Neo4j are tested separately.
This focuses on the command/event flow.
"""

import uuid

import pytest
from httpx import AsyncClient

from src.main import app
from src.config import config
from src.events.store import EventStoreClient


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.eventstore
class TestE2EUserEditWorkflow:
    """Test end-to-end user edit workflows via API."""

    async def test_edit_sentence_workflow(
        self,
        event_store_client,
    ):
        """
        Test sentence edit workflow via API.

        Workflow:
        1. User calls edit API to change sentence text
        2. Verify SentenceEdited event in EventStoreDB
        3. Verify edit history accessible via API
        """
        # Generate test IDs
        interview_id = str(uuid.uuid4())
        sentence_index = 0
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}"))

        # Create initial sentence via event (simulating pipeline)
        from src.events.sentence_events import create_sentence_created_event
        from src.events.envelope import Actor, ActorType

        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        initial_event = create_sentence_created_event(
            aggregate_id=sentence_id,
            version=0,
            interview_id=interview_id,
            index=sentence_index,
            text="Original sentence text.",
            actor=system_actor,
            correlation_id=str(uuid.uuid4()),
        )

        # Append initial event to EventStoreDB
        stream_name = f"Sentence-{sentence_id}"
        await event_store_client.append_events(
            stream_name=stream_name,
            events=[initial_event],
            expected_version=-1,  # New stream
        )

        # === Step 1: User edits sentence via API ===
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.put(
                f"/api/v1/edits/{interview_id}/sentences/{sentence_index}",
                json={"new_text": "Edited sentence text."},
                headers={"X-User-ID": "test-user-123", "X-Correlation-ID": str(uuid.uuid4())},
            )

        # API should return 202 Accepted
        assert response.status_code == 202, f"Expected 202, got {response.status_code}: {response.text}"
        result = response.json()
        assert result["sentence_id"] == sentence_id
        assert result["version"] == 1  # New event version

        # === Step 2: Verify SentenceEdited event in EventStoreDB ===
        events = await event_store_client.read_stream(stream_name)
        assert len(events) == 2, f"Expected 2 events (created + edited), got {len(events)}"

        edited_event = events[1]
        assert edited_event.event_type == "SentenceEdited"
        assert edited_event.aggregate_id == sentence_id
        assert edited_event.version == 1
        assert edited_event.actor.actor_type == "human"
        assert edited_event.actor.user_id == "test-user-123"
        assert edited_event.data["new_text"] == "Edited sentence text."

        # === Step 3: Verify edit history accessible via API ===
        async with AsyncClient(app=app, base_url="http://test") as client:
            history_response = await client.get(
                f"/api/v1/edits/{interview_id}/sentences/{sentence_index}/history"
            )

        assert history_response.status_code == 200
        history = history_response.json()
        assert len(history["events"]) == 2
        assert history["events"][0]["event_type"] == "SentenceCreated"
        assert history["events"][1]["event_type"] == "SentenceEdited"

        print("\n✓ Sentence edit workflow validated:")
        print("  - Edit API returned 202 Accepted")
        print("  - SentenceEdited event in EventStoreDB")
        print("  - Event metadata correct (actor, version)")
        print("  - History API returns both events")

    async def test_override_analysis_workflow(
        self,
        event_store_client,
    ):
        """
        Test analysis override workflow via API.

        Workflow:
        1. User calls override API to correct analysis
        2. Verify AnalysisOverridden event in EventStoreDB
        3. Verify override note saved
        """
        # Generate test IDs
        interview_id = str(uuid.uuid4())
        sentence_index = 0
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}"))

        # Create initial sentence and analysis via events (simulating pipeline)
        from src.events.sentence_events import (
            create_sentence_created_event,
            create_analysis_generated_event,
        )
        from src.events.envelope import Actor, ActorType

        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        correlation_id = str(uuid.uuid4())

        created_event = create_sentence_created_event(
            aggregate_id=sentence_id,
            version=0,
            interview_id=interview_id,
            index=sentence_index,
            text="Test sentence for analysis.",
            actor=system_actor,
            correlation_id=correlation_id,
        )

        analysis_event = create_analysis_generated_event(
            aggregate_id=sentence_id,
            version=1,
            model="gpt-4",
            model_version="2024-01",
            classification={"function_type": "question", "structure_type": "simple", "purpose": "inquiry"},
            keywords=["test", "analysis"],
            topics=["testing"],
            domain_keywords=[],
            confidence=0.95,
            actor=system_actor,
            correlation_id=correlation_id,
        )

        # Append initial events to EventStoreDB
        stream_name = f"Sentence-{sentence_id}"
        await event_store_client.append_events(
            stream_name=stream_name,
            events=[created_event, analysis_event],
            expected_version=-1,  # New stream
        )

        # === Step 1: User overrides analysis via API ===
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/edits/{interview_id}/sentences/{sentence_index}/analysis",
                json={
                    "classification": {"function_type": "statement"},
                    "keywords": ["corrected", "keywords"],
                    "override_note": "AI misclassified this as a question",
                },
                headers={"X-User-ID": "expert-123", "X-Correlation-ID": str(uuid.uuid4())},
            )

        # API should return 202 Accepted
        assert response.status_code == 202, f"Expected 202, got {response.status_code}: {response.text}"
        result = response.json()
        assert result["sentence_id"] == sentence_id
        assert result["version"] == 2  # New event version

        # === Step 2: Verify AnalysisOverridden event in EventStoreDB ===
        events = await event_store_client.read_stream(stream_name)
        assert len(events) == 3, f"Expected 3 events, got {len(events)}"

        override_event = events[2]
        assert override_event.event_type == "AnalysisOverridden"
        assert override_event.aggregate_id == sentence_id
        assert override_event.version == 2
        assert override_event.actor.actor_type == "human"
        assert override_event.actor.user_id == "expert-123"
        assert override_event.data["fields_overridden"]["classification"]["function_type"] == "statement"
        assert override_event.data["fields_overridden"]["keywords"] == ["corrected", "keywords"]
        assert override_event.data["override_note"] == "AI misclassified this as a question"

        print("\n✓ Analysis override workflow validated:")
        print("  - Override API returned 202 Accepted")
        print("  - AnalysisOverridden event in EventStoreDB")
        print("  - Event metadata correct (actor, version)")
        print("  - Override note saved in event")

    async def test_multiple_edits_on_same_sentence(
        self,
        event_store_client,
    ):
        """
        Test multiple sequential edits on the same sentence.

        Ensures:
        1. Version numbers increment correctly
        2. All events are persisted in order
        3. History reflects all changes
        """
        # Generate test IDs
        interview_id = str(uuid.uuid4())
        sentence_index = 0
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}"))

        # Create initial sentence
        from src.events.sentence_events import create_sentence_created_event
        from src.events.envelope import Actor, ActorType

        system_actor = Actor(actor_type=ActorType.SYSTEM, user_id="pipeline")
        initial_event = create_sentence_created_event(
            aggregate_id=sentence_id,
            version=0,
            interview_id=interview_id,
            index=sentence_index,
            text="Original text.",
            actor=system_actor,
            correlation_id=str(uuid.uuid4()),
        )

        stream_name = f"Sentence-{sentence_id}"
        await event_store_client.append_events(
            stream_name=stream_name,
            events=[initial_event],
            expected_version=-1,
        )

        # === Perform 3 sequential edits ===
        edits = [
            "First edit.",
            "Second edit.",
            "Third edit.",
        ]

        async with AsyncClient(app=app, base_url="http://test") as client:
            for i, new_text in enumerate(edits, start=1):
                response = await client.put(
                    f"/api/v1/edits/{interview_id}/sentences/{sentence_index}",
                    json={"new_text": new_text},
                    headers={"X-User-ID": f"user-{i}"},
                )
                assert response.status_code == 202
                result = response.json()
                assert result["version"] == i, f"Expected version {i}, got {result['version']}"

        # === Verify all events in EventStoreDB ===
        events = await event_store_client.read_stream(stream_name)
        assert len(events) == 4, f"Expected 4 events (1 created + 3 edits), got {len(events)}"

        # Check version sequence
        for i, event in enumerate(events):
            assert event.version == i, f"Event {i} has incorrect version: {event.version}"

        # === Verify history API returns all events ===
        async with AsyncClient(app=app, base_url="http://test") as client:
            history_response = await client.get(
                f"/api/v1/edits/{interview_id}/sentences/{sentence_index}/history"
            )

        assert history_response.status_code == 200
        history = history_response.json()
        assert len(history["events"]) == 4

        print("\n✓ Multiple edits workflow validated:")
        print("  - 3 sequential edits processed successfully")
        print("  - Version numbers increment correctly (0 → 1 → 2 → 3)")
        print("  - All events persisted in EventStoreDB")
        print("  - History API returns all 4 events")

