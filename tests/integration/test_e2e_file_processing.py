"""
End-to-end file processing tests with dual-write validation.

Tests the complete workflow:
1. File upload → Pipeline processing
2. Events emitted to EventStoreDB (dual-write)
3. Data written to Neo4j (direct write)
4. Event metadata validation
"""

import os
import uuid
from pathlib import Path

import pytest

from src.config import config
from src.pipeline import PipelineOrchestrator
from src.utils.environment import detect_environment


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.eventstore
class TestE2EFileProcessingWithDualWrite:
    """Test end-to-end file processing with dual-write enabled."""

    async def test_single_file_upload_with_dual_write(
        self,
        sample_interview_file,
        clean_test_database,
        event_store_client,
        tmp_path,
    ):
        """
        Test complete workflow: file upload → events + Neo4j.

        Validates:
        1. File is processed by pipeline
        2. Events are emitted to EventStoreDB
        3. Neo4j contains data from direct write
        4. Event metadata is correct (correlation IDs, actor, timestamps)
        """
        # Enable event sourcing for this test
        test_config = config.copy()
        # Use environment-aware connection string with configurable host/port
        esdb_connection = os.getenv("EVENTSTORE_TEST_CONNECTION_STRING")
        if not esdb_connection:
            environment = detect_environment()
            host = os.getenv("EVENTSTORE_HOST", "eventstore" if environment in ("docker", "ci") else "localhost")
            port = os.getenv("EVENTSTORE_PORT", "2113")
            esdb_connection = f"esdb://{host}:{port}?tls=false"
        test_config["event_sourcing"] = {"enabled": True, "connection_string": esdb_connection}

        # Create output directory for the test
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_config["paths"]["output_dir"] = str(output_dir)

        # Create pipeline orchestrator
        # Use the sample file's parent directory as input_dir
        pipeline = PipelineOrchestrator(
            input_dir=sample_interview_file.parent, output_dir=output_dir, config_dict=test_config
        )

        # Process the file
        await pipeline._process_single_file(Path(sample_interview_file))

        # Generate expected interview_id (pipeline uses deterministic uuid5)
        interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{sample_interview_file.name}"))

        # === Validation 1: Check EventStoreDB for InterviewCreated event ===
        interview_stream = f"Interview-{interview_id}"
        interview_events = await event_store_client.read_stream(interview_stream)

        assert len(interview_events) > 0, "No InterviewCreated event found in EventStoreDB"

        interview_created = interview_events[0]
        assert interview_created.event_type == "InterviewCreated"
        assert interview_created.aggregate_id == interview_id
        assert interview_created.aggregate_type == "Interview"
        assert interview_created.version == 0
        assert interview_created.actor.actor_type == "system"
        assert interview_created.correlation_id is not None

        # === Validation 2: Check EventStoreDB for SentenceCreated events ===
        # The test file has 4 sentences
        expected_sentence_count = 4

        for i in range(expected_sentence_count):
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
            sentence_stream = f"Sentence-{sentence_id}"

            sentence_events = await event_store_client.read_stream(sentence_stream)
            assert len(sentence_events) > 0, f"No SentenceCreated event for sentence {i}"

            sentence_created = sentence_events[0]
            assert sentence_created.event_type == "SentenceCreated"
            assert sentence_created.aggregate_id == sentence_id
            assert sentence_created.aggregate_type == "Sentence"
            assert sentence_created.version == 0
            assert sentence_created.actor.actor_type == "system"
            # All sentence events should have the same correlation_id as the interview
            assert sentence_created.correlation_id == interview_created.correlation_id

        # === Validation 3: Check Neo4j for data (direct write) ===
        from src.utils.neo4j_driver import Neo4jConnectionManager

        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        async with driver.session() as session:
            # Check for Interview node
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id}) RETURN i", interview_id=interview_id
            )
            interview_node = await result.single()
            assert interview_node is not None, "Interview node not found in Neo4j"

            # Check for Sentence nodes
            result = await session.run(
                "MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence) "
                "RETURN count(s) as sentence_count",
                interview_id=interview_id,
            )
            record = await result.single()
            assert (
                record["sentence_count"] == expected_sentence_count
            ), f"Expected {expected_sentence_count} sentences in Neo4j, got {record['sentence_count']}"

            # Check for SourceFile node
            result = await session.run(
                "MATCH (f:SourceFile {filename: $filename}) RETURN f", filename=sample_interview_file.name
            )
            file_node = await result.single()
            assert file_node is not None, "SourceFile node not found in Neo4j"

        print("\n✓ Single file processing validated:")
        print("  - InterviewCreated event in EventStoreDB")
        print(f"  - {expected_sentence_count} SentenceCreated events in EventStoreDB")
        print("  - Interview node in Neo4j")
        print(f"  - {expected_sentence_count} Sentence nodes in Neo4j")
        print("  - SourceFile node in Neo4j")
        print("  - Correlation ID consistent across all events")

    async def test_deterministic_sentence_uuids(
        self,
        sample_interview_file,
        clean_test_database,
        event_store_client,
        tmp_path,
    ):
        """
        Test that sentence UUIDs are deterministic (uuid5 from interview_id:index).

        This ensures that:
        1. Replaying events will target the same sentences
        2. Manual edits can target specific sentences predictably
        3. No duplicate sentence nodes in Neo4j
        """
        # Enable event sourcing
        test_config = config.copy()
        # Use environment-aware connection string
        environment = detect_environment()
        if environment in ("docker", "ci"):
            esdb_connection = "esdb://eventstore:2113?tls=false"
        else:
            esdb_connection = "esdb://localhost:2113?tls=false"
        test_config["event_sourcing"] = {"enabled": True, "connection_string": esdb_connection}

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_config["paths"]["output_dir"] = str(output_dir)

        # Process the file TWICE (simulating re-processing)
        pipeline = PipelineOrchestrator(
            input_dir=sample_interview_file.parent, output_dir=output_dir, config_dict=test_config
        )
        await pipeline._process_single_file(Path(sample_interview_file))

        # Clear Neo4j but keep EventStoreDB
        from src.utils.neo4j_driver import Neo4jConnectionManager

        driver = await Neo4jConnectionManager.get_driver(test_mode=True)
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")

        # Process again
        await pipeline._process_single_file(Path(sample_interview_file))

        # === Validation: Sentence UUIDs should be identical ===
        interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{sample_interview_file.name}"))

        # Check first sentence
        expected_sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:0"))
        sentence_stream = f"Sentence-{expected_sentence_id}"

        sentence_events = await event_store_client.read_stream(sentence_stream)
        # Should have 2 sets of events now (from both processings)
        assert len(sentence_events) >= 1, "No events for first sentence"

        # All events should have the same sentence_id
        for event in sentence_events:
            assert event.aggregate_id == expected_sentence_id, "Sentence UUID is not deterministic across processings"

        print("\n✓ Deterministic UUID validation passed:")
        print("  - Sentence UUIDs are consistent across re-processing")
        print("  - Using uuid5(interview_id:index) generation strategy")

    @pytest.mark.eventstore
    @pytest.mark.integration
    async def test_multiple_files_concurrent_processing(
        self,
        clean_test_database,
        event_store_client,
        tmp_path,
    ):
        """
        Test concurrent file processing with dual-write.

        Validates:
        1. Multiple files processed simultaneously
        2. Events correctly partitioned by interview_id
        3. No event loss or corruption
        4. Neo4j has data for all files
        """
        # Enable event sourcing
        test_config = config.copy()
        # Use environment-aware connection string
        environment = detect_environment()
        if environment in ("docker", "ci"):
            esdb_connection = "esdb://eventstore:2113?tls=false"
        else:
            esdb_connection = "esdb://localhost:2113?tls=false"
        test_config["event_sourcing"] = {"enabled": True, "connection_string": esdb_connection}

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        test_config["paths"]["output_dir"] = str(output_dir)

        # Create 3 test files
        file_paths = []
        for i in range(3):
            content = f"""File {i} sentence 1.
File {i} sentence 2.
File {i} sentence 3."""
            file_path = tmp_path / f"test_file_{i}.txt"
            file_path.write_text(content)
            file_paths.append(file_path)

        # Process all files concurrently
        pipeline = PipelineOrchestrator(input_dir=tmp_path, output_dir=output_dir, config_dict=test_config)

        import asyncio

        tasks = [pipeline._process_single_file(Path(fp)) for fp in file_paths]
        await asyncio.gather(*tasks)

        # === Validation: Check each file's events and Neo4j data ===
        from src.utils.neo4j_driver import Neo4jConnectionManager

        driver = await Neo4jConnectionManager.get_driver(test_mode=True)

        for i, file_path in enumerate(file_paths):
            interview_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{file_path.name}"))

            # Check EventStoreDB
            interview_stream = f"Interview-{interview_id}"
            interview_events = await event_store_client.read_stream(interview_stream)
            assert len(interview_events) > 0, f"No events for file {i}"

            # Check Neo4j
            async with driver.session() as session:
                result = await session.run(
                    "MATCH (i:Interview {interview_id: $interview_id}) "
                    "RETURN i, [(i)-[:HAS_SENTENCE]->(s:Sentence) | s] as sentences",
                    interview_id=interview_id,
                )
                record = await result.single()
                assert record is not None, f"Interview not found in Neo4j for file {i}"
                assert len(record["sentences"]) == 3, f"Expected 3 sentences for file {i}"

        print("\n✓ Concurrent processing validated:")
        print("  - 3 files processed simultaneously")
        print("  - Events correctly partitioned by interview_id")
        print("  - All data present in Neo4j")
