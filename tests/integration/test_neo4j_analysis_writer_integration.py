# tests/integration/test_neo4j_analysis_writer_integration.py
"""
Integration tests that use the real Neo4jAnalysisWriter instead of mocked AnalysisService.
These tests verify the complete flow from pipeline execution to Neo4j storage.

M2.8: Updated to use event-first dual-write pattern with projection service.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from src.io.neo4j_analysis_writer import Neo4jAnalysisWriter
from src.io.neo4j_map_storage import Neo4jMapStorage
from src.pipeline import PipelineOrchestrator
from src.pipeline_event_emitter import PipelineEventEmitter
from src.projections.handlers.interview_handlers import InterviewCreatedHandler
from src.projections.handlers.sentence_handlers import (
    SentenceCreatedHandler,
    AnalysisGeneratedHandler,
)
from src.utils.neo4j_driver import Neo4jConnectionManager

# Mark all tests in this module as asyncio and require Neo4j
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def process_events_through_projection(
    event_store,
    interview_id: str,
    num_sentences: int,
    process_analyses: bool = True,
):
    """
    M2.8 Helper: Process events through projection service handlers.

    Args:
        event_store: EventStoreClient instance
        interview_id: Interview UUID
        num_sentences: Number of sentences to process
        process_analyses: Whether to process AnalysisGenerated events
    """
    import logging
    logger = logging.getLogger(__name__)

    # Process InterviewCreated event
    interview_handler = InterviewCreatedHandler()
    interview_stream = f"Interview-{interview_id}"
    interview_events = await event_store.read_stream(interview_stream)

    logger.info(f"[DEBUG] Reading {interview_stream}, got {len(interview_events)} events")

    for event in interview_events:
        if event.event_type == "InterviewCreated":
            logger.info(f"[DEBUG] Processing InterviewCreated v{event.version}")
            await interview_handler.handle(event)
            logger.info(f"[DEBUG] Completed InterviewCreated v{event.version}")

    # Process SentenceCreated and optionally AnalysisGenerated events
    sentence_handler = SentenceCreatedHandler()
    analysis_handler = AnalysisGeneratedHandler() if process_analyses else None

    for i in range(num_sentences):
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{i}"))
        sentence_stream = f"Sentence-{sentence_id}"
        sentence_events = await event_store.read_stream(sentence_stream)

        logger.info(f"[DEBUG] Reading {sentence_stream}, got {len(sentence_events)} events")

        for idx, event in enumerate(sentence_events):
            logger.info(f"[DEBUG] Event {idx+1}/{len(sentence_events)}: {event.event_type} v{event.version} for {event.aggregate_id[:8]}...")

            if event.event_type == "SentenceCreated":
                logger.info(f"[DEBUG] Processing SentenceCreated v{event.version}")
                await sentence_handler.handle(event)
                logger.info(f"[DEBUG] Completed SentenceCreated v{event.version}")
            elif event.event_type == "AnalysisGenerated" and analysis_handler:
                logger.info(f"[DEBUG] Processing AnalysisGenerated v{event.version}")
                await analysis_handler.handle(event)
                logger.info(f"[DEBUG] Completed AnalysisGenerated v{event.version}")


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Helper function to load JSONL files."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class TestNeo4jAnalysisWriterIntegration:
    """Integration tests for Neo4jAnalysisWriter with real pipeline execution."""

    @pytest.fixture
    def mock_config(self):
        """Provides a mock configuration for testing."""
        return {
            "paths": {
                "output_dir": "test_output",
                "map_dir": "test_maps",
                "map_suffix": "_map.jsonl",
                "analysis_suffix": "_analysis.jsonl",
                "logs_dir": "test_logs",
            },
            "pipeline": {
                "num_concurrent_files": 1,
                "default_cardinality_limits": {
                    "HAS_FUNCTION": 1,
                    "HAS_STRUCTURE": 1,
                    "HAS_PURPOSE": 1,
                    "MENTIONS_KEYWORD": 6,
                    "MENTIONS_TOPIC": None,  # Unlimited
                    "MENTIONS_DOMAIN_KEYWORD": None,  # Unlimited
                },
            },
            "preprocessing": {"context_windows": {"immediate": 1, "broader": 3, "observer": 5}},
            "classification": {"local": {"prompt_files": {"no_context": "dummy"}}},
            "domain_keywords": [],
            "openai": {"model_name": "gpt-4"},
        }

    @pytest.fixture
    def integration_dirs(self, tmp_path):
        """Creates temporary directories for integration testing."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        map_dir = tmp_path / "maps"
        input_dir.mkdir()
        output_dir.mkdir()
        map_dir.mkdir()
        return input_dir, output_dir, map_dir

    @pytest.fixture
    def test_project_and_interview_ids(self):
        """Provides test project and interview IDs."""
        return str(uuid.uuid4()), str(uuid.uuid4())

    async def test_pipeline_with_neo4j_analysis_writer_success(
        self, integration_dirs, mock_config, test_project_and_interview_ids, clean_test_database, clean_event_store
    ):
        """M2.8: Test complete pipeline execution using Neo4jAnalysisWriter with event-first dual-write."""
        input_dir, output_dir, map_dir = integration_dirs
        project_id, interview_id = test_project_and_interview_ids

        # M2.8: Create event emitter for dual-write
        event_emitter = PipelineEventEmitter(clean_event_store)

        # Create test input file
        test_file = input_dir / "test_interview.txt"
        test_content = "Hello world. How are you today? This is a test sentence."
        test_file.write_text(test_content)

        # Mock the sentence analysis to return predictable results
        mock_analysis_results = [
            {
                "sentence_id": 0,
                "sequence_order": 0,
                "sentence": "Hello world.",
                "function_type": "greeting",
                "structure_type": "simple",
                "purpose": "social_interaction",
                "topics": ["communication", "casual_greeting"],
                "overall_keywords": ["hello", "greeting"],
                "domain_keywords": ["social"],
            },
            {
                "sentence_id": 1,
                "sequence_order": 1,
                "sentence": "How are you today?",
                "function_type": "question",
                "structure_type": "interrogative",
                "purpose": "inquiry",
                "topics": ["communication", "personal_inquiry"],
                "overall_keywords": ["question", "wellbeing"],
                "domain_keywords": ["personal"],
            },
            {
                "sentence_id": 2,
                "sequence_order": 2,
                "sentence": "This is a test sentence.",
                "function_type": "statement",
                "structure_type": "declarative",
                "purpose": "information",
                "topics": ["testing", "test_data"],
                "overall_keywords": ["test", "sentence"],
                "domain_keywords": ["testing"],
            },
        ]

        # Create a custom orchestrator that uses Neo4jAnalysisWriter
        class TestOrchestrator(PipelineOrchestrator):
            def __init__(self, *args, event_emitter_override=None, **kwargs):
                self.project_id = project_id
                self.interview_id = interview_id
                super().__init__(*args, **kwargs)
                # M2.8: Override event_emitter after parent initialization
                if event_emitter_override:
                    self.event_emitter = event_emitter_override

            def _setup_file_io(self, file_path: Path, interview_id: str = None, project_id: str = None, correlation_id: str = None):
                """M2.8: Override to use Neo4jAnalysisWriter with event-first dual-write."""
                from src.io.local_storage import LocalTextDataSource
                from src.utils.path_helpers import generate_pipeline_paths

                # Generate paths for map storage (still using local for maps)
                paths = generate_pipeline_paths(
                    input_file=file_path,
                    map_dir=self.map_dir_path,
                    output_dir=self.output_dir_path,
                    map_suffix=self.map_suffix,
                    analysis_suffix=self.analysis_suffix,
                    task_id=self.task_id,
                )

                # Use provided IDs or fall back to test's IDs
                actual_interview_id = interview_id or self.interview_id
                actual_project_id = project_id or self.project_id

                # Use local storage for data source and map storage
                data_source = LocalTextDataSource(file_path)
                map_storage = Neo4jMapStorage(actual_project_id, actual_interview_id, event_emitter=self.event_emitter, correlation_id=correlation_id)

                # M2.8: Use Neo4jAnalysisWriter with event_emitter for dual-write
                analysis_writer = Neo4jAnalysisWriter(actual_project_id, actual_interview_id, event_emitter=self.event_emitter)

                return data_source, map_storage, analysis_writer, paths

        # Mock the sentence analyzer to return our test data
        with patch(
            "src.agents.sentence_analyzer.SentenceAnalyzer.classify_sentence", new_callable=AsyncMock
        ) as mock_classify:
            # Configure mock to return analysis results without sentence metadata
            def mock_classify_side_effect(sentence, *args, **kwargs):
                for result in mock_analysis_results:
                    if result["sentence"] == sentence:
                        # Return only the analysis fields, not the sentence metadata
                        return {
                            k: v
                            for k, v in result.items()
                            if k not in ["sentence_id", "sequence_order", "sentence", "filename"]
                        }
                return {
                    "function_type": "unknown",
                    "structure_type": "unknown",
                    "purpose": "unknown",
                    "topics": [],
                    "overall_keywords": [],
                    "domain_keywords": [],
                }

            mock_classify.side_effect = mock_classify_side_effect

            # Create and execute the test orchestrator
            orchestrator = TestOrchestrator(
                input_dir=input_dir,
                output_dir=output_dir,
                map_dir=map_dir,
                config_dict=mock_config,
                task_id="test-neo4j-integration",
                event_emitter_override=event_emitter,  # M2.8: Inject event emitter
            )

            await orchestrator.execute(specific_file=test_file.name)

        # M2.8: Process events through projection service to populate Neo4j
        await process_events_through_projection(clean_event_store, interview_id, num_sentences=3, process_analyses=True)

        # M2.8: Verify the results were stored in Neo4j via projection service
        async with await Neo4jConnectionManager.get_session() as session:
            # M2.8: Use the interview_id we provided to the orchestrator
            # Check that sentences were stored
            sentences_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN s.sentence_id as sentence_id, s.text as text, s.sequence_order as sequence_order
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )

            sentences = []
            async for record in sentences_result:
                sentences.append(
                    {
                        "sentence_id": record["sentence_id"],
                        "text": record["text"],
                        "sequence_order": record["sequence_order"],
                    }
                )

            assert len(sentences) == 3
            assert sentences[0]["text"] == "Hello world."
            assert sentences[1]["text"] == "How are you today?"
            assert sentences[2]["text"] == "This is a test sentence."

            # Check that analysis nodes were created
            analysis_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN count(a) as analysis_count
                """,
                interview_id=interview_id,
            )
            analysis_record = await analysis_result.single()
            assert analysis_record is not None
            analysis_count = analysis_record["analysis_count"]
            assert analysis_count == 3

            # Check that dimension relationships were created
            # Check function types
            function_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:HAS_FUNCTION]->(f:FunctionType)
                RETURN f.name as function_name, s.sequence_order as sequence_order
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )

            functions = []
            async for record in function_result:
                functions.append({"sequence_order": record["sequence_order"], "function_name": record["function_name"]})

            assert len(functions) == 3
            assert functions[0]["function_name"] == "greeting"
            assert functions[1]["function_name"] == "question"
            assert functions[2]["function_name"] == "statement"

            # Check keywords (should respect cardinality limits)
            keyword_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
                RETURN k.text as keyword, s.sequence_order as sequence_order
                ORDER BY s.sequence_order, k.text
                """,
                interview_id=interview_id,
            )

            keywords = []
            async for record in keyword_result:
                keywords.append({"sequence_order": record["sequence_order"], "keyword": record["keyword"]})

            # Should have keywords for all sentences (limited by cardinality)
            assert len(keywords) > 0

            # M2.8: Verify that keywords respect cardinality limits (max 6 per sentence)
            # Note: Cardinality is enforced at event emission time
            keyword_counts = {}
            for kw in keywords:
                sequence_order = kw["sequence_order"]
                keyword_counts[sequence_order] = keyword_counts.get(sequence_order, 0) + 1

            for sequence_order, count in keyword_counts.items():
                assert count <= 6, f"Sentence {sequence_order} has {count} keywords, exceeding limit of 6"

    async def test_pipeline_with_neo4j_analysis_writer_error_handling(
        self, integration_dirs, mock_config, test_project_and_interview_ids, clean_test_database, clean_event_store
    ):
        """M2.8: Test that analysis errors are properly handled with event-first dual-write."""
        input_dir, output_dir, map_dir = integration_dirs
        project_id, interview_id = test_project_and_interview_ids

        # M2.8: Create event emitter for dual-write
        event_emitter = PipelineEventEmitter(clean_event_store)

        # Create test input file
        test_file = input_dir / "test_error.txt"
        test_content = "Good sentence. Bad sentence that will fail."
        test_file.write_text(test_content)

        # Create a custom orchestrator that uses Neo4jAnalysisWriter
        class TestOrchestrator(PipelineOrchestrator):
            def __init__(self, *args, event_emitter_override=None, **kwargs):
                self.project_id = project_id
                self.interview_id = interview_id
                super().__init__(*args, **kwargs)
                # M2.8: Override event_emitter after parent initialization
                if event_emitter_override:
                    self.event_emitter = event_emitter_override

            def _setup_file_io(self, file_path: Path, interview_id: str = None, project_id: str = None, correlation_id: str = None):
                """M2.8: Override to use Neo4jAnalysisWriter with event-first dual-write."""
                from src.io.local_storage import LocalTextDataSource
                from src.utils.path_helpers import generate_pipeline_paths

                paths = generate_pipeline_paths(
                    input_file=file_path,
                    map_dir=self.map_dir_path,
                    output_dir=self.output_dir_path,
                    map_suffix=self.map_suffix,
                    analysis_suffix=self.analysis_suffix,
                    task_id=self.task_id,
                )

                # Use provided IDs or fall back to test's IDs
                actual_interview_id = interview_id or self.interview_id
                actual_project_id = project_id or self.project_id

                data_source = LocalTextDataSource(file_path)
                map_storage = Neo4jMapStorage(actual_project_id, actual_interview_id, event_emitter=self.event_emitter, correlation_id=correlation_id)
                # M2.8: Pass event_emitter for dual-write
                analysis_writer = Neo4jAnalysisWriter(actual_project_id, actual_interview_id, event_emitter=self.event_emitter)

                return data_source, map_storage, analysis_writer, paths

        # Mock the sentence analyzer to simulate an error for the second sentence
        with patch(
            "src.agents.sentence_analyzer.SentenceAnalyzer.classify_sentence", new_callable=AsyncMock
        ) as mock_classify:

            def mock_classify_side_effect(sentence, *args, **kwargs):
                if "Good sentence" in sentence:
                    return {
                        "function_type": "statement",
                        "structure_type": "simple",
                        "purpose": "information",
                        "topics": ["testing"],
                        "overall_keywords": ["good", "test"],
                        "domain_keywords": ["testing"],
                    }
                elif "Bad sentence" in sentence:
                    raise ValueError("Simulated analysis error")
                return {
                    "function_type": "unknown",
                    "structure_type": "unknown",
                    "purpose": "unknown",
                    "topics": [],
                    "overall_keywords": [],
                    "domain_keywords": [],
                }

            mock_classify.side_effect = mock_classify_side_effect

            # Create and execute the test orchestrator
            orchestrator = TestOrchestrator(
                input_dir=input_dir,
                output_dir=output_dir,
                map_dir=map_dir,
                config_dict=mock_config,
                task_id="test-neo4j-error-handling",
                event_emitter_override=event_emitter,  # M2.8: Inject event emitter
            )

            await orchestrator.execute(specific_file=test_file.name)

        # M2.8: Process events through projection service
        # Note: Only the successful sentence will have AnalysisGenerated event
        await process_events_through_projection(clean_event_store, interview_id, num_sentences=2, process_analyses=True)

        # Verify the results in Neo4j
        async with await Neo4jConnectionManager.get_session() as session:
            # Check that both sentences were stored (including the error one)
            sentences_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN count(s) as sentence_count
                """,
                interview_id=interview_id,
            )
            sentences_record = await sentences_result.single()
            assert sentences_record is not None
            sentence_count = sentences_record["sentence_count"]
            assert sentence_count == 2

            # M2.8: Check analysis nodes - should have one successful and one error
            # Note: In M2.8, errors don't create AnalysisGenerated events, so no Analysis node
            analysis_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN a.error_data as error_data, s.sequence_order as sequence_order
                ORDER BY s.sequence_order
                """,
                interview_id=interview_id,
            )

            analyses = []
            async for record in analysis_result:
                analyses.append({"sequence_order": record["sequence_order"], "error_data": record["error_data"]})

            # Should only have one successful analysis (sentence 0)
            # Errors don't generate events in M2.8, so no Analysis node for sentence 1
            assert len(analyses) == 1

            # First sentence should have no error data
            assert analyses[0]["sequence_order"] == 0
            assert analyses[0]["error_data"] is None

            # M2.8: Verify that sentence 1 exists but has no analysis node (error case)
            sentence1_analysis_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sequence_order: 1})
                OPTIONAL MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN s.sequence_order as sequence_order, count(a) as analysis_count
                """,
                interview_id=interview_id,
            )
            sentence1_record = await sentence1_analysis_result.single()
            assert sentence1_record is not None
            assert sentence1_record["sequence_order"] == 1
            assert sentence1_record["analysis_count"] == 0  # No analysis node for error

            # Check that successful sentence has dimension relationships
            function_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sequence_order: 0})
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:HAS_FUNCTION]->(f:FunctionType)
                RETURN f.name as function_name
                """,
                interview_id=interview_id,
            )
            function_record = await function_result.single()
            assert function_record is not None
            assert function_record["function_name"] == "statement"

            # Error sentences don't have analysis nodes, so no dimension relationships to check

    async def test_pipeline_with_neo4j_cardinality_limits(
        self, integration_dirs, mock_config, test_project_and_interview_ids, clean_test_database, clean_event_store
    ):
        """M2.8: Test that cardinality limits are properly enforced with event-first dual-write."""
        input_dir, output_dir, map_dir = integration_dirs
        project_id, interview_id = test_project_and_interview_ids

        # M2.8: Create event emitter for dual-write
        event_emitter = PipelineEventEmitter(clean_event_store)

        # Create test input file
        test_file = input_dir / "test_cardinality.txt"
        test_content = "This sentence has many keywords for testing cardinality limits."
        test_file.write_text(test_content)

        # Create a custom orchestrator that uses Neo4jAnalysisWriter
        class TestOrchestrator(PipelineOrchestrator):
            def __init__(self, *args, event_emitter_override=None, **kwargs):
                self.project_id = project_id
                self.interview_id = interview_id
                super().__init__(*args, **kwargs)
                # M2.8: Override event_emitter after parent initialization
                if event_emitter_override:
                    self.event_emitter = event_emitter_override

            def _setup_file_io(self, file_path: Path, interview_id: str = None, project_id: str = None, correlation_id: str = None):
                """Override to use Neo4jAnalysisWriter instead of LocalJsonlAnalysisWriter."""
                from src.io.local_storage import LocalTextDataSource
                from src.utils.path_helpers import generate_pipeline_paths

                paths = generate_pipeline_paths(
                    input_file=file_path,
                    map_dir=self.map_dir_path,
                    output_dir=self.output_dir_path,
                    map_suffix=self.map_suffix,
                    analysis_suffix=self.analysis_suffix,
                    task_id=self.task_id,
                )

                # Use provided IDs or fall back to test's IDs
                actual_interview_id = interview_id or self.interview_id
                actual_project_id = project_id or self.project_id

                data_source = LocalTextDataSource(file_path)
                map_storage = Neo4jMapStorage(actual_project_id, actual_interview_id, event_emitter=self.event_emitter, correlation_id=correlation_id)
                # M2.8: Pass event_emitter for dual-write
                analysis_writer = Neo4jAnalysisWriter(actual_project_id, actual_interview_id, event_emitter=self.event_emitter)

                return data_source, map_storage, analysis_writer, paths

        # Mock the sentence analyzer to return many keywords (more than the limit)
        with patch(
            "src.agents.sentence_analyzer.SentenceAnalyzer.classify_sentence", new_callable=AsyncMock
        ) as mock_classify:
            mock_classify.return_value = {
                "function_type": "statement",
                "structure_type": "complex",
                "purpose": "testing",
                "topics": ["testing", "cardinality_testing"],
                # Return 10 keywords (more than the limit of 6)
                "overall_keywords": [
                    "keyword1",
                    "keyword2",
                    "keyword3",
                    "keyword4",
                    "keyword5",
                    "keyword6",
                    "keyword7",
                    "keyword8",
                    "keyword9",
                    "keyword10",
                ],
                "domain_keywords": [
                    "domain1",
                    "domain2",
                    "domain3",
                    "domain4",
                    "domain5",
                    "domain6",
                    "domain7",
                    "domain8",  # 8 domain keywords (no limit)
                ],
            }

            # Create and execute the test orchestrator
            orchestrator = TestOrchestrator(
                input_dir=input_dir,
                output_dir=output_dir,
                map_dir=map_dir,
                config_dict=mock_config,
                task_id="test-neo4j-cardinality",
                event_emitter_override=event_emitter,  # M2.8: Inject event emitter
            )

            await orchestrator.execute(specific_file=test_file.name)

        # M2.8: Process events through projection service
        await process_events_through_projection(clean_event_store, interview_id, num_sentences=1, process_analyses=True)

        # M2.8: Verify cardinality limits are enforced (limits applied at event emission time)
        async with await Neo4jConnectionManager.get_session() as session:
            # M2.8: Use the interview_id we provided to the orchestrator
            # Check that only 6 keywords were stored (cardinality limit)
            keyword_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
                RETURN count(k) as keyword_count
                """,
                interview_id=interview_id,
            )
            keyword_record = await keyword_result.single()
            assert keyword_record is not None
            keyword_count = keyword_record["keyword_count"]
            assert keyword_count == 6, f"Expected 6 keywords due to cardinality limit, got {keyword_count}"

            # Check that all domain keywords were stored (no limit)
            domain_keyword_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:MENTIONS_DOMAIN_KEYWORD]->(dk:DomainKeyword)
                RETURN count(dk) as domain_keyword_count
                """,
                interview_id=interview_id,
            )
            domain_keyword_record = await domain_keyword_result.single()
            assert domain_keyword_record is not None
            domain_keyword_count = domain_keyword_record["domain_keyword_count"]
            assert domain_keyword_count == 8, f"Expected 8 domain keywords (no limit), got {domain_keyword_count}"

            # Check that single-value dimensions are enforced (only 1 function type)
            function_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:HAS_FUNCTION]->(f:FunctionType)
                RETURN count(f) as function_count
                """,
                interview_id=interview_id,
            )
            function_record = await function_result.single()
            assert function_record is not None
            function_count = function_record["function_count"]
            assert function_count == 1, f"Expected 1 function type (cardinality limit), got {function_count}"
