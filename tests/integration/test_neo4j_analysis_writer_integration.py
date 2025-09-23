# tests/integration/test_neo4j_analysis_writer_integration.py
"""
Integration tests that use the real Neo4jAnalysisWriter instead of mocked AnalysisService.
These tests verify the complete flow from pipeline execution to Neo4j storage.
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
from src.utils.neo4j_driver import Neo4jConnectionManager

# Mark all tests in this module as asyncio and require Neo4j
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


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
        self, integration_dirs, mock_config, test_project_and_interview_ids, clean_test_database
    ):
        """Test complete pipeline execution using Neo4jAnalysisWriter for result storage."""
        input_dir, output_dir, map_dir = integration_dirs
        project_id, interview_id = test_project_and_interview_ids

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
            def __init__(self, *args, **kwargs):
                self.project_id = project_id
                self.interview_id = interview_id
                super().__init__(*args, **kwargs)

            def _setup_file_io(self, file_path: Path):
                """Override to use Neo4jAnalysisWriter instead of LocalJsonlAnalysisWriter."""
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

                # Use local storage for data source and map storage
                data_source = LocalTextDataSource(file_path)
                map_storage = Neo4jMapStorage(self.project_id, self.interview_id)

                # Use Neo4jAnalysisWriter instead of LocalJsonlAnalysisWriter
                analysis_writer = Neo4jAnalysisWriter(self.project_id, self.interview_id)

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
            )

            await orchestrator.execute(specific_file=test_file.name)

        # Verify the results were stored in Neo4j
        async with await Neo4jConnectionManager.get_session() as session:
            # First, find the actual project and interview IDs that were created
            # (since the pipeline generates its own IDs)
            project_result = await session.run("MATCH (p:Project) RETURN p.project_id as project_id LIMIT 1")
            project_record = await project_result.single()
            assert project_record is not None
            actual_project_id = project_record["project_id"]

            interview_result = await session.run("MATCH (i:Interview) RETURN i.interview_id as interview_id LIMIT 1")
            interview_record = await interview_result.single()
            assert interview_record is not None
            actual_interview_id = interview_record["interview_id"]

            # Check that sentences were stored
            sentences_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                RETURN s.sentence_id as sentence_id, s.text as text, s.sequence_order as sequence_order
                ORDER BY s.sentence_id
                """,
                interview_id=actual_interview_id,
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
                interview_id=actual_interview_id,
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
                RETURN f.name as function_name, s.sentence_id as sentence_id
                ORDER BY s.sentence_id
                """,
                interview_id=actual_interview_id,
            )

            functions = []
            async for record in function_result:
                functions.append({"sentence_id": record["sentence_id"], "function_name": record["function_name"]})

            assert len(functions) == 3
            assert functions[0]["function_name"] == "greeting"
            assert functions[1]["function_name"] == "question"
            assert functions[2]["function_name"] == "statement"

            # Check keywords (should respect cardinality limits)
            keyword_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
                RETURN k.text as keyword, s.sentence_id as sentence_id
                ORDER BY s.sentence_id, k.text
                """,
                interview_id=actual_interview_id,
            )

            keywords = []
            async for record in keyword_result:
                keywords.append({"sentence_id": record["sentence_id"], "keyword": record["keyword"]})

            # Should have keywords for all sentences (limited by cardinality)
            assert len(keywords) > 0

            # Verify that keywords respect cardinality limits (max 6 per sentence)
            keyword_counts = {}
            for kw in keywords:
                sentence_id = kw["sentence_id"]
                keyword_counts[sentence_id] = keyword_counts.get(sentence_id, 0) + 1

            for sentence_id, count in keyword_counts.items():
                assert count <= 6, f"Sentence {sentence_id} has {count} keywords, exceeding limit of 6"

    async def test_pipeline_with_neo4j_analysis_writer_error_handling(
        self, integration_dirs, mock_config, test_project_and_interview_ids, clean_test_database
    ):
        """Test that analysis errors are properly handled and stored in Neo4j."""
        input_dir, output_dir, map_dir = integration_dirs
        project_id, interview_id = test_project_and_interview_ids

        # Create test input file
        test_file = input_dir / "test_error.txt"
        test_content = "Good sentence. Bad sentence that will fail."
        test_file.write_text(test_content)

        # Create a custom orchestrator that uses Neo4jAnalysisWriter
        class TestOrchestrator(PipelineOrchestrator):
            def __init__(self, *args, **kwargs):
                self.project_id = project_id
                self.interview_id = interview_id
                super().__init__(*args, **kwargs)

            def _setup_file_io(self, file_path: Path):
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

                data_source = LocalTextDataSource(file_path)
                map_storage = Neo4jMapStorage(self.project_id, self.interview_id)
                analysis_writer = Neo4jAnalysisWriter(self.project_id, self.interview_id)

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
            )

            await orchestrator.execute(specific_file=test_file.name)

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

            # Check analysis nodes - should have one successful and one error
            analysis_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN a.error_data as error_data, s.sentence_id as sentence_id
                ORDER BY s.sentence_id
                """,
                interview_id=interview_id,
            )

            analyses = []
            async for record in analysis_result:
                analyses.append({"sentence_id": record["sentence_id"], "error_data": record["error_data"]})

            # Should only have one successful analysis (sentence 0)
            # Error results are not stored in Neo4j by design
            assert len(analyses) == 1

            # First sentence should have no error data
            assert analyses[0]["sentence_id"] == 0
            assert analyses[0]["error_data"] is None

            # Verify that sentence 1 exists but has no analysis node
            sentence1_analysis_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: 1})
                OPTIONAL MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                RETURN s.sentence_id as sentence_id, a.sentence_id as analysis_sentence_id
                """,
                interview_id=interview_id,
            )
            sentence1_record = await sentence1_analysis_result.single()
            assert sentence1_record is not None
            assert sentence1_record["sentence_id"] == 1
            assert sentence1_record["analysis_sentence_id"] is None  # No analysis node

            # Check that successful sentence has dimension relationships
            function_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: 0})
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
        self, integration_dirs, mock_config, test_project_and_interview_ids, clean_test_database
    ):
        """Test that cardinality limits are properly enforced in Neo4j storage."""
        input_dir, output_dir, map_dir = integration_dirs
        project_id, interview_id = test_project_and_interview_ids

        # Create test input file
        test_file = input_dir / "test_cardinality.txt"
        test_content = "This sentence has many keywords for testing cardinality limits."
        test_file.write_text(test_content)

        # Create a custom orchestrator that uses Neo4jAnalysisWriter
        class TestOrchestrator(PipelineOrchestrator):
            def __init__(self, *args, **kwargs):
                self.project_id = project_id
                self.interview_id = interview_id
                super().__init__(*args, **kwargs)

            def _setup_file_io(self, file_path: Path):
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

                data_source = LocalTextDataSource(file_path)
                map_storage = Neo4jMapStorage(self.project_id, self.interview_id)
                analysis_writer = Neo4jAnalysisWriter(self.project_id, self.interview_id)

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
            )

            await orchestrator.execute(specific_file=test_file.name)

        # Verify cardinality limits are enforced
        async with await Neo4jConnectionManager.get_session() as session:
            # First, find the actual project and interview IDs that were created
            # (since the pipeline generates its own IDs)
            project_result = await session.run("MATCH (p:Project) RETURN p.project_id as project_id LIMIT 1")
            project_record = await project_result.single()
            assert project_record is not None
            actual_project_id = project_record["project_id"]

            interview_result = await session.run("MATCH (i:Interview) RETURN i.interview_id as interview_id LIMIT 1")
            interview_record = await interview_result.single()
            assert interview_record is not None
            actual_interview_id = interview_record["interview_id"]

            # Check that only 6 keywords were stored (cardinality limit)
            keyword_result = await session.run(
                """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
                MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
                RETURN count(k) as keyword_count
                """,
                interview_id=actual_interview_id,
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
                interview_id=actual_interview_id,
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
                interview_id=actual_interview_id,
            )
            function_record = await function_result.single()
            assert function_record is not None
            function_count = function_record["function_count"]
            assert function_count == 1, f"Expected 1 function type (cardinality limit), got {function_count}"
