"""
tests/persistence/test_save_analysis_to_graph_unit.py

Unit tests for the save_analysis_to_graph function focusing on logic validation
without requiring a running Neo4j instance. These tests verify parameter validation,
error handling, and orchestration logic.

These tests follow cardinal rules:
1. Test actual functionality with realistic scenarios
2. Focus on unit-level validation of the function's behavior
3. Use mocking strategically to isolate the function under test
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.persistence.graph_persistence import save_analysis_to_graph


class TestSaveAnalysisToGraphValidation:
    """Test input validation and early return logic."""

    @pytest.mark.asyncio
    async def test_missing_sentence_id_handling(self):
        """Test that missing sentence_id causes early return with warning."""
        invalid_data = {
            "sequence_order": 0,
            "sentence": "Missing sentence_id field",
        }

        with patch("src.persistence.graph_persistence.logger") as mock_logger:
            # Should return early without attempting database operations
            await save_analysis_to_graph(invalid_data, "test.txt", MagicMock())

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "missing core fields" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_missing_sequence_order_handling(self):
        """Test that missing sequence_order causes early return with warning."""
        invalid_data = {
            "sentence_id": 1,
            "sentence": "Missing sequence_order field",
        }

        with patch("src.persistence.graph_persistence.logger") as mock_logger:
            await save_analysis_to_graph(invalid_data, "test.txt", MagicMock())

            mock_logger.warning.assert_called_once()
            assert "missing core fields" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_missing_sentence_text_handling(self):
        """Test that missing sentence text causes early return with warning."""
        invalid_data = {
            "sentence_id": 1,
            "sequence_order": 0,
            # Missing sentence field
        }

        with patch("src.persistence.graph_persistence.logger") as mock_logger:
            await save_analysis_to_graph(invalid_data, "test.txt", MagicMock())

            mock_logger.warning.assert_called_once()
            assert "missing core fields" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_empty_data_handling(self):
        """Test that completely empty data causes early return with warning."""
        with patch("src.persistence.graph_persistence.logger") as mock_logger:
            await save_analysis_to_graph({}, "test.txt", MagicMock())

            mock_logger.warning.assert_called_once()
            assert "missing core fields" in mock_logger.warning.call_args[0][0]


class TestSaveAnalysisToGraphOrchestration:
    """Test the 6-step Cypher orchestration logic with mocked database."""

    @pytest.fixture
    def realistic_analysis_data(self):
        """Provide realistic sentence analysis data for testing."""
        return {
            "sentence_id": 42,
            "sequence_order": 3,
            "sentence": "The candidate demonstrated strong problem-solving skills during the technical interview.",
            "function_type": "declarative",
            "structure_type": "complex",
            "purpose": "evaluation",
            "topic_level_1": "technical_assessment",
            "topic_level_3": "problem_solving_evaluation",
            "overall_keywords": ["candidate", "problem-solving", "technical", "interview"],
            "domain_keywords": ["assessment", "skills", "evaluation"],
        }

    @pytest.fixture
    def minimal_analysis_data(self):
        """Provide minimal required analysis data for testing edge cases."""
        return {
            "sentence_id": 1,
            "sequence_order": 0,
            "sentence": "Hello world.",
        }

    @pytest.fixture
    def mock_session(self):
        """Create a mock Neo4j session for testing."""
        session = AsyncMock()
        session.run = AsyncMock()
        return session

    @pytest.fixture
    def mock_connection_manager(self, mock_session):
        """Create a mock connection manager that returns our mock session."""
        manager = MagicMock()

        # Create a proper async context manager mock
        async_context = AsyncMock()
        async_context.__aenter__ = AsyncMock(return_value=mock_session)
        async_context.__aexit__ = AsyncMock(return_value=None)
        manager.get_session = AsyncMock(return_value=async_context)
        return manager

    @pytest.mark.asyncio
    async def test_complete_orchestration_query_sequence(
        self, realistic_analysis_data, mock_connection_manager, mock_session
    ):
        """Test that all 6 orchestration steps execute the correct queries."""
        filename = "test_interview.txt"

        await save_analysis_to_graph(realistic_analysis_data, filename, mock_connection_manager)

        # Verify the session was acquired
        mock_connection_manager.get_session.assert_called_once()

        # Verify all expected queries were executed
        assert mock_session.run.call_count >= 5  # At least: sentence, types, topics, keywords, follows

        # Extract all query calls
        query_calls = mock_session.run.call_args_list

        # Verify Step 1 & 2: SourceFile and Sentence creation
        sentence_query = query_calls[0][0][0]
        assert "MERGE (f:SourceFile" in sentence_query
        assert "MERGE (s:Sentence" in sentence_query
        assert "MERGE (s)-[:PART_OF_FILE]->(f)" in sentence_query

        # Verify parameters were passed correctly
        sentence_params = query_calls[0][1]["parameters"]
        assert sentence_params["sentence_id"] == realistic_analysis_data["sentence_id"]
        assert sentence_params["filename"] == filename
        assert sentence_params["text"] == realistic_analysis_data["sentence"]

        # Verify Step 3: Type nodes creation (function_type, structure_type, purpose)
        type_queries_found = 0
        for call in query_calls[1:]:
            query = call[0][0]
            if "HAS_FUNCTION_TYPE" in query or "HAS_STRUCTURE_TYPE" in query or "HAS_PURPOSE" in query:
                type_queries_found += 1
        assert type_queries_found == 3  # Should have 3 type queries

        # Verify Step 4: Topic nodes creation
        topic_queries_found = any("HAS_TOPIC" in call[0][0] for call in query_calls)
        assert topic_queries_found

        # Verify Step 5: Keyword nodes creation
        keyword_queries_found = sum(
            1
            for call in query_calls
            if "MENTIONS_OVERALL_KEYWORD" in call[0][0] or "MENTIONS_DOMAIN_KEYWORD" in call[0][0]
        )
        assert keyword_queries_found == 2  # Should have overall and domain keyword queries

    @pytest.mark.asyncio
    async def test_minimal_data_orchestration(self, minimal_analysis_data, mock_connection_manager, mock_session):
        """Test orchestration with minimal data (no optional fields)."""
        filename = "test_minimal.txt"

        await save_analysis_to_graph(minimal_analysis_data, filename, mock_connection_manager)

        # Should still create basic sentence structure
        mock_connection_manager.get_session.assert_called_once()
        assert mock_session.run.call_count >= 1

        # First query should be sentence creation
        sentence_query = mock_session.run.call_args_list[0][0][0]
        assert "MERGE (f:SourceFile" in sentence_query
        assert "MERGE (s:Sentence" in sentence_query

        # Should not have type, topic, or keyword queries
        all_queries = " ".join(call[0][0] for call in mock_session.run.call_args_list)
        assert "HAS_FUNCTION_TYPE" not in all_queries
        assert "HAS_TOPIC" not in all_queries
        assert "MENTIONS_OVERALL_KEYWORD" not in all_queries

    @pytest.mark.asyncio
    async def test_follows_relationship_logic(self, mock_connection_manager, mock_session):
        """Test FOLLOWS relationship creation logic."""
        # Test with sequence_order > 0
        data_with_predecessor = {
            "sentence_id": 5,
            "sequence_order": 3,
            "sentence": "This sentence should have a FOLLOWS relationship.",
        }

        await save_analysis_to_graph(data_with_predecessor, "test.txt", mock_connection_manager)

        # Should include FOLLOWS query
        follows_queries = [call for call in mock_session.run.call_args_list if ":FOLLOWS" in call[0][0]]
        assert len(follows_queries) == 1

        follows_query = follows_queries[0][0][0]
        assert "MERGE (s1)-[r:FOLLOWS]->(s2)" in follows_query

        # Test with sequence_order = 0 (no predecessor)
        mock_session.reset_mock()
        data_no_predecessor = {
            "sentence_id": 1,
            "sequence_order": 0,
            "sentence": "First sentence, no FOLLOWS relationship.",
        }

        await save_analysis_to_graph(data_no_predecessor, "test.txt", mock_connection_manager)

        # Should not include FOLLOWS query
        follows_queries = [call for call in mock_session.run.call_args_list if ":FOLLOWS" in call[0][0]]
        assert len(follows_queries) == 0

    @pytest.mark.asyncio
    async def test_keyword_array_processing(self, mock_connection_manager, mock_session):
        """Test that keyword arrays are processed with UNWIND queries."""
        data_with_keywords = {
            "sentence_id": 10,
            "sequence_order": 0,
            "sentence": "Sentence with various keywords.",
            "overall_keywords": ["keyword1", "keyword2", "keyword3"],
            "domain_keywords": ["domain1", "domain2"],
        }

        await save_analysis_to_graph(data_with_keywords, "test.txt", mock_connection_manager)

        # Find keyword queries
        keyword_queries = [
            call for call in mock_session.run.call_args_list if "UNWIND" in call[0][0] and "keyword" in call[0][0]
        ]

        assert len(keyword_queries) == 2  # Overall and domain keywords

        # Verify UNWIND syntax is used
        overall_query = next(call for call in keyword_queries if "overall_keywords" in call[0][0])
        assert "UNWIND $overall_keywords" in overall_query[0][0]

        domain_query = next(call for call in keyword_queries if "domain_keywords" in call[0][0])
        assert "UNWIND $domain_keywords" in domain_query[0][0]

    @pytest.mark.asyncio
    async def test_parameter_binding_correctness(self, realistic_analysis_data, mock_connection_manager, mock_session):
        """Test that all parameters are correctly bound to queries."""
        filename = "parameter_test.txt"

        await save_analysis_to_graph(realistic_analysis_data, filename, mock_connection_manager)

        # Check that all query calls received the expected parameters
        for call in mock_session.run.call_args_list:
            if len(call) > 1 and "parameters" in call[1]:
                params = call[1]["parameters"]

                # Core parameters should always be present
                assert "sentence_id" in params
                assert "filename" in params
                assert params["sentence_id"] == realistic_analysis_data["sentence_id"]
                assert params["filename"] == filename

    @pytest.mark.asyncio
    async def test_database_error_propagation(self, realistic_analysis_data, mock_connection_manager):
        """Test that database errors are properly logged and propagated."""
        # Make the session raise an exception
        mock_session = AsyncMock()
        mock_session.run.side_effect = Exception("Database connection failed")

        async_context = AsyncMock()
        async_context.__aenter__ = AsyncMock(return_value=mock_session)
        async_context.__aexit__ = AsyncMock(return_value=None)
        mock_connection_manager.get_session = AsyncMock(return_value=async_context)

        with patch("src.persistence.graph_persistence.logger") as mock_logger:
            with pytest.raises(Exception, match="Database connection failed"):
                await save_analysis_to_graph(realistic_analysis_data, "error_test.txt", mock_connection_manager)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Failed during graph update" in mock_logger.error.call_args[0][0]
            # Verify exc_info=True for stack trace
            assert mock_logger.error.call_args[1]["exc_info"] is True

    @pytest.mark.asyncio
    async def test_follows_relationship_error_handling(self, mock_connection_manager):
        """Test that FOLLOWS relationship errors are handled gracefully."""
        # Create a session that fails only on FOLLOWS query
        mock_session = AsyncMock()

        def selective_failure(query, **kwargs):
            if ":FOLLOWS" in query:
                raise Exception("FOLLOWS relationship failed")
            return AsyncMock()

        mock_session.run.side_effect = selective_failure
        async_context = AsyncMock()
        async_context.__aenter__ = AsyncMock(return_value=mock_session)
        async_context.__aexit__ = AsyncMock(return_value=None)
        mock_connection_manager.get_session = AsyncMock(return_value=async_context)

        data_with_follows = {
            "sentence_id": 5,
            "sequence_order": 2,
            "sentence": "This should trigger FOLLOWS relationship creation.",
        }

        with patch("src.persistence.graph_persistence.logger") as mock_logger:
            # Should not raise exception - FOLLOWS errors are handled gracefully
            await save_analysis_to_graph(data_with_follows, "follows_error_test.txt", mock_connection_manager)

            # Should log a warning about FOLLOWS failure
            mock_logger.warning.assert_called_once()
            assert "Could not create :FOLLOWS relationship" in mock_logger.warning.call_args[0][0]


class TestSaveAnalysisToGraphEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_none_values_in_optional_fields(self):
        """Test handling of explicit None values in optional fields."""
        data_with_nones = {
            "sentence_id": 100,
            "sequence_order": 0,
            "sentence": "Sentence with explicit None values.",
            "function_type": None,
            "structure_type": None,
            "purpose": None,
            "topic_level_1": None,
            "topic_level_3": None,
            "overall_keywords": None,
            "domain_keywords": None,
        }

        mock_session = AsyncMock()
        mock_connection_manager = MagicMock()

        # Create a proper async context manager mock
        async_context = AsyncMock()
        async_context.__aenter__ = AsyncMock(return_value=mock_session)
        async_context.__aexit__ = AsyncMock(return_value=None)
        mock_connection_manager.get_session = AsyncMock(return_value=async_context)

        await save_analysis_to_graph(data_with_nones, "none_test.txt", mock_connection_manager)

        # Should only have sentence creation query, no optional queries
        assert mock_session.run.call_count == 1

        # Verify it's the sentence creation query
        sentence_query = mock_session.run.call_args_list[0][0][0]
        assert "MERGE (f:SourceFile" in sentence_query
        assert "MERGE (s:Sentence" in sentence_query

    @pytest.mark.asyncio
    async def test_empty_keyword_arrays(self):
        """Test handling of empty keyword arrays."""
        data_with_empty_arrays = {
            "sentence_id": 200,
            "sequence_order": 0,
            "sentence": "Sentence with empty keyword arrays.",
            "overall_keywords": [],
            "domain_keywords": [],
        }

        mock_session = AsyncMock()
        mock_connection_manager = MagicMock()

        # Create a proper async context manager mock
        async_context = AsyncMock()
        async_context.__aenter__ = AsyncMock(return_value=mock_session)
        async_context.__aexit__ = AsyncMock(return_value=None)
        mock_connection_manager.get_session = AsyncMock(return_value=async_context)

        await save_analysis_to_graph(data_with_empty_arrays, "empty_arrays_test.txt", mock_connection_manager)

        # Should not have keyword queries for empty arrays
        keyword_queries = [
            call for call in mock_session.run.call_args_list if "UNWIND" in call[0][0] and "keyword" in call[0][0]
        ]
        assert len(keyword_queries) == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_parameters(self):
        """Test handling of special characters in analysis data."""
        special_char_data = {
            "sentence_id": 300,
            "sequence_order": 0,
            "sentence": "Sentence with special chars: àáâãäå, ñ, ü, and symbols like @#$%^&*()!",
            "function_type": "declarative-with-hyphens",
            "structure_type": "complex/compound",
            "purpose": "testing & validation",
            "topic_level_1": "special_characters_topic",
            "overall_keywords": ["special-chars", "symbols@test", "unicode-ñ"],
            "domain_keywords": ["testing&validation", "edge-cases"],
        }

        mock_session = AsyncMock()
        mock_connection_manager = MagicMock()

        # Create a proper async context manager mock
        async_context = AsyncMock()
        async_context.__aenter__ = AsyncMock(return_value=mock_session)
        async_context.__aexit__ = AsyncMock(return_value=None)
        mock_connection_manager.get_session = AsyncMock(return_value=async_context)

        # Should not raise exception with special characters
        await save_analysis_to_graph(special_char_data, "special_chars_test.txt", mock_connection_manager)

        # Verify parameters were passed correctly
        sentence_params = mock_session.run.call_args_list[0][1]["parameters"]
        assert sentence_params["text"] == special_char_data["sentence"]
        assert sentence_params["function_type"] == special_char_data["function_type"]
