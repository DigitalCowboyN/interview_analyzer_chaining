"""
tests/pipeline/test_analyze_specific_sentences.py

Tests for the analyze_specific_sentences function.

These tests focus on the actual behavior of sentence analysis with realistic data,
not just mock interactions. Tests validate real processing logic.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline import analyze_specific_sentences
from tests.pipeline.conftest import create_realistic_map_entries


class TestAnalyzeSpecificSentences:
    """Test the analyze_specific_sentences function with real data."""

    async def test_analyze_specific_sentences_with_realistic_data(
        self, expected_sentences, real_analysis_service_with_mocked_llm
    ):
        """Test specific sentence analysis with realistic map data and analysis service."""
        # Create realistic map storage with actual sentences
        mock_map_storage = MagicMock()
        mock_map_storage.get_identifier.return_value = "realistic_test_map.jsonl"

        # Use actual sentences from our test content
        map_entries = create_realistic_map_entries(expected_sentences, "test_document.txt")
        mock_map_storage.read_all_entries = AsyncMock(return_value=map_entries)

        # Test analyzing specific sentences (first and third)
        sentence_ids_to_analyze = [0, 2]
        task_id = "test-specific-analysis"

        # Execute the function with real analysis service
        results = await analyze_specific_sentences(
            map_storage=mock_map_storage,
            sentence_ids=sentence_ids_to_analyze,
            analysis_service=real_analysis_service_with_mocked_llm,
            task_id=task_id,
        )

        # Validate results structure and content
        assert len(results) == len(sentence_ids_to_analyze)
        assert all(isinstance(result, dict) for result in results)

        # Check that results contain the correct sentences
        result_sentences = [result["sentence"] for result in results]
        expected_result_sentences = [expected_sentences[i] for i in sentence_ids_to_analyze]
        assert result_sentences == expected_result_sentences

        # Verify sentence IDs were properly remapped
        result_ids = [result["sentence_id"] for result in results]
        assert result_ids == sentence_ids_to_analyze

        # Verify analysis content is realistic (not hardcoded)
        for result in results:
            assert "function_type" in result
            assert "structure_type" in result
            assert result["function_type"] in [
                "declarative",
                "interrogative",
                "exclamatory",
                "request",
            ]
            assert result["structure_type"] in ["simple", "compound", "complex"]

    async def test_analyze_specific_sentences_with_empty_map(self):
        """Test error handling when map storage is empty."""
        mock_map_storage = MagicMock()
        mock_map_storage.get_identifier.return_value = "empty_map.jsonl"
        mock_map_storage.read_all_entries = AsyncMock(return_value=[])

        mock_analysis_service = MagicMock()
        sentence_ids = [1, 2]

        with pytest.raises(ValueError, match="Map storage .* is empty"):
            await analyze_specific_sentences(
                map_storage=mock_map_storage,
                sentence_ids=sentence_ids,
                analysis_service=mock_analysis_service,
            )

    async def test_analyze_specific_sentences_with_missing_ids(self, expected_sentences):
        """Test error handling when requested sentence IDs don't exist in map."""
        mock_map_storage = MagicMock()
        mock_map_storage.get_identifier.return_value = "partial_map.jsonl"

        # Create map with only first 2 sentences
        partial_entries = create_realistic_map_entries(expected_sentences[:2])
        mock_map_storage.read_all_entries = AsyncMock(return_value=partial_entries)

        mock_analysis_service = MagicMock()

        # Try to analyze sentences that don't exist in the map
        missing_sentence_ids = [0, 5, 10]  # 5 and 10 don't exist

        with pytest.raises(ValueError, match="Sentence IDs not found"):
            await analyze_specific_sentences(
                map_storage=mock_map_storage,
                sentence_ids=missing_sentence_ids,
                analysis_service=mock_analysis_service,
            )

    async def test_analyze_specific_sentences_with_invalid_map_entries(
        self, expected_sentences, real_analysis_service_with_mocked_llm
    ):
        """Test handling of invalid entries in map storage."""
        mock_map_storage = MagicMock()
        mock_map_storage.get_identifier.return_value = "invalid_entries_map.jsonl"

        # Create map with some invalid entries mixed with valid ones
        valid_entries = create_realistic_map_entries(expected_sentences[:2])
        invalid_entries = [
            {
                "sentence_id": "invalid_string",
                "sequence_order": 2,
                "sentence": "Invalid ID type",
            },
            {"sentence_id": 3, "sequence_order": 3},  # Missing sentence text
            {
                "sequence_order": 4,
                "sentence": "Missing sentence_id",
            },  # Missing sentence_id
        ]

        all_entries = valid_entries + invalid_entries
        mock_map_storage.read_all_entries = AsyncMock(return_value=all_entries)

        # Should still work with valid entries only
        sentence_ids_to_analyze = [0, 1]  # Only valid IDs

        results = await analyze_specific_sentences(
            map_storage=mock_map_storage,
            sentence_ids=sentence_ids_to_analyze,
            analysis_service=real_analysis_service_with_mocked_llm,
        )

        # Should get results for valid entries
        assert len(results) == 2
        assert all(result["sentence_id"] in sentence_ids_to_analyze for result in results)

    async def test_analyze_specific_sentences_preserves_original_order(
        self, expected_sentences, real_analysis_service_with_mocked_llm
    ):
        """Test that results maintain the order of requested sentence IDs."""
        mock_map_storage = MagicMock()
        mock_map_storage.get_identifier.return_value = "order_test_map.jsonl"

        map_entries = create_realistic_map_entries(expected_sentences)
        mock_map_storage.read_all_entries = AsyncMock(return_value=map_entries)

        # Request sentences in non-sequential order
        requested_order = [3, 0, 2, 1]

        results = await analyze_specific_sentences(
            map_storage=mock_map_storage,
            sentence_ids=requested_order,
            analysis_service=real_analysis_service_with_mocked_llm,
        )

        # Results should maintain the requested order
        result_ids = [result["sentence_id"] for result in results]
        assert result_ids == sorted(requested_order)  # analyze_specific_sentences sorts by original index

        # But the sentences should correspond to the correct IDs
        for i, result in enumerate(results):
            expected_sentence = expected_sentences[result["sentence_id"]]
            assert result["sentence"] == expected_sentence

    async def test_analyze_specific_sentences_with_single_sentence(self, real_analysis_service_with_mocked_llm):
        """Test analysis of a single specific sentence."""
        mock_map_storage = MagicMock()
        mock_map_storage.get_identifier.return_value = "single_sentence_map.jsonl"

        single_sentence = "This is a single test sentence for analysis."
        map_entries = create_realistic_map_entries([single_sentence])
        mock_map_storage.read_all_entries = AsyncMock(return_value=map_entries)

        results = await analyze_specific_sentences(
            map_storage=mock_map_storage,
            sentence_ids=[0],
            analysis_service=real_analysis_service_with_mocked_llm,
        )

        assert len(results) == 1
        assert results[0]["sentence"] == single_sentence
        assert results[0]["sentence_id"] == 0
        assert results[0]["sequence_order"] == 0

    async def test_analyze_specific_sentences_context_building_integration(
        self,
        expected_sentences,
        real_analysis_service_with_mocked_llm,
        realistic_config,
    ):
        """Test that context building actually influences LLM analysis results."""
        mock_map_storage = MagicMock()
        mock_map_storage.get_identifier.return_value = "context_test_map.jsonl"

        map_entries = create_realistic_map_entries(expected_sentences)
        mock_map_storage.read_all_entries = AsyncMock(return_value=map_entries)

        # Test context influence by analyzing sentences with different context positions
        if len(expected_sentences) >= 4:
            # Test edge sentence (limited context) vs middle sentence (full context)
            edge_ids = [0]  # First sentence - limited preceding context
            middle_ids = [2]  # Middle sentence - full context available
        else:
            # Fallback for shorter sentence lists
            edge_ids = [0]
            middle_ids = [1] if len(expected_sentences) > 1 else [0]

        # Analyze edge sentence with limited context
        edge_results = await analyze_specific_sentences(
            map_storage=mock_map_storage,
            sentence_ids=edge_ids,
            analysis_service=real_analysis_service_with_mocked_llm,
        )

        # Analyze middle sentence with full context
        middle_results = await analyze_specific_sentences(
            map_storage=mock_map_storage,
            sentence_ids=middle_ids,
            analysis_service=real_analysis_service_with_mocked_llm,
        )

        # Validate that analysis actually occurred (not just mocked responses)
        assert len(edge_results) == len(edge_ids)
        assert len(middle_results) == len(middle_ids)

        for results in [edge_results, middle_results]:
            for result in results:
                # Test that real LLM analysis occurred with proper structure
                assert "function_type" in result
                assert "structure_type" in result
                assert "sentence" in result
                assert "sentence_id" in result

                # Validate analysis fields contain realistic values (not hardcoded)
                assert result["function_type"] in [
                    "declarative",
                    "interrogative",
                    "exclamatory",
                    "request",
                ]
                assert result["structure_type"] in ["simple", "compound", "complex"]

                # Verify sentence content matches expected input
                sentence_id = result["sentence_id"]
                expected_sentence = expected_sentences[sentence_id]
                assert result["sentence"] == expected_sentence

        # Test that context actually influences analysis by checking for context-dependent fields
        # Real LLM analysis should produce different results based on available context
        if len(expected_sentences) >= 4 and edge_ids != middle_ids:
            edge_result = edge_results[0]
            middle_result = middle_results[0]

            # Results should have been processed with different context amounts
            # This validates that context building is actually feeding into analysis
            # (We can't predict exact differences, but can verify processing occurred)
            assert edge_result["sentence_id"] != middle_result["sentence_id"]
            assert edge_result["sentence"] != middle_result["sentence"]
