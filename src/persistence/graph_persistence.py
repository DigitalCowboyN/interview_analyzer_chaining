"""
src/persistence/graph_persistence.py

Contains functions for saving analysis results to a Neo4j graph database.
"""

from typing import Any, Dict, Optional

from src.utils.logger import get_logger
from src.utils.neo4j_driver import Neo4jConnectionManager

logger = get_logger()


async def save_analysis_to_graph(
    analysis_data: Dict[str, Any],
    filename: str,
    interview_id_or_connection_manager,
    connection_manager: Optional[Neo4jConnectionManager] = None,
) -> None:
    """
    Saves a single sentence's analysis data to the Neo4j graph.

    This function orchestrates the creation/merging of nodes (Sentence, SourceFile,
    FunctionType, StructureType, Purpose, Topic, Keyword) and their relationships
    based on the provided analysis data and the proposed graph schema.

    Args:
        analysis_data (Dict[str, Any]): The dictionary containing analysis results
                                         for one sentence. Expected keys include
                                         'sentence_id', 'sequence_order', 'sentence',
                                         'function_type', 'structure_type', 'purpose',
                                         'topic_level_1', 'topic_level_3',
                                         'overall_keywords', 'domain_keywords'.
        filename (str): The base filename of the source text file.
        connection_manager (Neo4jConnectionManager): Instance to handle Neo4j connection
                                                    and query execution.

    Returns:
        None

    Raises:
        Exception: Propagates exceptions from the Neo4j driver or connection manager
                   if database operations fail. Consider adding more specific error handling.
    """
    sentence_id = analysis_data.get("sentence_id")
    sequence_order = analysis_data.get("sequence_order")
    sentence_text = analysis_data.get("sentence")

    if sentence_id is None or sequence_order is None or sentence_text is None:
        logger.warning(
            f"Skipping graph save for analysis data due to missing core fields "
            f"(sentence_id, sequence_order, sentence): {analysis_data}"
        )
        return

    # Handle backward compatibility: detect old vs new calling pattern
    if connection_manager is None:
        # Old calling pattern: save_analysis_to_graph(data, filename, connection_manager)
        actual_connection_manager = interview_id_or_connection_manager
        interview_id = None
        logger.warning(
            f"save_analysis_to_graph called with old signature for sentence {sentence_id} from '{filename}'. "
            f"Using legacy behavior without interview_id disambiguation."
        )
    else:
        # New calling pattern: save_analysis_to_graph(data, filename, interview_id, connection_manager)
        interview_id = interview_id_or_connection_manager
        actual_connection_manager = connection_manager

    logger.debug(f"Saving analysis for sentence {sentence_id} from file '{filename}' to graph.")

    # Handle case where interview_id is not provided (legacy mode)
    if not interview_id:
        logger.debug(
            f"No interview_id provided for sentence {sentence_id} from '{filename}', using legacy sentence matching."
        )
    else:
        logger.debug(f"Using interview_id '{interview_id}' for precise sentence matching.")

    params = {
        "filename": filename,
        "interview_id": interview_id,
        "sentence_id": sentence_id,
        "sequence_order": sequence_order,
        "text": sentence_text,
        # Add other analysis data fields as needed for later steps
        "function_type": analysis_data.get("function_type"),
        "structure_type": analysis_data.get("structure_type"),
        "purpose": analysis_data.get("purpose"),
        "topic_level_1": analysis_data.get("topic_level_1"),
        "topic_level_3": analysis_data.get("topic_level_3"),
        "overall_keywords": analysis_data.get("overall_keywords", []),
        "domain_keywords": analysis_data.get("domain_keywords", []),
    }
    # Filter out None values for cleaner parameter passing, if desired by Cypher logic
    # params = {k: v for k, v in params.items() if v is not None}

    try:
        # Use the imported singleton instance, assuming driver is pre-initialized
        # AWAIT the coroutine to get the actual session context manager
        async with await actual_connection_manager.get_session() as session:
            # ARCHITECTURAL FIX: Reuse existing sentences created by Neo4jMapStorage/Neo4jAnalysisWriter
            # instead of creating duplicate sentences with filename properties.
            # This ensures Analysis nodes are linked to sentences that have filename properties.

            if interview_id:
                # New behavior: Use interview_id for precise sentence matching (prevents ID collisions)
                query_sentence = """
                MERGE (f:SourceFile {filename: $filename})
                WITH f
                // Find the sentence by matching via specific Interview and sentence_id
                // This handles sentence ID collisions between different files/interviews
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->
                      (s:Sentence {sentence_id: $sentence_id})
                SET s.filename = $filename
                MERGE (s)-[:PART_OF_FILE]->(f)
                """
            else:
                # Legacy behavior: Match any sentence with the given sentence_id (may have collisions)
                query_sentence = """
                MERGE (f:SourceFile {filename: $filename})
                WITH f
                MATCH (s:Sentence {sentence_id: $sentence_id})
                SET s.filename = $filename
                MERGE (s)-[:PART_OF_FILE]->(f)
                """
            # Execute using the session
            await session.run(query_sentence, parameters=params)
            logger.debug(f"Merged SourceFile and Sentence nodes for sentence {sentence_id} from '{filename}'.")

            # ARCHITECTURAL FIX: Remove duplicate dimension relationship creation.
            # Neo4jAnalysisWriter already creates Analysis nodes and links them to dimension nodes.
            # This function now only handles file-specific relationships (filename and PART_OF_FILE).
            logger.debug(
                f"Skipped dimension relationships for sentence {sentence_id} - handled by Neo4jAnalysisWriter."
            )

            # --- 6. MERGE :FOLLOWS Relationship (Optional but Recommended) ---
            # Only attempt to create FOLLOWS if sequence_order > 0
            if sequence_order is not None and sequence_order > 0:
                follows_query = """
                // Match the current sentence (s2)
                MATCH (s2:Sentence {sentence_id: $sentence_id})
                // Match the previous sentence (s1) in the same file with same filename
                MATCH (s1:Sentence {sequence_order: $sequence_order - 1})
                WHERE s1.filename = $filename
                // Ensure the relationship doesn't already exist, then create it
                MERGE (s1)-[r:FOLLOWS]->(s2)
                """
                # Note: No parameters needed beyond sequence_order and sentence_id,
                # which are already in the params dictionary.
                # We calculate the previous sequence order directly in the query.
                # Need to handle the case where sequence_order might be 0 or None explicitly.
                try:
                    await session.run(follows_query, parameters=params)
                    logger.debug(f"Merged :FOLLOWS relationship for sentence {sentence_id}.")
                except Exception as follows_e:
                    # Log specifically if FOLLOWS fails, but don't necessarily fail the whole save
                    logger.warning(
                        f"Could not create :FOLLOWS relationship for sentence {sentence_id} "
                        f"(prev: {sequence_order - 1}): {follows_e}"
                    )

        # Changed level to INFO
        logger.info(f"Successfully completed graph updates for sentence {sentence_id} from '{filename}'.")

    except Exception as e:
        logger.error(
            f"Failed during graph update for sentence {sentence_id} from '{filename}': {e}",
            exc_info=True,
        )
        raise
