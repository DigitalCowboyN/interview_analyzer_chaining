"""
src/persistence/graph_persistence.py

Contains functions for saving analysis results to a Neo4j graph database.
"""

from typing import Dict, Any
from src.utils.neo4j_driver import Neo4jConnectionManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def save_analysis_to_graph(
    analysis_data: Dict[str, Any], 
    filename: str, 
    connection_manager: Neo4jConnectionManager
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
    sentence_id = analysis_data.get('sentence_id')
    sequence_order = analysis_data.get('sequence_order')
    sentence_text = analysis_data.get('sentence')

    if sentence_id is None or sequence_order is None or sentence_text is None:
        logger.warning(f"Skipping graph save for analysis data due to missing core fields (sentence_id, sequence_order, sentence): {analysis_data}")
        return

    logger.debug(f"Saving analysis for sentence {sentence_id} from file '{filename}' to graph.")

    params = {
        'filename': filename,
        'sentence_id': sentence_id,
        'sequence_order': sequence_order,
        'text': sentence_text,
        # Add other analysis data fields as needed for later steps
        'function_type': analysis_data.get('function_type'),
        'structure_type': analysis_data.get('structure_type'),
        'purpose': analysis_data.get('purpose'),
        'topic_level_1': analysis_data.get('topic_level_1'),
        'topic_level_3': analysis_data.get('topic_level_3'),
        'overall_keywords': analysis_data.get('overall_keywords', []),
        'domain_keywords': analysis_data.get('domain_keywords', [])
    }
    # Filter out None values for cleaner parameter passing, if desired by Cypher logic
    # params = {k: v for k, v in params.items() if v is not None}

    try:
        # Combine steps 1 & 2 into a single query for efficiency and atomicity
        # Ensure SourceFile exists, then MERGE Sentence and link it.
        # Using the (filename, sentence_id) NODE KEY for Sentence merge.
        query_sentence = """
        MERGE (f:SourceFile {filename: $filename})
        MERGE (s:Sentence {sentence_id: $sentence_id, filename: $filename})
        ON CREATE SET 
            s.text = $text, 
            s.sequence_order = $sequence_order
        ON MATCH SET 
            s.text = $text, 
            s.sequence_order = $sequence_order
        MERGE (s)-[:PART_OF_FILE]->(f)
        """
        await connection_manager.execute_query(query_sentence, parameters=params)
        logger.debug(f"Merged SourceFile and Sentence nodes for sentence {sentence_id} from '{filename}'.")

        # --- 3. MERGE Type Nodes & Relationships (FunctionType, StructureType, Purpose) ---
        type_queries = []
        # FunctionType
        if params.get('function_type'):
            type_queries.append("""
            MATCH (s:Sentence {sentence_id: $sentence_id, filename: $filename})
            MERGE (t:FunctionType {name: $function_type})
            MERGE (s)-[:HAS_FUNCTION_TYPE]->(t)
            """)
        # StructureType
        if params.get('structure_type'):
            type_queries.append("""
            MATCH (s:Sentence {sentence_id: $sentence_id, filename: $filename})
            MERGE (t:StructureType {name: $structure_type})
            MERGE (s)-[:HAS_STRUCTURE_TYPE]->(t)
            """)
        # Purpose
        if params.get('purpose'):
            type_queries.append("""
            MATCH (s:Sentence {sentence_id: $sentence_id, filename: $filename})
            MERGE (t:Purpose {name: $purpose})
            MERGE (s)-[:HAS_PURPOSE]->(t)
            """)
        
        # Execute type queries if any exist
        if type_queries:
            # Consider running these in a single transaction if possible/needed
            # For simplicity here, running sequentially
            for query in type_queries:
                await connection_manager.execute_query(query, parameters=params)
            logger.debug(f"Merged type nodes/relationships for sentence {sentence_id}.")

        # --- 4. MERGE Topic Nodes & Relationships ---
        topic_query = """
        MATCH (s:Sentence {sentence_id: $sentence_id, filename: $filename})
        // Use UNWIND to process multiple topics efficiently if they are passed as a list
        // For now, handling topic_level_1 and topic_level_3 separately
        // Consider refactoring if topics become a list in analysis_data
        WITH s
        WHERE $topic_level_1 IS NOT NULL
        MERGE (t1:Topic {name: $topic_level_1})
        MERGE (s)-[:HAS_TOPIC]->(t1)
        WITH s // Pass s along for the next optional part
        WHERE $topic_level_3 IS NOT NULL
        MERGE (t3:Topic {name: $topic_level_3})
        MERGE (s)-[:HAS_TOPIC]->(t3)
        """
        # Only execute if at least one topic level exists
        if params.get('topic_level_1') or params.get('topic_level_3'):
            await connection_manager.execute_query(topic_query, parameters=params)
            logger.debug(f"Merged topic nodes/relationships for sentence {sentence_id}.")

        # --- 5. MERGE Keyword Nodes & Relationships ---
        keyword_queries = []
        # Overall Keywords
        if params.get('overall_keywords'):
            # Use UNWIND for efficient list processing
            keyword_queries.append("""
            MATCH (s:Sentence {sentence_id: $sentence_id, filename: $filename})
            UNWIND $overall_keywords AS keyword_text
            MERGE (k:Keyword {text: keyword_text})
            MERGE (s)-[:MENTIONS_OVERALL_KEYWORD]->(k)
            """)
        # Domain Keywords
        if params.get('domain_keywords'):
            keyword_queries.append("""
            MATCH (s:Sentence {sentence_id: $sentence_id, filename: $filename})
            UNWIND $domain_keywords AS keyword_text
            MERGE (k:Keyword {text: keyword_text})
            MERGE (s)-[:MENTIONS_DOMAIN_KEYWORD]->(k)
            """)

        # Execute keyword queries if any exist
        if keyword_queries:
            for query in keyword_queries:
                await connection_manager.execute_query(query, parameters=params)
            logger.debug(f"Merged keyword nodes/relationships for sentence {sentence_id}.")
        
        # --- 6. MERGE :FOLLOWS Relationship (Optional but Recommended) ---
        # TODO: Implement Cypher query and execution

        logger.debug(f"Successfully processed graph updates for sentence {sentence_id} from '{filename}'.")

    except Exception as e:
        logger.error(f"Failed during graph update for sentence {sentence_id} from '{filename}': {e}", exc_info=True)
        raise 