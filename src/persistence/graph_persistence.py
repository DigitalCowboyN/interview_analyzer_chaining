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
        # TODO: Implement Cypher query and execution

        # --- 4. MERGE Topic Nodes & Relationships ---
        # TODO: Implement Cypher query and execution
        
        # --- 5. MERGE Keyword Nodes & Relationships ---
        # TODO: Implement Cypher query and execution
        
        # --- 6. MERGE :FOLLOWS Relationship (Optional but Recommended) ---
        # TODO: Implement Cypher query and execution

        logger.debug(f"Successfully processed graph updates for sentence {sentence_id} from '{filename}'.")

    except Exception as e:
        logger.error(f"Failed during graph update for sentence {sentence_id} from '{filename}': {e}", exc_info=True)
        raise 