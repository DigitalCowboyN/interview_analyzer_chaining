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

    try:
        # --- 1. MERGE SourceFile Node ---
        # TODO: Implement Cypher query and execution
        
        # --- 2. MERGE Sentence Node & Link to SourceFile ---
        # Ensure properties like text, sequence_order are set/updated
        # TODO: Implement Cypher query and execution

        # --- 3. MERGE Type Nodes & Relationships (FunctionType, StructureType, Purpose) ---
        # Example for FunctionType:
        # function_type = analysis_data.get('function_type')
        # if function_type:
        #     # MERGE (ft:FunctionType {name: $name})
        #     # MERGE (s)-[:HAS_FUNCTION_TYPE]->(ft) where s is the Sentence node
        # TODO: Implement for all three types
        
        # --- 4. MERGE Topic Nodes & Relationships ---
        # Handle topic_level_1 and topic_level_3
        # TODO: Implement Cypher query and execution
        
        # --- 5. MERGE Keyword Nodes & Relationships ---
        # Handle overall_keywords (MENTIONS_OVERALL_KEYWORD)
        # Handle domain_keywords (MENTIONS_DOMAIN_KEYWORD)
        # TODO: Implement Cypher query and execution
        
        # --- 6. MERGE :FOLLOWS Relationship (Optional but Recommended) ---
        # Find previous sentence (sequence_order - 1 in the same file) and MERGE relationship
        # TODO: Implement Cypher query and execution

        logger.debug(f"Successfully saved sentence {sentence_id} from '{filename}' to graph.")

    except Exception as e:
        logger.error(f"Failed to save sentence {sentence_id} from '{filename}' to graph: {e}", exc_info=True)
        # Re-raise the exception so the caller (e.g., writer) knows about the failure
        raise 