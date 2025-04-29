"""
src/io/neo4j_analysis_writer.py

Neo4j implementation of the SentenceAnalysisWriter protocol.
"""
from .protocols import SentenceAnalysisWriter
from src.utils.neo4j_driver import Neo4jConnectionManager
from src.utils.logger import get_logger
from src.config import config # Import config to access cardinality limits
from typing import List, Dict, Any, Set, Optional
import uuid
import datetime

logger = get_logger()

class Neo4jAnalysisWriter(SentenceAnalysisWriter):
    """
    Writes sentence analysis results to a Neo4j graph database.
    Connects Analysis nodes to Sentences and related dimension nodes (Keywords, Topics, etc.).
    Handles cardinality constraints and edit flags based on configuration and graph state.
    """
    def __init__(self, project_id: str, interview_id: str):
        """
        Initializes the writer for a specific project and interview context.

        Args:
            project_id (str): The unique ID (UUID ideally) of the project.
            interview_id (str): The unique ID (UUID ideally) of the interview.
        """
        if not project_id or not interview_id:
            raise ValueError("project_id and interview_id cannot be empty")
        self.project_id = str(project_id)
        self.interview_id = str(interview_id)
        # Load default cardinality limits from config once during initialization
        self.default_limits = config.get('pipeline', {}).get('default_cardinality_limits', {})
        logger.debug(f"Neo4jAnalysisWriter initialized for Project: {self.project_id}, Interview: {self.interview_id}")

    async def initialize(self):
        """
        Ensures the associated Project and Interview nodes exist.
        (Usually not strictly needed if MapStorage ran first, but good practice).
        """
        logger.debug(f"Initializing Neo4j writer for Interview: {self.interview_id}")
        # Optional: Can run a lightweight query to ensure Project/Interview exist
        # or simply rely on them being created by Neo4jMapStorage.initialize
        # For now, we assume they exist.
        pass

    async def write_result(self, result: Dict[str, Any]):
        """
        Writes a single analysis result to the Neo4j graph.

        - Finds the corresponding Sentence node.
        - Creates/updates the Analysis node linked to the Sentence.
        - Creates/updates dimension nodes (Keyword, Topic, etc.).
        - Creates/updates relationships between Analysis and dimension nodes,
          respecting cardinality limits and edit flags.

        Args:
            result (Dict[str, Any]): Dictionary containing analysis data, including
                                     'sentence_id' and analysis fields. Can also be
                                     an error dictionary {sentence_id: ..., error: True, ...}
        """
        sentence_id = result.get("sentence_id")
        if sentence_id is None:
            logger.error(f"write_result missing 'sentence_id' in result: {result}")
            # Or raise ValueError? For now, log and skip.
            return
        
        # Check if the result itself is an error reported by the analysis service
        is_error_result = result.get("error", False)
        
        logger.debug(f"Writing analysis result for Sentence ID: {sentence_id} in Interview: {self.interview_id}")
        
        try:
            async with Neo4jConnectionManager.get_session() as session:
                # Pass necessary instance data to the transaction function
                await session.write_transaction(
                    self._write_result_tx, 
                    self.project_id, 
                    self.interview_id, 
                    sentence_id,
                    result,
                    is_error_result,
                    self.default_limits
                )
            logger.debug(f"Successfully wrote analysis result for Sentence ID: {sentence_id}")
        except Exception as e:
            logger.error(f"Failed Neo4j write_result for Sentence ID {sentence_id} (Interview {self.interview_id}): {e}", exc_info=True)
            raise

    # Transaction function for write_result
    @staticmethod
    def _write_result_tx(tx, project_id: str, interview_id: str, sentence_id: int, result: Dict[str, Any], is_error_result: bool, default_limits: Dict):
        logger.debug(f"Running write_result transaction for Sentence ID: {sentence_id}")

        # 1. Find the Sentence node
        match_sentence_query = ("""
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
            RETURN s
        """)
        sentence_node_result = tx.run(match_sentence_query, interview_id=interview_id, sentence_id=sentence_id)
        sentence_node = sentence_node_result.single()
        
        if not sentence_node:
            logger.error(f"_write_result_tx: Sentence node {sentence_id} not found for Interview {interview_id}. Cannot write analysis.")
            # Raise an error or just return? Raising is safer to signal inconsistency.
            raise ValueError(f"Sentence node {sentence_id} not found for Interview {interview_id}")
        
        # 2. MERGE the Analysis node linked to the Sentence
        #    Set properties on create.
        #    Store the raw error structure if it's an error result.
        analysis_props_on_create = {
            "analysis_id": str(uuid.uuid4()),
            "model_name": result.get("model_name", config.get("openai", {}).get("model_name")), # Get model from result or config
            "created_at": datetime.datetime.now(),
            "is_edited": False,
            "error_data": result if is_error_result else None # Store error details if it's an error
        }
        # Filter out None values
        analysis_props_on_create = {k: v for k, v in analysis_props_on_create.items() if v is not None}

        merge_analysis_query = ("""
            MATCH (s:Sentence {sentence_id: $sentence_id})<-[:HAS_SENTENCE]-(:Interview {interview_id: $interview_id})
            MERGE (s)-[:HAS_ANALYSIS]->(a:Analysis)
            ON CREATE SET a += $props
            RETURN a
        """)
        analysis_node_result = tx.run(merge_analysis_query, 
                                      sentence_id=sentence_id, 
                                      interview_id=interview_id,
                                      props=analysis_props_on_create)
        analysis_node = analysis_node_result.single()["a"] # Get the analysis node
        logger.debug(f"_write_result_tx: Merged Analysis node for Sentence {sentence_id}")

        # --- If it was an error result, stop here --- 
        if is_error_result:
            logger.warning(f"_write_result_tx: Stored error information for Sentence {sentence_id}. No further dimensions processed.")
            return # Do not process dimensions for error results

        # --- Step 3: Handle Dimensions --- 
        
        # Fetch project-specific cardinality overrides once
        project_limits = Neo4jAnalysisWriter._fetch_project_limits(tx, project_id)

        # Helper function calls for each dimension type
        # Function Type (Cardinality 1)
        Neo4jAnalysisWriter._handle_dimension_link(
            tx=tx,
            analysis_node_id=analysis_node.id, # Pass internal ID
            dimension_label="FunctionType",
            dimension_key="name",
            dimension_value=result.get("function_type"),
            relationship_type="HAS_FUNCTION",
            cardinality_limit=project_limits.get("HAS_FUNCTION", default_limits.get("HAS_FUNCTION", 1)), # Default 1
            props_on_create={}
        )

        # Structure Type (Assume Cardinality 1 for now)
        Neo4jAnalysisWriter._handle_dimension_link(
            tx=tx,
            analysis_node_id=analysis_node.id,
            dimension_label="StructureType",
            dimension_key="name",
            dimension_value=result.get("structure_type"),
            relationship_type="HAS_STRUCTURE",
            cardinality_limit=project_limits.get("HAS_STRUCTURE", default_limits.get("HAS_STRUCTURE", 1)), # Default 1
            props_on_create={}
        )
        
        # Purpose (Assume Cardinality 1 for now)
        Neo4jAnalysisWriter._handle_dimension_link(
            tx=tx,
            analysis_node_id=analysis_node.id,
            dimension_label="Purpose",
            dimension_key="name",
            dimension_value=result.get("purpose"),
            relationship_type="HAS_PURPOSE",
            cardinality_limit=project_limits.get("HAS_PURPOSE", default_limits.get("HAS_PURPOSE", 1)), # Default 1
            props_on_create={}
        )

        # Topics (Cardinality Many)
        Neo4jAnalysisWriter._handle_dimension_list_link(
            tx=tx,
            analysis_node_id=analysis_node.id,
            dimension_label="Topic",
            dimension_key="name",
            dimension_values=result.get("topics", []), # Assuming 'topics' key holds a list of topic names
            relationship_type="MENTIONS_TOPIC",
            cardinality_limit=project_limits.get("MENTIONS_TOPIC", default_limits.get("MENTIONS_TOPIC", None)), # Default unlimited
            props_on_create=lambda v: {"name": v} # Function to create props for new Topic node
            # We might need to handle level property for Topic here too
        )
        
        # Keywords (Cardinality N)
        Neo4jAnalysisWriter._handle_dimension_list_link(
            tx=tx,
            analysis_node_id=analysis_node.id,
            dimension_label="Keyword",
            dimension_key="text",
            dimension_values=result.get("keywords", []), # Assuming 'keywords' key holds a list of keyword strings
            relationship_type="MENTIONS_KEYWORD",
            cardinality_limit=project_limits.get("MENTIONS_KEYWORD", default_limits.get("MENTIONS_KEYWORD", 6)), # Default 6
            props_on_create=lambda v: {"text": v}
        )

        # Domain Keywords (Cardinality Many)
        Neo4jAnalysisWriter._handle_dimension_list_link(
            tx=tx,
            analysis_node_id=analysis_node.id,
            dimension_label="DomainKeyword",
            dimension_key="text",
            dimension_values=result.get("domain_keywords", []), # Assuming 'domain_keywords' key
            relationship_type="MENTIONS_DOMAIN_KEYWORD",
            cardinality_limit=project_limits.get("MENTIONS_DOMAIN_KEYWORD", default_limits.get("MENTIONS_DOMAIN_KEYWORD", None)), # Default unlimited
            props_on_create=lambda v: {"text": v, "is_custom": False} # Assume keywords from analysis are not custom? Or derive?
        )

    @staticmethod
    def _fetch_project_limits(tx, project_id: str) -> Dict[str, Optional[int]]:
        "Fetches cardinality limit overrides from the Project node." 
        query = ("""
            MATCH (p:Project {project_id: $project_id})
            RETURN p.max_functions_limit as HAS_FUNCTION,
                   p.max_structures_limit as HAS_STRUCTURE,
                   p.max_purposes_limit as HAS_PURPOSE,
                   p.max_keywords_limit as MENTIONS_KEYWORD,
                   p.max_topics_limit as MENTIONS_TOPIC,
                   p.max_domain_keywords_limit as MENTIONS_DOMAIN_KEYWORD
        """)
        result = tx.run(query, project_id=project_id)
        record = result.single()
        return dict(record) if record else {}

    # Helper for single-value dimension links (handles Max 1 cardinality)
    @staticmethod
    def _handle_dimension_link(tx, *, analysis_node_id: int, dimension_label: str, dimension_key: str, dimension_value: Optional[str], relationship_type: str, cardinality_limit: Optional[int], props_on_create: Dict):
        if not dimension_value:
            # If value is None or empty, potentially remove existing link (if not edited)
            # For now, we just do nothing if no value provided.
            # TODO: Add logic to remove existing link if value is None and rel not edited?
            logger.debug(f"Skipping link {relationship_type} from Analysis {analysis_node_id}: No value provided.")
            return
        
        # Create dimension node properties
        dim_props = {dimension_key: dimension_value}
        dim_props.update(props_on_create) # Add extra props like level if needed

        # Cypher Query:
        # 1. MATCH the Analysis node by its internal ID.
        # 2. OPTIONAL MATCH any existing relationship of this type AND its target node.
        # 3. If relationship exists AND rel.is_edited = true, RETURN (do nothing).
        # 4. If relationship exists AND rel.is_edited = false, DELETE the relationship.
        # 5. MERGE the dimension node (ensures uniqueness based on key property).
        # 6. CREATE the new relationship from Analysis to Dimension node, set is_edited = false.
        query = f"""
        MATCH (a:Analysis) WHERE id(a) = $analysis_node_id
        OPTIONAL MATCH (a)-[r_old:{relationship_type}]->(d_old:{dimension_label})
        
        // Check if edit protection applies
        WITH a, r_old, d_old, CASE WHEN r_old IS NOT NULL AND r_old.is_edited = true THEN true ELSE false END as is_protected
        WHERE is_protected = false // Proceed only if not protected
        
        // Delete old relationship if it exists (and wasn't protected)
        FOREACH (_ IN CASE WHEN r_old IS NOT NULL THEN [1] ELSE [] END | DELETE r_old)
        
        // Merge dimension node and create new relationship
        MERGE (d_new:{dimension_label} {{{dimension_key}: $dim_value}})
        ON CREATE SET d_new += $props_on_create
        CREATE (a)-[r_new:{relationship_type} {{is_edited: false}}]->(d_new)
        """
        
        # Combine parameters
        params = {
            "analysis_node_id": analysis_node_id,
            "dim_value": dimension_value,
            "props_on_create": props_on_create
        }
        
        tx.run(query, params)
        logger.debug(f"Handled link {relationship_type} from Analysis {analysis_node_id} to {dimension_label} {dimension_value}")

    # Helper for multi-value dimension links (handles Max N cardinality)
    @staticmethod
    def _handle_dimension_list_link(tx, *, analysis_node_id: int, dimension_label: str, dimension_key: str, dimension_values: List[str], relationship_type: str, cardinality_limit: Optional[int], props_on_create: callable):
        if not dimension_values:
            logger.debug(f"Skipping list link {relationship_type} from Analysis {analysis_node_id}: No values provided.")
            # Optionally, remove all non-edited existing links if the input list is empty
            # query_delete_all_unedited = f"""
            # MATCH (a:Analysis)-[r:{relationship_type}]->(d:{dimension_label}) 
            # WHERE id(a) = $analysis_node_id AND (r.is_edited IS NULL OR r.is_edited = false) 
            # DELETE r
            # """
            # tx.run(query_delete_all_unedited, analysis_node_id=analysis_node_id)
            # logger.debug(f"Removed existing unedited {relationship_type} links for Analysis {analysis_node_id} due to empty input list.")
            return

        # Ensure uniqueness in input values
        unique_dimension_values = set(dimension_values)

        # 1. Get current state: existing linked values and their edit status
        query_get_existing = f"""
        MATCH (a:Analysis)-[r:{relationship_type}]->(d:{dimension_label})
        WHERE id(a) = $analysis_node_id
        RETURN d.{dimension_key} AS value, COALESCE(r.is_edited, false) AS edited
        """
        existing_results = tx.run(query_get_existing, analysis_node_id=analysis_node_id)
        
        existing_edited_values = set()
        existing_unedited_values = set()
        for record in existing_results:
            if record["edited"]:
                existing_edited_values.add(record["value"])
            else:
                existing_unedited_values.add(record["value"])
        
        logger.debug(f"{relationship_type} existing state: Edited={existing_edited_values}, Unedited={existing_unedited_values}")

        # 2. Identify changes
        target_values_set = unique_dimension_values
        values_to_add_set = target_values_set - existing_edited_values - existing_unedited_values
        unedited_values_to_remove_set = existing_unedited_values - target_values_set

        # 3. Calculate available slots for new items (considering the limit and preserved edited items)
        slots_available = float('inf') # Assume infinite slots if limit is None
        if cardinality_limit is not None and cardinality_limit >= 0:
            slots_available = max(0, cardinality_limit - len(existing_edited_values))
        
        logger.debug(f"{relationship_type}: Limit={cardinality_limit}, Edited={len(existing_edited_values)}, AvailableSlots={slots_available}")
        
        # 4. Delete obsolete unedited relationships
        if unedited_values_to_remove_set:
            query_delete_unedited = f"""
            MATCH (a:Analysis)-[r:{relationship_type}]->(d:{dimension_label})
            WHERE id(a) = $analysis_node_id 
              AND (r.is_edited IS NULL OR r.is_edited = false) 
              AND d.{dimension_key} IN $values_to_remove
            DELETE r
            """
            tx.run(query_delete_unedited, 
                   analysis_node_id=analysis_node_id, 
                   values_to_remove=list(unedited_values_to_remove_set))
            logger.debug(f"Deleted {len(unedited_values_to_remove_set)} obsolete unedited {relationship_type} links for Analysis {analysis_node_id}")

        # 5. Add new relationships up to the available slot limit
        if values_to_add_set and slots_available > 0:
            # Limit the list of values to add based on available slots
            values_to_add_list = list(values_to_add_set)
            if slots_available != float('inf'):
                values_to_add_list = values_to_add_list[:int(slots_available)] # Take only as many as allowed
            
            if values_to_add_list:
                query_add_new = f"""
                MATCH (a:Analysis) WHERE id(a) = $analysis_node_id
                UNWIND $values_to_add AS new_value
                MERGE (d:{dimension_label} {{{dimension_key}: new_value}})
                ON CREATE SET d += $props_on_create_func(new_value) // Use lambda for props
                MERGE (a)-[r:{relationship_type} {{is_edited: false}}]->(d)
                """
                
                # Need to handle props_on_create which is a lambda
                # We pass the list and generate props within the query or pass structured data
                # For simplicity, let's assume props_on_create(value) returns the dict
                # The Cypher query needs adjusting if props depend on the value in a complex way
                # Simpler: Pass the base props, the query uses the value for the key property only
                
                # Re-evaluate parameter passing for props_on_create lambda
                # Let's assume props_on_create lambda is simple enough {key: value}
                # We can construct the dimension node properties beforehand if needed
                
                tx.run(query_add_new,
                       analysis_node_id=analysis_node_id,
                       values_to_add=values_to_add_list,
                       # This lambda won't work directly as a parameter.
                       # Pass base properties or handle complex props differently.
                       # For Keyword/Topic where props_on_create=lambda v: {key: v},
                       # the MERGE statement handles it implicitly.
                       # If props_on_create is more complex, adjust query or params.
                       props_on_create_func=props_on_create # This is illustrative, not direct Cypher param
                       # We might need to pre-compute props if they are complex
                       )
                logger.debug(f"Added {len(values_to_add_list)} new {relationship_type} links for Analysis {analysis_node_id}")
            else:
                 logger.debug(f"No new {relationship_type} links to add for Analysis {analysis_node_id} (limit reached or no new values)")
        else:
             logger.debug(f"No new {relationship_type} links to add for Analysis {analysis_node_id} (slots_available={slots_available}, values_to_add={values_to_add_set})")

    async def finalize(self):
        """
        Finalizes the analysis writing operation (likely a no-op).
        """
        logger.debug(f"Finalizing Neo4j writer for Interview: {self.interview_id}")
        pass # Sessions managed by context manager

    async def read_analysis_ids(self) -> Set[int]:
        """
        Reads sentence IDs that have associated Analysis nodes for the Interview.
        """
        logger.debug(f"Reading analysis IDs for Interview: {self.interview_id}")
        cypher = ("""
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
            // Ensure there is an Analysis node linked
            WHERE (s)-[:HAS_ANALYSIS]->(:Analysis)
            RETURN s.sentence_id AS sentenceId
        """)

        sentence_ids: Set[int] = set()
        try:
            async with Neo4jConnectionManager.get_session() as session:
                result = await session.run(cypher, interview_id=self.interview_id)
                # Process results asynchronously
                async for record in result:
                    s_id = record["sentenceId"]
                    if isinstance(s_id, int):
                        sentence_ids.add(s_id)
                    else:
                        logger.warning(f"Read non-integer sentence_id ({s_id}) with analysis for Interview {self.interview_id}, skipping.")
                        
            logger.debug(f"Successfully read {len(sentence_ids)} unique sentence IDs with analysis for Interview: {self.interview_id}")
            return sentence_ids
        except Exception as e:
            logger.error(f"Failed to read analysis IDs for Interview {self.interview_id}: {e}", exc_info=True)
            raise

    def get_identifier(self) -> str:
        """
        Returns the interview ID as the identifier for this writer context.
        """
        return self.interview_id 