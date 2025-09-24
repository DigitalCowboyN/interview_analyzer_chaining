"""
src/io/neo4j_analysis_writer.py

Neo4j implementation of the SentenceAnalysisWriter protocol.
"""

import datetime
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from src.config import config  # Import config to access cardinality limits
from src.utils.logger import get_logger
from src.utils.neo4j_driver import Neo4jConnectionManager

from .protocols import SentenceAnalysisWriter

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
        self.default_limits = config.get("pipeline", {}).get("default_cardinality_limits", {})
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
            async with await Neo4jConnectionManager.get_session() as session:
                # Run the analysis writing operations directly on the session
                await self._run_write_result_queries(
                    session,
                    self.project_id,
                    self.interview_id,
                    sentence_id,
                    result,
                    is_error_result,
                    self.default_limits,
                )
            logger.debug(f"Successfully wrote analysis result for Sentence ID: {sentence_id}")
        except Exception as e:
            logger.error(
                f"Failed Neo4j write_result for Sentence ID {sentence_id} (Interview {self.interview_id}): {e}",
                exc_info=True,
            )
            raise

    # Session-based function for write_result
    async def _run_write_result_queries(
        self,
        session,
        project_id: str,
        interview_id: str,
        sentence_id: int,
        result: Dict[str, Any],
        is_error_result: bool,
        default_limits: Dict,
    ):
        logger.debug(f"Running write_result transaction for Sentence ID: {sentence_id}")

        # 1. Find the Sentence node
        match_sentence_query = """
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: $sentence_id})
            RETURN s
        """
        sentence_node_result = await session.run(
            match_sentence_query, interview_id=interview_id, sentence_id=sentence_id
        )
        sentence_node = await sentence_node_result.single()

        if not sentence_node:
            logger.error(
                f"_write_result_tx: Sentence node {sentence_id} not found for Interview {interview_id}. "
                f"Cannot write analysis."
            )
            # Raise an error or just return? Raising is safer to signal inconsistency.
            raise ValueError(f"Sentence node {sentence_id} not found for Interview {interview_id}")

        # 2. MERGE the Analysis node linked to the Sentence
        #    Set properties on create.
        #    Store the raw error structure if it's an error result.
        analysis_props_on_create = {
            "analysis_id": str(uuid.uuid4()),
            "model_name": result.get(
                "model_name", config.get("openai", {}).get("model_name")
            ),  # Get model from result or config
            "created_at": datetime.datetime.now(),
            "is_edited": False,
            "error_data": (
                json.dumps(result) if is_error_result else None
            ),  # Store error details as JSON string if it's an error
        }
        # Filter out None values
        analysis_props_on_create = {k: v for k, v in analysis_props_on_create.items() if v is not None}

        merge_analysis_query = """
            MATCH (s:Sentence {sentence_id: $sentence_id})<-[:HAS_SENTENCE]-(:Interview {interview_id: $interview_id})
            MERGE (s)-[:HAS_ANALYSIS]->(a:Analysis)
            ON CREATE SET a += $props
            ON MATCH SET a += $props
            RETURN a
        """
        analysis_node_result = await session.run(
            merge_analysis_query,
            sentence_id=sentence_id,
            interview_id=interview_id,
            props=analysis_props_on_create,
        )
        analysis_node = (await analysis_node_result.single())["a"]  # Get the analysis node
        logger.debug(f"_write_result_tx: Merged Analysis node for Sentence {sentence_id}")

        # --- If it was an error result, stop here ---
        if is_error_result:
            logger.warning(
                f"_write_result_tx: Stored error information for Sentence {sentence_id}. "
                f"No further dimensions processed."
            )
            return  # Do not process dimensions for error results

        # --- Step 3: Handle Dimensions ---

        # Fetch project-specific cardinality overrides once
        project_limits = await self._fetch_project_limits(session, project_id)

        # Helper function calls for each dimension type
        # Function Type (Cardinality 1)
        await self._handle_dimension_link(
            session=session,
            analysis_node_id=analysis_node.element_id,  # Pass internal ID
            dimension_label="FunctionType",
            dimension_key="name",
            dimension_value=result.get("function_type"),
            relationship_type="HAS_FUNCTION",
            cardinality_limit=(
                project_limits.get("HAS_FUNCTION")
                if project_limits.get("HAS_FUNCTION") is not None
                else default_limits.get("HAS_FUNCTION", 1)
            ),  # Default 1
            props_on_create={},
        )

        # Structure Type (Assume Cardinality 1 for now)
        await self._handle_dimension_link(
            session=session,
            analysis_node_id=analysis_node.element_id,
            dimension_label="StructureType",
            dimension_key="name",
            dimension_value=result.get("structure_type"),
            relationship_type="HAS_STRUCTURE",
            cardinality_limit=(
                project_limits.get("HAS_STRUCTURE")
                if project_limits.get("HAS_STRUCTURE") is not None
                else default_limits.get("HAS_STRUCTURE", 1)
            ),  # Default 1
            props_on_create={},
        )

        # Purpose (Assume Cardinality 1 for now)
        await self._handle_dimension_link(
            session=session,
            analysis_node_id=analysis_node.element_id,
            dimension_label="Purpose",
            dimension_key="name",
            dimension_value=result.get("purpose"),
            relationship_type="HAS_PURPOSE",
            cardinality_limit=(
                project_limits.get("HAS_PURPOSE")
                if project_limits.get("HAS_PURPOSE") is not None
                else default_limits.get("HAS_PURPOSE", 1)
            ),  # Default 1
            props_on_create={},
        )

        # Topics (Cardinality Many)
        await self._handle_dimension_list_link(
            session=session,
            analysis_node_id=analysis_node.element_id,
            dimension_label="Topic",
            dimension_key="name",
            dimension_values=result.get("topics", []),  # Assuming 'topics' key holds a list of topic names
            relationship_type="MENTIONS_TOPIC",
            cardinality_limit=(
                project_limits.get("MENTIONS_TOPIC")
                if project_limits.get("MENTIONS_TOPIC") is not None
                else default_limits.get("MENTIONS_TOPIC", None)
            ),  # Default unlimited
            props_on_create=lambda v: {"name": v},  # Function to create props for new Topic node
            # We might need to handle level property for Topic here too
        )

        # Keywords (Cardinality N)
        await self._handle_dimension_list_link(
            session=session,
            analysis_node_id=analysis_node.element_id,
            dimension_label="Keyword",
            dimension_key="text",
            dimension_values=result.get(
                "overall_keywords", []
            ),  # Fixed: Use 'overall_keywords' to match SentenceAnalyzer output
            relationship_type="MENTIONS_OVERALL_KEYWORD",
            cardinality_limit=(
                project_limits.get("MENTIONS_OVERALL_KEYWORD")
                if project_limits.get("MENTIONS_OVERALL_KEYWORD") is not None
                else default_limits.get("MENTIONS_OVERALL_KEYWORD", 6)
            ),  # Default 6
            props_on_create=lambda v: {"text": v},
        )

        # Domain Keywords (Cardinality Many)
        await self._handle_dimension_list_link(
            session=session,
            analysis_node_id=analysis_node.element_id,
            dimension_label="DomainKeyword",
            dimension_key="text",
            dimension_values=result.get("domain_keywords", []),  # Assuming 'domain_keywords' key
            relationship_type="MENTIONS_DOMAIN_KEYWORD",
            cardinality_limit=(
                project_limits.get("MENTIONS_DOMAIN_KEYWORD")
                if project_limits.get("MENTIONS_DOMAIN_KEYWORD") is not None
                else default_limits.get("MENTIONS_DOMAIN_KEYWORD", None)
            ),  # Default unlimited
            props_on_create=lambda v: {
                "text": v,
                "is_custom": False,
            },  # Assume keywords from analysis are not custom? Or derive?
        )

    async def _fetch_project_limits(self, session, project_id: str) -> Dict[str, Optional[int]]:
        "Fetches cardinality limit overrides from the Project node."
        query = """
            MATCH (p:Project {project_id: $project_id})
            RETURN p.max_functions_limit as HAS_FUNCTION,
                   p.max_structures_limit as HAS_STRUCTURE,
                   p.max_purposes_limit as HAS_PURPOSE,
                   p.max_keywords_limit as MENTIONS_OVERALL_KEYWORD,
                   p.max_topics_limit as MENTIONS_TOPIC,
                   p.max_domain_keywords_limit as MENTIONS_DOMAIN_KEYWORD
        """
        result = await session.run(query, project_id=project_id)
        record = await result.single()
        return dict(record) if record else {}

    # Helper for single-value dimension links (handles Max 1 cardinality)
    async def _handle_dimension_link(
        self,
        session,
        *,
        analysis_node_id: int,
        dimension_label: str,
        dimension_key: str,
        dimension_value: Optional[str],
        relationship_type: str,
        cardinality_limit: Optional[int],
        props_on_create: Dict,
    ):
        if not dimension_value:
            # If value is None or empty, potentially remove existing link (if not edited)
            # For now, we just do nothing if no value provided.
            # TODO: Add logic to remove existing link if value is None and rel not edited?
            logger.debug(f"Skipping link {relationship_type} from Analysis {analysis_node_id}: No value provided.")
            return

        # Check cardinality limit - if it's 0, don't create any relationships
        if cardinality_limit is not None and cardinality_limit == 0:
            logger.debug(
                f"Skipping link {relationship_type} from Analysis {analysis_node_id}: " f"Cardinality limit is 0."
            )
            return

        # Create dimension node properties
        dim_props = {dimension_key: dimension_value}
        dim_props.update(props_on_create)  # Add extra props like level if needed

        # Cypher Query:
        # 1. MATCH the Analysis node by its internal ID.
        # 2. OPTIONAL MATCH any existing relationship of this type AND its target node.
        # 3. If relationship exists AND rel.is_edited = true, RETURN (do nothing).
        # 4. If relationship exists AND rel.is_edited = false, DELETE the relationship.
        # 5. MERGE the dimension node (ensures uniqueness based on key property).
        # 6. CREATE the new relationship from Analysis to Dimension node, set is_edited = false.
        query = f"""
        MATCH (a:Analysis) WHERE elementId(a) = $analysis_node_id
        OPTIONAL MATCH (a)-[r_old:{relationship_type}]->(d_old:{dimension_label})

        // Check if edit protection applies
        WITH a, r_old, d_old,
             CASE WHEN r_old IS NOT NULL AND r_old.is_edited = true THEN true ELSE false END as is_protected
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
            "props_on_create": props_on_create,
        }

        await session.run(query, params)
        logger.debug(
            f"Handled link {relationship_type} from Analysis {analysis_node_id} to {dimension_label} {dimension_value}"
        )

    # Helper for multi-value dimension links (handles Max N cardinality)
    async def _handle_dimension_list_link(
        self,
        session,
        *,
        analysis_node_id: int,
        dimension_label: str,
        dimension_key: str,
        dimension_values: List[str],
        relationship_type: str,
        cardinality_limit: Optional[int],
        props_on_create: Callable[[str], Dict[str, Any]],
    ):
        if not dimension_values:
            logger.debug(
                f"Skipping list link {relationship_type} from Analysis {analysis_node_id}: " f"No values provided."
            )
            # Optionally, remove all non-edited existing links if the input list is empty
            # query_delete_all_unedited = f"""
            # MATCH (a:Analysis)-[r:{relationship_type}]->(d:{dimension_label})
            # WHERE elementId(a) = $analysis_node_id AND (r.is_edited IS NULL OR r.is_edited = false)
            # DELETE r
            # """
            # tx.run(query_delete_all_unedited, analysis_node_id=analysis_node_id)
            # logger.debug(f"Removed existing unedited {relationship_type} links for Analysis "
            #              f"{analysis_node_id} due to empty input list.")
            return

        # Ensure uniqueness in input values
        unique_dimension_values = set(dimension_values)

        # 1. Get current state: existing linked values and their edit status
        query_get_existing = f"""
        MATCH (a:Analysis)-[r:{relationship_type}]->(d:{dimension_label})
        WHERE elementId(a) = $analysis_node_id
        RETURN d.{dimension_key} AS value, COALESCE(r.is_edited, false) AS edited
        """
        existing_results = await session.run(query_get_existing, analysis_node_id=analysis_node_id)

        existing_edited_values = set()
        existing_unedited_values = set()
        async for record in existing_results:
            if record["edited"]:
                existing_edited_values.add(record["value"])
            else:
                existing_unedited_values.add(record["value"])

        logger.debug(
            f"{relationship_type} existing state: Edited={existing_edited_values}, "
            f"Unedited={existing_unedited_values}"
        )

        # 2. Identify changes
        target_values_set = unique_dimension_values
        values_to_add_set = target_values_set - existing_edited_values - existing_unedited_values
        unedited_values_to_remove_set = existing_unedited_values - target_values_set

        # 3. Calculate available slots for new items (considering the limit and preserved edited items)
        slots_available = float("inf")  # Assume infinite slots if limit is None
        if cardinality_limit is not None and cardinality_limit >= 0:
            slots_available = max(0, cardinality_limit - len(existing_edited_values))

        logger.debug(
            f"{relationship_type}: Limit={cardinality_limit}, "
            f"Edited={len(existing_edited_values)}, AvailableSlots={slots_available}"
        )

        # 4. Delete obsolete unedited relationships
        if unedited_values_to_remove_set:
            query_delete_unedited = f"""
            MATCH (a:Analysis)-[r:{relationship_type}]->(d:{dimension_label})
            WHERE elementId(a) = $analysis_node_id
              AND (r.is_edited IS NULL OR r.is_edited = false)
              AND d.{dimension_key} IN $values_to_remove
            DELETE r
            """
            await session.run(
                query_delete_unedited,
                analysis_node_id=analysis_node_id,
                values_to_remove=list(unedited_values_to_remove_set),
            )
            logger.debug(
                f"Deleted {len(unedited_values_to_remove_set)} obsolete unedited {relationship_type} "
                f"links for Analysis {analysis_node_id}"
            )

        # 5. Add new relationships up to the available slot limit
        if values_to_add_set and slots_available > 0:
            # Preserve original order by filtering the original list and deduplicating
            seen = set()
            values_to_add_list = []
            for value in dimension_values:
                if value in values_to_add_set and value not in seen:
                    values_to_add_list.append(value)
                    seen.add(value)
            if slots_available != float("inf"):
                values_to_add_list = values_to_add_list[: int(slots_available)]  # Take only as many as allowed

            if values_to_add_list:
                # Process each value individually to handle props_on_create lambda
                for new_value in values_to_add_list:
                    # Compute properties for this specific value using the lambda
                    computed_props = props_on_create(new_value)

                    query_add_single = f"""
                    MATCH (a:Analysis) WHERE elementId(a) = $analysis_node_id
                    MERGE (d:{dimension_label} {{{dimension_key}: $new_value}})
                    ON CREATE SET d += $computed_props
                    CREATE (a)-[r:{relationship_type} {{is_edited: false}}]->(d)
                    """

                    await session.run(
                        query_add_single,
                        analysis_node_id=analysis_node_id,
                        new_value=new_value,
                        computed_props=computed_props,
                    )
                logger.debug(
                    f"Added {len(values_to_add_list)} new {relationship_type} links " f"for Analysis {analysis_node_id}"
                )
            else:
                logger.debug(
                    f"No new {relationship_type} links to add for Analysis {analysis_node_id} "
                    f"(limit reached or no new values)"
                )
        else:
            logger.debug(
                f"No new {relationship_type} links to add for Analysis {analysis_node_id} "
                f"(slots_available={slots_available}, values_to_add={values_to_add_set})"
            )

    async def finalize(self):
        """
        Finalizes the analysis writing operation (likely a no-op).
        """
        logger.debug(f"Finalizing Neo4j writer for Interview: {self.interview_id}")
        pass  # Sessions managed by context manager

    async def read_analysis_ids(self) -> Set[int]:
        """
        Reads sentence IDs that have associated Analysis nodes for the Interview.
        """
        logger.debug(f"Reading analysis IDs for Interview: {self.interview_id}")
        cypher = """
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
            // Ensure there is an Analysis node linked
            WHERE (s)-[:HAS_ANALYSIS]->(:Analysis)
            RETURN s.sentence_id AS sentenceId
        """

        sentence_ids: Set[int] = set()
        try:
            async with await Neo4jConnectionManager.get_session() as session:
                result = await session.run(cypher, interview_id=self.interview_id)
                # Process results asynchronously
                async for record in result:
                    s_id = record["sentenceId"]
                    if isinstance(s_id, int):
                        sentence_ids.add(s_id)
                    else:
                        logger.warning(
                            f"Read non-integer sentence_id ({s_id}) with analysis for Interview "
                            f"{self.interview_id}, skipping."
                        )

            logger.debug(
                f"Successfully read {len(sentence_ids)} unique sentence IDs with analysis "
                f"for Interview: {self.interview_id}"
            )
            return sentence_ids
        except Exception as e:
            logger.error(
                f"Failed to read analysis IDs for Interview {self.interview_id}: {e}",
                exc_info=True,
            )
            raise

    def get_identifier(self) -> str:
        """
        Returns the interview ID as the identifier for this writer context.
        """
        return self.interview_id
