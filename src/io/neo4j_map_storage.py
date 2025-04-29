"""
src/io/neo4j_map_storage.py

Neo4j implementation of the ConversationMapStorage protocol.
"""
from .protocols import ConversationMapStorage
from src.utils.neo4j_driver import Neo4jConnectionManager
from src.utils.logger import get_logger
from typing import List, Dict, Any, Set, Optional
import uuid
import datetime

logger = get_logger()

class Neo4jMapStorage(ConversationMapStorage):
    """
    Stores and retrieves conversation map data (Projects, Interviews, Sentences)
    in a Neo4j graph database.
    """
    def __init__(self, project_id: str, interview_id: str):
        """
        Initializes the storage manager for a specific project and interview.

        Args:
            project_id (str): The unique ID (UUID ideally) of the project.
            interview_id (str): The unique ID (UUID ideally) of the interview.
        """
        if not project_id or not interview_id:
            raise ValueError("project_id and interview_id cannot be empty")
        self.project_id = str(project_id)
        self.interview_id = str(interview_id)
        logger.debug(f"Neo4jMapStorage initialized for Project: {self.project_id}, Interview: {self.interview_id}")

    async def initialize(self):
        """
        Ensures the Project and Interview nodes exist and clears old data for the interview.
        Uses MERGE for Project and Interview creation/matching.
        Detaches and deletes any existing Sentence nodes linked to this Interview.
        """
        logger.debug(f"Initializing Neo4j storage for Interview: {self.interview_id}")
        
        # Use a transaction for multiple operations
        try:
            async with Neo4jConnectionManager.get_session() as session:
                # Using write_transaction ensures atomicity for the combined operation
                await session.write_transaction(self._initialize_tx)
            logger.info(f"Neo4j initialization complete for Interview: {self.interview_id}")
        except Exception as e:
            logger.error(f"Failed Neo4j initialization for Interview {self.interview_id}: {e}", exc_info=True)
            # Re-raise the exception to signal failure to the caller
            raise

    # Transaction function for initialize
    # Note: Transaction functions should not be async themselves when used with session.write_transaction
    @staticmethod
    def _initialize_tx(tx, project_id: str, interview_id: str):
        logger.debug(f"Running initialization transaction for Interview: {interview_id}")
        # 1. Ensure Project exists
        #    Set properties only on creation to avoid overwriting existing ones.
        #    Store created_at as ISO 8601 string or Neo4j DateTime.
        project_query = ("""
            MERGE (p:Project {project_id: $project_id})
            ON CREATE SET p.created_at = datetime(), p.created_by = 'system'
            RETURN p
        """)
        tx.run(project_query, project_id=project_id)
        logger.debug(f"_initialize_tx: Merged Project {project_id}")

        # 2. Ensure Interview exists and link it to Project
        #    Set properties on creation. 
        #    We need the original filename here - assuming it might be same as interview_id for now?
        #    Or it needs to be passed differently. Placeholder for filename.
        interview_query = ("""
            MATCH (p:Project {project_id: $project_id})
            MERGE (i:Interview {interview_id: $interview_id})
            ON CREATE SET i.created_at = datetime(), i.processed_by = 'system', i.filename = $interview_id
            MERGE (p)-[:CONTAINS_INTERVIEW]->(i)
            RETURN i
        """)
        tx.run(interview_query, project_id=project_id, interview_id=interview_id)
        logger.debug(f"_initialize_tx: Merged Interview {interview_id} and relationship to Project {project_id}")

        # 3. Delete old sentences and downstream analyses for this interview
        #    MATCH the interview node.
        #    OPTIONAL MATCH its sentences.
        #    With the sentences (if any), DETACH DELETE them.
        #    DETACH DELETE handles removing relationships before node deletion.
        delete_query = ("""
            MATCH (i:Interview {interview_id: $interview_id})
            OPTIONAL MATCH (i)-[:HAS_SENTENCE]->(s:Sentence)
            DETACH DELETE s
        """)
        result = tx.run(delete_query, interview_id=interview_id)
        summary = result.consume() # Consume result to get summary info
        logger.debug(f"_initialize_tx: Detached and deleted {summary.counters.nodes_deleted} old sentence nodes for Interview {interview_id}")

    async def write_entry(self, entry: Dict[str, Any]):
        """
        Writes a single sentence entry to the Neo4j graph, respecting edit flags.

        - Finds the parent Interview node.
        - Merges the Sentence node based on sentence_id.
        - Conditionally updates Sentence properties if Sentence.is_edited is false.
        - Manages :NEXT_SENTENCE and :FIRST_SENTENCE relationships based on sequence_order.

        Args:
            entry (Dict[str, Any]): Dictionary containing sentence data, including:
                                   'sentence_id' (int), 'sequence_order' (int), 
                                   'sentence' (str), and optional fields like 
                                   'start_time', 'end_time', 'speaker'.
        """
        required_keys = ["sentence_id", "sequence_order", "sentence"]
        if not all(key in entry for key in required_keys):
            logger.error(f"write_entry missing required keys in entry: {entry}")
            raise ValueError("Entry dict missing required keys (sentence_id, sequence_order, sentence)")
            
        sentence_id = entry["sentence_id"]
        logger.debug(f"Writing entry for Sentence ID: {sentence_id} in Interview: {self.interview_id}")
        
        try:
            async with Neo4jConnectionManager.get_session() as session:
                await session.write_transaction(self._write_entry_tx, self.interview_id, entry)
            logger.debug(f"Successfully wrote entry for Sentence ID: {sentence_id}")
        except Exception as e:
            logger.error(f"Failed Neo4j write_entry for Sentence ID {sentence_id} (Interview {self.interview_id}): {e}", exc_info=True)
            raise

    # Transaction function for write_entry
    @staticmethod
    def _write_entry_tx(tx, interview_id: str, entry: Dict[str, Any]):
        logger.debug(f"Running write_entry transaction for Sentence ID: {entry.get('sentence_id')}")
        
        # Extract data, providing defaults for optional fields
        s_id = entry["sentence_id"]
        s_order = entry["sequence_order"]
        s_text = entry["sentence"]
        s_start = entry.get("start_time") # Will be None if not present
        s_end = entry.get("end_time")
        s_speaker = entry.get("speaker")

        # Properties to set on creation
        create_props = {
            "sequence_order": s_order,
            "text": s_text,
            "start_time": s_start,
            "end_time": s_end,
            "speaker": s_speaker,
            "is_edited": False
        }
        # Filter out None values from create_props to avoid setting null properties explicitly
        create_props = {k: v for k, v in create_props.items() if v is not None}

        # Properties to set on match (conditionally)
        # Note: We only update if is_edited is false
        update_props = {
            "sequence_order": s_order,
            "text": s_text,
            "start_time": s_start,
            "end_time": s_end,
            "speaker": s_speaker
        }
        # Filter out None values from update_props
        update_props = {k: v for k, v in update_props.items() if v is not None}

        # 1. MERGE Sentence node and conditionally SET properties
        #    Ensure relationship to Interview exists
        merge_sentence_query = ("""
            MATCH (i:Interview {interview_id: $interview_id})
            MERGE (i)-[:HAS_SENTENCE]->(s:Sentence {sentence_id: $s_id})
            ON CREATE SET s += $create_props
            ON MATCH SET 
                s = CASE WHEN s.is_edited = false THEN apoc.map.merge(s, $update_props) ELSE s END
            RETURN s.is_edited as was_edited
        """) 
        # Note: Using apoc.map.merge requires APOC plugin installed in Neo4j.
        # Alternative without APOC:
        # ON MATCH
        #   SET s.sequence_order = CASE WHEN s.is_edited = false THEN $s_order ELSE s.sequence_order END,
        #       s.text = CASE WHEN s.is_edited = false THEN $s_text ELSE s.text END,
        #       s.start_time = CASE WHEN s.is_edited = false THEN $s_start ELSE s.start_time END,
        #       s.end_time = CASE WHEN s.is_edited = false THEN $s_end ELSE s.end_time END,
        #       s.speaker = CASE WHEN s.is_edited = false THEN $s_speaker ELSE s.speaker END

        result = tx.run(merge_sentence_query, 
                        interview_id=interview_id, 
                        s_id=s_id, 
                        create_props=create_props, 
                        update_props=update_props, 
                        # Pass individual props too if not using APOC alternative
                        # s_order=s_order, s_text=s_text, s_start=s_start, s_end=s_end, s_speaker=s_speaker
                        )
        # Check if the update was skipped due to is_edited flag
        # record = result.single()
        # if record and record['was_edited']:
        #    logger.warning(f"Sentence {s_id} update skipped, node was previously edited.")
        logger.debug(f"_write_entry_tx: Merged Sentence {s_id}")

        # 2. Manage sequence relationships (:FIRST_SENTENCE, :NEXT_SENTENCE)
        if s_order == 0:
            first_sentence_query = ("""
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: $s_id})
                MERGE (i)-[:FIRST_SENTENCE]->(s)
            """)
            tx.run(first_sentence_query, interview_id=interview_id, s_id=s_id)
            logger.debug(f"_write_entry_tx: Merged :FIRST_SENTENCE for Sentence {s_id}")
        else:
            prev_s_id = s_order - 1
            next_sentence_query = ("""
                MATCH (i:Interview {interview_id: $interview_id})
                MATCH (prev_s:Sentence {sentence_id: $prev_s_id})<-[:HAS_SENTENCE]-(i)
                MATCH (s:Sentence {sentence_id: $s_id})<-[:HAS_SENTENCE]-(i)
                MERGE (prev_s)-[:NEXT_SENTENCE]->(s)
            """)
            tx.run(next_sentence_query, interview_id=interview_id, prev_s_id=prev_s_id, s_id=s_id)
            logger.debug(f"_write_entry_tx: Merged :NEXT_SENTENCE from {prev_s_id} to {s_id}")

    async def finalize(self):
        """
        Finalizes the storage operation (likely a no-op for managed sessions).
        Optionally update Interview metadata here.
        """
        logger.debug(f"Finalizing Neo4j storage for Interview: {self.interview_id}")
        pass # Sessions managed by context manager

    async def read_all_entries(self) -> List[Dict[str, Any]]:
        """
        Reads all sentence entries for the associated Interview from Neo4j,
        ordered by sequence_order.
        """
        logger.debug(f"Reading all entries for Interview: {self.interview_id}")
        cypher = ("""
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
            RETURN s
            ORDER BY s.sequence_order ASC
        """)
        
        entries: List[Dict[str, Any]] = []
        try:
            async with Neo4jConnectionManager.get_session() as session:
                result = await session.run(cypher, interview_id=self.interview_id)
                # Process results asynchronously
                async for record in result:
                    sentence_node = record["s"]
                    # Convert the Node object to a dictionary of its properties
                    # Handle potential Neo4j DateTime/Date/Time types if necessary
                    entry_data = dict(sentence_node)
                    # Ensure required fields are present (optional, depends on strictness)
                    # if not all(k in entry_data for k in ['sentence_id', 'sequence_order', 'sentence']):
                    #    logger.warning(f"Skipping entry with missing required fields: {entry_data.get('sentence_id')}")
                    #    continue
                    entries.append(entry_data)
            
            logger.debug(f"Successfully read {len(entries)} entries for Interview: {self.interview_id}")
            return entries
        except Exception as e:
            logger.error(f"Failed to read entries for Interview {self.interview_id}: {e}", exc_info=True)
            # Re-raise or return empty list depending on desired error handling
            raise

    async def read_sentence_ids(self) -> Set[int]:
        """
        Reads all unique sentence IDs for the associated Interview from Neo4j.
        """
        logger.debug(f"Reading sentence IDs for Interview: {self.interview_id}")
        cypher = ("""
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
            RETURN s.sentence_id AS sentenceId
        """)
        # Note: Using RETURN DISTINCT s.sentence_id might be slightly more efficient 
        # if the driver/db doesn't optimize the set creation well, but adding to a Python set handles uniqueness.

        sentence_ids: Set[int] = set()
        try:
            async with Neo4jConnectionManager.get_session() as session:
                result = await session.run(cypher, interview_id=self.interview_id)
                # Process results asynchronously
                async for record in result:
                    s_id = record["sentenceId"]
                    # Ensure the ID is an integer before adding
                    if isinstance(s_id, int):
                        sentence_ids.add(s_id)
                    else:
                        logger.warning(f"Found non-integer sentence_id ({s_id}) for Interview {self.interview_id}, skipping.")
                        
            logger.debug(f"Successfully read {len(sentence_ids)} unique sentence IDs for Interview: {self.interview_id}")
            return sentence_ids
        except Exception as e:
            logger.error(f"Failed to read sentence IDs for Interview {self.interview_id}: {e}", exc_info=True)
            # Re-raise or return empty set depending on desired error handling
            raise

    def get_identifier(self) -> str:
        """
        Returns the interview ID as the identifier for this storage context.
        """
        return self.interview_id
