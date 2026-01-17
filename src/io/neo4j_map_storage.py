"""
src/io/neo4j_map_storage.py

Neo4j implementation of the ConversationMapStorage protocol.
"""

import warnings
from typing import Any, Dict, List, Set

from src.utils.logger import get_logger
from src.utils.metrics import metrics_tracker
from src.utils.neo4j_driver import Neo4jConnectionManager

from .protocols import ConversationMapStorage

logger = get_logger()


class Neo4jMapStorage(ConversationMapStorage):
    """
    Stores and retrieves conversation map data (Projects, Interviews, Sentences)
    in a Neo4j graph database.
    """

    def __init__(self, project_id: str, interview_id: str, event_emitter=None, correlation_id=None):
        """
        Initializes the storage manager for a specific project and interview.

        Args:
            project_id (str): The unique ID (UUID ideally) of the project.
            interview_id (str): The unique ID (UUID ideally) of the interview.
            event_emitter: Optional PipelineEventEmitter for dual-write phase.
            correlation_id: Optional correlation ID for event tracking.
        """
        if not project_id or not interview_id:
            raise ValueError("project_id and interview_id cannot be empty")
        self.project_id = str(project_id)
        self.interview_id = str(interview_id)
        self.event_emitter = event_emitter
        self.correlation_id = correlation_id

        # M2.8: Deprecation warning for direct write path
        if event_emitter is None:
            warnings.warn(
                "Direct Neo4j writes without event_emitter are deprecated and will be removed in M3.0. "
                "Pass an event_emitter instance to enable event-first dual-write architecture. "
                "See M2.8_MIGRATION_SUMMARY.md for migration guidance.",
                DeprecationWarning,
                stacklevel=2
            )
            logger.warning(
                "DEPRECATED: Neo4jMapStorage initialized without event_emitter. "
                "Direct writes will be removed in M3.0. Use event-first dual-write pattern."
            )

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
            async with await Neo4jConnectionManager.get_session() as session:
                # Create Project and Interview nodes, clear old data
                await self._run_initialization_queries(session)
            logger.info(f"Neo4j initialization complete for Interview: {self.interview_id}")
        except Exception as e:
            logger.error(
                f"Failed Neo4j initialization for Interview {self.interview_id}: {e}",
                exc_info=True,
            )
            # Re-raise the exception to signal failure to the caller
            raise

    async def _run_initialization_queries(self, session):
        """Run initialization queries directly on the session."""
        logger.debug(f"Running initialization queries for Interview: {self.interview_id}")
        # 1. Ensure Project exists
        project_query = """
            MERGE (p:Project {project_id: $project_id})
            ON CREATE SET p.created_at = datetime(), p.created_by = 'system'
            RETURN p
        """
        await session.run(project_query, project_id=self.project_id)
        logger.debug(f"Merged Project {self.project_id}")

        # 2. Ensure Interview exists and link it to Project
        # Direct writes set event_version=0 (projection service will update to actual version)
        interview_query = """
            MATCH (p:Project {project_id: $project_id})
            MERGE (i:Interview {interview_id: $interview_id})
            ON CREATE SET
                i.created_at = datetime(),
                i.processed_by = 'system',
                i.filename = $interview_id,
                i.event_version = 0,
                i.source = 'pipeline_direct'
            MERGE (p)-[:CONTAINS_INTERVIEW]->(i)
            RETURN i
        """
        await session.run(interview_query, project_id=self.project_id, interview_id=self.interview_id)
        logger.debug(f"Merged Interview {self.interview_id} and relationship to Project {self.project_id}")

        # 3. Delete old sentences and downstream analyses for this interview
        delete_query = """
            MATCH (i:Interview {interview_id: $interview_id})
            OPTIONAL MATCH (i)-[:HAS_SENTENCE]->(s:Sentence)
            DETACH DELETE s
        """
        result = await session.run(delete_query, interview_id=self.interview_id)
        summary = await result.consume()  # Consume result to get summary info
        logger.debug(
            f"Detached and deleted {summary.counters.nodes_deleted} old sentence nodes "
            f"for Interview {self.interview_id}"
        )

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

        # STEP 1: EMIT EVENT FIRST (CRITICAL - Event-First Dual-Write)
        # Events are the source of truth. If event emission fails, abort the operation.
        if self.event_emitter:
            try:
                await self.event_emitter.emit_sentence_created(
                    interview_id=self.interview_id,
                    index=entry["sequence_order"],
                    text=entry["sentence"],
                    speaker=entry.get("speaker"),
                    start_ms=entry.get("start_time"),
                    end_ms=entry.get("end_time"),
                    correlation_id=self.correlation_id,
                )
                logger.info(f"✅ Event emitted for sentence {sentence_id}")

                # Track event-first success metric
                metrics_tracker.increment_dual_write_event_first_success()

            except Exception as event_error:
                # EVENT FAILURE = OPERATION FAILURE
                # This is architecturally correct: events are the immutable source of truth.
                # If we cannot persist the event, we must not write to Neo4j.

                # Track event-first failure metric
                metrics_tracker.increment_dual_write_event_first_failure()

                logger.error(
                    f"❌ CRITICAL: Failed to emit SentenceCreated event for sentence {sentence_id}. "
                    f"Aborting Neo4j write to maintain event-first integrity. Error: {event_error}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Event emission failed for sentence {sentence_id}. "
                    f"Operation aborted to maintain event-first integrity."
                ) from event_error

        # M2.8: REMOVED DIRECT NEO4J WRITES
        # Projection service is now the SOLE writer to Neo4j.
        # Events are the single source of truth - projection service reads events and writes to Neo4j.
        logger.info(f"✅ Event emitted for sentence {sentence_id}. Projection service will handle Neo4j write.")

    async def _run_write_entry_queries(self, session, entry: Dict[str, Any]):
        """Run write entry queries directly on the session."""
        logger.debug(f"Running write_entry queries for Sentence ID: {entry.get('sentence_id')}")

        # Extract data, providing defaults for optional fields
        s_id = entry["sentence_id"]
        s_order = entry["sequence_order"]
        s_text = entry["sentence"]
        s_start = entry.get("start_time")
        s_end = entry.get("end_time")
        s_speaker = entry.get("speaker")

        # Properties to set on creation
        # Note: Direct writes get event_version=0 (placeholder before projection service updates it)
        # Projection service will set event_version to actual event version when it processes events
        create_props = {
            "sequence_order": s_order,
            "text": s_text,
            "start_time": s_start,
            "end_time": s_end,
            "speaker": s_speaker,
            "is_edited": False,
            "event_version": 0,  # Direct writes are version 0 (before projection)
            "source": "pipeline_direct",  # Track source of write
        }
        # Filter out None values from create_props
        create_props = {k: v for k, v in create_props.items() if v is not None}

        # Properties to set on match (conditionally)
        update_props = {
            "sequence_order": s_order,
            "text": s_text,
            "start_time": s_start,
            "end_time": s_end,
            "speaker": s_speaker,
        }
        # Filter out None values from update_props
        update_props = {k: v for k, v in update_props.items() if v is not None}

        # 1. MERGE Sentence node and conditionally SET properties
        merge_sentence_query = """
            MATCH (i:Interview {interview_id: $interview_id})
            MERGE (i)-[:HAS_SENTENCE]->(s:Sentence {sentence_id: $s_id})
            ON CREATE SET s += $create_props
            ON MATCH SET
                s.sequence_order = CASE WHEN s.is_edited = false THEN $sequence_order ELSE s.sequence_order END,
                s.text = CASE WHEN s.is_edited = false THEN $text ELSE s.text END,
                s.start_time = CASE WHEN s.is_edited = false THEN $start_time ELSE s.start_time END,
                s.end_time = CASE WHEN s.is_edited = false THEN $end_time ELSE s.end_time END,
                s.speaker = CASE WHEN s.is_edited = false THEN $speaker ELSE s.speaker END
            RETURN s.is_edited as was_edited
        """
        await session.run(
            merge_sentence_query,
            interview_id=self.interview_id,
            s_id=s_id,
            create_props=create_props,
            sequence_order=s_order,
            text=s_text,
            start_time=s_start,
            end_time=s_end,
            speaker=s_speaker,
        )
        logger.debug(f"Merged Sentence {s_id}")

        # 2. Manage sequence relationships (:FIRST_SENTENCE, :NEXT_SENTENCE)
        if s_order == 0:
            first_sentence_query = """
                MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence {sentence_id: $s_id})
                MERGE (i)-[:FIRST_SENTENCE]->(s)
            """
            await session.run(first_sentence_query, interview_id=self.interview_id, s_id=s_id)
            logger.debug(f"Merged :FIRST_SENTENCE for Sentence {s_id}")
        else:
            prev_sequence_order = s_order - 1
            next_sentence_query = """
                MATCH (i:Interview {interview_id: $interview_id})
                MATCH (prev_s:Sentence {sequence_order: $prev_sequence_order})<-[:HAS_SENTENCE]-(i)
                MATCH (s:Sentence {sentence_id: $s_id})<-[:HAS_SENTENCE]-(i)
                MERGE (prev_s)-[:NEXT_SENTENCE]->(s)
            """
            await session.run(
                next_sentence_query,
                interview_id=self.interview_id,
                prev_sequence_order=prev_sequence_order,
                s_id=s_id,
            )
            logger.debug(f"Merged :NEXT_SENTENCE from sequence_order {prev_sequence_order} to sentence_id {s_id}")

    async def finalize(self):
        """
        Finalizes the storage operation (likely a no-op for managed sessions).
        Optionally update Interview metadata here.
        """
        logger.debug(f"Finalizing Neo4j storage for Interview: {self.interview_id}")
        pass  # Sessions managed by context manager

    async def read_all_entries(self) -> List[Dict[str, Any]]:
        """
        Reads all sentence entries for the associated Interview from Neo4j,
        ordered by sequence_order.
        """
        logger.debug(f"Reading all entries for Interview: {self.interview_id}")
        cypher = """
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
            RETURN s
            ORDER BY s.sequence_order ASC
        """

        entries: List[Dict[str, Any]] = []
        try:
            async with await Neo4jConnectionManager.get_session() as session:
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
            logger.error(
                f"Failed to read entries for Interview {self.interview_id}: {e}",
                exc_info=True,
            )
            # Re-raise or return empty list depending on desired error handling
            raise

    async def read_sentence_ids(self) -> Set[str]:
        """
        M2.8: Reads all unique sentence IDs (UUIDs) for the associated Interview from Neo4j.

        Note: In M2.8, sentence_ids are UUID strings generated by the projection service.
        """
        logger.debug(f"Reading sentence IDs for Interview: {self.interview_id}")
        cypher = """
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
            RETURN s.sentence_id AS sentenceId
        """

        sentence_ids: Set[str] = set()
        try:
            async with await Neo4jConnectionManager.get_session() as session:
                result = await session.run(cypher, interview_id=self.interview_id)
                # Process results asynchronously
                async for record in result:
                    s_id = record["sentenceId"]
                    # M2.8: sentence_ids are now strings (UUIDs)
                    if s_id is not None:
                        sentence_ids.add(str(s_id))
                    else:
                        logger.warning(
                            f"Found null sentence_id for Interview {self.interview_id}, skipping."
                        )

            logger.debug(
                f"Successfully read {len(sentence_ids)} unique sentence IDs for Interview: {self.interview_id}"
            )
            return sentence_ids
        except Exception as e:
            logger.error(
                f"Failed to read sentence IDs for Interview {self.interview_id}: {e}",
                exc_info=True,
            )
            # Re-raise or return empty set depending on desired error handling
            raise

    def get_identifier(self) -> str:
        """
        Returns the interview ID as the identifier for this storage context.
        """
        return self.interview_id
