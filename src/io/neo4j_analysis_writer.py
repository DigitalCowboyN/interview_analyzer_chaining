"""
src/io/neo4j_analysis_writer.py

Neo4j implementation of the SentenceAnalysisWriter protocol.

Note: Direct Neo4j writes have been removed as of M3.0.
The projection service is now the sole writer to Neo4j.
This class now only emits events - the projection service handles materialization.
"""

from typing import Any, Dict, Optional, Set

from src.utils.logger import get_logger
from src.utils.metrics import metrics_tracker
from src.utils.neo4j_driver import Neo4jConnectionManager

from .protocols import SentenceAnalysisWriter

logger = get_logger()


class Neo4jAnalysisWriter(SentenceAnalysisWriter):
    """
    Emits analysis events for the projection service to materialize in Neo4j.
    Direct Neo4j writes have been removed - projection service is the sole writer.
    """

    def __init__(self, project_id: str, interview_id: str, event_emitter=None, correlation_id=None):
        """
        Initializes the writer for a specific project and interview context.

        Args:
            project_id (str): The unique ID (UUID ideally) of the project.
            interview_id (str): The unique ID (UUID ideally) of the interview.
            event_emitter: PipelineEventEmitter for event sourcing.
            correlation_id: Optional correlation ID for event tracking.
        """
        if not project_id or not interview_id:
            raise ValueError("project_id and interview_id cannot be empty")
        self.project_id = str(project_id)
        self.interview_id = str(interview_id)
        self.event_emitter = event_emitter
        self.correlation_id = correlation_id

        if event_emitter is None:
            logger.warning(
                "Neo4jAnalysisWriter initialized without event_emitter. "
                "Event sourcing is required - projection service handles Neo4j writes."
            )

        logger.debug(f"Neo4jAnalysisWriter initialized for Project: {self.project_id}, Interview: {self.interview_id}")

    async def initialize(self):
        """
        Ensures the associated Project and Interview nodes exist.
        (Usually not strictly needed if MapStorage ran first, but good practice).
        """
        logger.debug(f"Initializing Neo4j writer for Interview: {self.interview_id}")
        # Rely on them being created by Neo4jMapStorage.initialize or projection service
        pass

    async def write_result(self, result: Dict[str, Any]):
        """
        Emits an AnalysisGenerated event for the projection service to handle.
        Direct Neo4j writes have been removed - projection service is the sole writer.

        Args:
            result (Dict[str, Any]): Dictionary containing analysis data, including
                                     'sentence_id' and analysis fields.
        """
        sentence_id = result.get("sentence_id")
        if sentence_id is None:
            logger.error(f"write_result missing 'sentence_id' in result: {result}")
            return

        # Check if the result is an error reported by the analysis service
        is_error_result = result.get("error", False)

        logger.debug(f"Processing analysis result for Sentence ID: {sentence_id} in Interview: {self.interview_id}")

        # Skip event emission for error results (projection service doesn't need them)
        if is_error_result:
            logger.warning(
                f"Skipping event emission for error result of sentence {sentence_id}: "
                f"{result.get('error_message', 'Unknown error')}"
            )
            return

        # Emit event for projection service to materialize in Neo4j
        if self.event_emitter:
            try:
                await self.event_emitter.emit_analysis_generated(
                    interview_id=self.interview_id,
                    sentence_index=sentence_id,
                    analysis_data=result,
                    correlation_id=self.correlation_id,
                )
                logger.info(f"✅ AnalysisGenerated event emitted for sentence {sentence_id}")
                metrics_tracker.increment_dual_write_event_first_success()

            except Exception as event_error:
                metrics_tracker.increment_dual_write_event_first_failure()
                logger.error(
                    f"❌ Failed to emit AnalysisGenerated event for sentence {sentence_id}: {event_error}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Event emission failed for sentence {sentence_id} analysis."
                ) from event_error
        else:
            logger.warning(
                f"No event_emitter configured - cannot emit event for sentence {sentence_id}. "
                "Projection service will not receive this analysis."
            )

    async def finalize(self):
        """
        Finalizes the analysis writing operation (no-op).
        """
        logger.debug(f"Finalizing Neo4j writer for Interview: {self.interview_id}")
        pass

    async def read_analysis_ids(self) -> Set[int]:
        """
        Reads sentence indices (sequence_order) that have associated Analysis nodes for the Interview.

        Note: In M2.8+, sentence nodes use UUID-based sentence_id, but we return the sequence_order (index)
        for compatibility with the existing interface.
        """
        logger.debug(f"Reading analysis IDs for Interview: {self.interview_id}")
        cypher = """
            MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
            WHERE (s)-[:HAS_ANALYSIS]->(:Analysis)
            RETURN s.sequence_order AS sentenceIndex
        """

        sentence_ids: Set[int] = set()
        try:
            async with await Neo4jConnectionManager.get_session() as session:
                result = await session.run(cypher, interview_id=self.interview_id)
                async for record in result:
                    s_index = record["sentenceIndex"]
                    if isinstance(s_index, int):
                        sentence_ids.add(s_index)
                    else:
                        logger.warning(
                            f"Read non-integer sequence_order ({s_index}) with analysis for Interview "
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
