"""Ask engine (M4.6): retrieve → fuse → assemble → one synthesis call.

Read-side only. Degradation doctrine: a dead embedder drops the vector
channel (flagged), zero hits skip the LLM entirely, and synthesis failure
raises SynthesisUnavailable still carrying the full retrieval result.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from src.ask import context as ctx
from src.ask import fusion, reader
from src.projections.handlers.embedding_handlers import _sanitize
from src.utils.logger import get_logger
from src.utils.neo4j_driver import Neo4jConnectionManager

logger = get_logger()

CHANNEL_K = 20  # per-channel candidate depth before fusion


class AskCitation(BaseModel):
    fragment_id: str


class AskAnswer(BaseModel):
    answer: str
    citations: List[AskCitation]


class AskResult(BaseModel):
    project_id: str
    question: str
    answer: Optional[str]
    citations: List[Dict[str, str]]
    retrieval: Dict[str, Any]
    provider: str = ""
    model: str = ""


class SynthesisUnavailable(Exception):
    """LLM chain failed or returned an invalid response; retrieval survives."""

    def __init__(self, message: str, result: AskResult):
        super().__init__(message)
        self.result = result


class AskEngine:
    """Answers a question against one project's graph."""

    _fulltext_ensured: bool = False  # per-process; ensure_fulltext_index is idempotent DDL

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        from src.config import config as global_config

        self.config = config_dict if config_dict is not None else global_config

    def _build_embedder(self):
        """Patchable seam (resolution-engine precedent)."""
        from src.enrichment.embedder import get_embedder

        return get_embedder(self.config)

    def _build_agent(self):
        """Patchable seam."""
        from src.agents.failover_agent import get_failover_agent

        return get_failover_agent(self.config)

    def _match_names(self, question: str, names: List[Dict[str, Any]]):
        """Deterministic query analysis: which canonicals/persons does the
        question mention (name or surface, case-insensitive substring)?"""
        q = question.lower()
        canonical_ids, person_ids = [], []
        for row in names:
            needles = [row["name"], *row.get("surfaces", [])]
            if any(n and n.lower() in q for n in needles):
                (canonical_ids if row["kind"] == "entity" else person_ids).append(row["id"])
        return canonical_ids, person_ids

    async def ask(self, project_id: str, question: str, top_k: int = 12) -> AskResult:
        flags: Dict[str, str] = {}
        rankings: Dict[str, List[str]] = {}

        async with await Neo4jConnectionManager.get_session() as session:
            if not await reader.project_exists(session, project_id):
                raise ValueError(f"Project {project_id} not found")
            if not AskEngine._fulltext_ensured:
                await reader.ensure_fulltext_index(session)
                AskEngine._fulltext_ensured = True

            # vector channel (embedder failure drops it; a single dead index
            # degrades to the other index's hits instead of killing the channel)
            vector = None
            try:
                embedder = self._build_embedder()
                vector = (await embedder.embed([question]))[0]
                model = _sanitize(embedder.model_name)
            except Exception as exc:
                logger.warning(f"ask: vector channel unavailable ({type(exc).__name__})")
                flags["vector_unavailable"] = type(exc).__name__

            if vector is not None:
                frag_rows: List[Dict[str, Any]] = []
                utt_rows: List[Dict[str, Any]] = []
                try:
                    frag_rows = await reader.vector_fragment_rows(
                        session, project_id, f"fragment_embedding_{model}", vector, CHANNEL_K
                    )
                except Exception as exc:
                    logger.warning(f"ask: fragment vector index unavailable ({type(exc).__name__})")
                    flags["vector_fragment_unavailable"] = type(exc).__name__
                try:
                    utt_rows = await reader.vector_utterance_rows(
                        session, project_id, f"utterance_embedding_{model}", vector, CHANNEL_K
                    )
                except Exception as exc:
                    logger.warning(f"ask: utterance vector index unavailable ({type(exc).__name__})")
                    flags["vector_utterance_unavailable"] = type(exc).__name__
                if "vector_fragment_unavailable" in flags and "vector_utterance_unavailable" in flags:
                    flags["vector_unavailable"] = flags["vector_fragment_unavailable"]
                seen: Dict[str, float] = {}
                for row in frag_rows + utt_rows:
                    seen[row["fragment_id"]] = max(
                        seen.get(row["fragment_id"], 0.0), row["score"]
                    )
                if seen:
                    rankings["vector"] = [
                        fid for fid, _ in sorted(seen.items(), key=lambda x: (-x[1], x[0]))
                    ]

            # fulltext channel
            query_text = reader.sanitize_fulltext_query(question)
            if query_text:
                ft_rows = await reader.fulltext_rows(
                    session, project_id, query_text, CHANNEL_K
                )
                rankings["fulltext"] = [r["fragment_id"] for r in ft_rows]

            # graph channel (anchored on names the question mentions)
            names = await reader.name_rows(session, project_id)
            canonical_ids, person_ids = self._match_names(question, names)
            if canonical_ids or person_ids:
                anchor_rows = await reader.graph_anchor_rows(
                    session, project_id, canonical_ids, person_ids
                )
                rankings["graph"] = fusion.rank_by_count(anchor_rows)

            fragment_ids = fusion.top_k(fusion.rrf_merge(rankings), top_k)
            rows = await reader.context_rows(session, fragment_ids) if fragment_ids else []

        blocks = ctx.build_blocks(rows)
        retrieval = {
            "channels": {name: len(ranked) for name, ranked in rankings.items()},
            "flags": flags,
            "fragments": [b["fragment_id"] for b in blocks],
        }

        if not blocks:
            return AskResult(
                project_id=project_id, question=question,
                answer="No grounding found in this project for that question.",
                citations=[], retrieval=retrieval,
            )

        from src.utils.helpers import load_yaml

        template = load_yaml("prompts/ask_prompts.yaml")["ask_synthesis"]["prompt"]
        prompt = ctx.render_prompt(template, question, blocks)
        partial = AskResult(
            project_id=project_id, question=question, answer=None,
            citations=[], retrieval=retrieval,
        )
        try:
            call_result = await self._build_agent().call(
                prompt, schema=AskAnswer.model_json_schema()
            )
            parsed = AskAnswer.model_validate(call_result.data)
        except ValidationError as exc:
            raise SynthesisUnavailable(f"invalid synthesis response: {exc.error_count()} errors", partial)
        except Exception as exc:
            raise SynthesisUnavailable(f"synthesis failed: {type(exc).__name__}", partial)

        citations = ctx.quotes_for([c.fragment_id for c in parsed.citations], blocks)
        return AskResult(
            project_id=project_id, question=question, answer=parsed.answer,
            citations=citations, retrieval=retrieval,
            provider=call_result.provider, model=call_result.model,
        )
