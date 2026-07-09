"""Layer 2 enrichment orchestrator.

Loads a Layer 1 interview (speakers, utterances, fragments), runs the extractor
registry over it, and emits enrichment events through the event-sourced
repositories. Resume-aware: skips fragments already analyzed unless forced.
"""

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.enrichment.embedder import encode_vector, get_embedder
from src.enrichment.executor import EnrichmentExecutor
from src.enrichment.graph_context import FragmentView, GraphContextBuilder
from src.enrichment.registry import ExtractorRegistry
from src.events.aggregates import Interview, Sentence
from src.events.envelope import Actor, ActorType, generate_correlation_id
from src.events.repository import get_interview_repository, get_sentence_repository
from src.utils.logger import get_logger

logger = get_logger()

ENRICHMENT_MODEL_VERSION = "m4.2"


class EnrichmentResult(BaseModel):
    """Summary of one enriched interview."""

    interview_id: str
    fragments_enriched: int
    fragments_skipped: int
    entities_extracted: int
    claims_extracted: int
    embeddings_generated: int


class EnrichmentOrchestrator:
    """Runs the extractor registry over one interview and emits events."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        from src.config import config as global_config

        self.config = config_dict if config_dict is not None else global_config

    def _build_executor(self) -> EnrichmentExecutor:
        """Construct the executor (patchable seam for tests)."""
        from src.agents.failover_agent import get_failover_agent
        from src.utils.helpers import load_yaml

        agent = get_failover_agent(self.config)
        specs = ExtractorRegistry.load("config/extractors.yaml")
        prompts = load_yaml("prompts/core_extractors.yaml")
        domain_keywords = self.config.get("domain_keywords", [])
        concurrency = self.config.get("pipeline", {}).get("num_analysis_workers", 10)
        return EnrichmentExecutor(agent, specs, prompts, domain_keywords, concurrency)

    async def enrich_interview(
        self, interview_id: str, correlation_id: Optional[str] = None, force: bool = False
    ) -> EnrichmentResult:
        """Enrich every (or every un-analyzed) fragment and utterance."""
        correlation_id = correlation_id or generate_correlation_id()
        actor = Actor(actor_type=ActorType.SYSTEM, user_id="enrichment")

        interview_repo = get_interview_repository()
        interview = await interview_repo.load(interview_id)
        if interview is None:
            raise ValueError(f"Interview {interview_id} not found")

        sentence_repo = get_sentence_repository()
        fragment_count = interview.metadata.get("fragment_count", 0)

        # Load all fragments; select which to enrich (resume-awareness).
        all_sentences: Dict[int, Sentence] = {}
        to_enrich: List[Sentence] = []
        skipped = 0
        for index in range(fragment_count):
            sid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{index}"))
            sentence = await sentence_repo.load(sid)
            if sentence is None:
                continue
            all_sentences[index] = sentence
            if sentence.analysis_model is not None and not force:
                skipped += 1
            else:
                to_enrich.append(sentence)

        speaker_handle = {sid: info["handle"] for sid, info in interview.speakers.items()}
        utterance_of_fragment = {
            fid: uid
            for uid, u in interview.utterances.items()
            if not u.get("removed")
            for fid in u["fragment_ids"]
        }

        # Context windows must see EVERY fragment (skipped neighbors included);
        # only the to-enrich subset is sent to the executor. Building windows
        # from the subset would corrupt prompts exactly in resume mode.
        all_views = [
            FragmentView(
                index=s.index,
                text=s.text,
                speaker_handle=speaker_handle.get(s.speaker_id, "S?"),
                utterance_id=utterance_of_fragment.get(s.aggregate_id),
            )
            for s in sorted(all_sentences.values(), key=lambda s: s.index)
        ]
        utterance_texts = self._utterance_texts(interview, all_sentences)
        all_contexts = GraphContextBuilder(
            self.config.get("preprocessing", {}).get("context_windows", {})
        ).build_all(all_views, utterance_texts)

        enrich_indices = {s.index for s in to_enrich}
        fragment_views = [v for v in all_views if v.index in enrich_indices]
        contexts = [all_contexts[pos] for pos, v in enumerate(all_views) if v.index in enrich_indices]

        executor = self._build_executor()
        embedder = get_embedder(self.config)

        entities_count, embeddings_count = await self._emit_fragment_results(
            executor, embedder, to_enrich, fragment_views, contexts, sentence_repo,
            actor, correlation_id,
        )
        claims_count, utt_embeddings = await self._emit_utterance_results(
            executor, embedder, interview, utterance_texts, actor, correlation_id, force,
        )
        await interview_repo.save(interview)

        logger.info(
            f"Enriched interview {interview_id}: {len(to_enrich)} fragments "
            f"({skipped} skipped), {entities_count} entity sets, {claims_count} claims, "
            f"{embeddings_count + utt_embeddings} embeddings"
        )
        return EnrichmentResult(
            interview_id=interview_id,
            fragments_enriched=len(to_enrich),
            fragments_skipped=skipped,
            entities_extracted=entities_count,
            claims_extracted=claims_count,
            embeddings_generated=embeddings_count + utt_embeddings,
        )

    def _utterance_texts(
        self, interview: Interview, all_sentences: Dict[int, Sentence]
    ) -> Dict[str, str]:
        """Join member-fragment texts (in fragment_ids order) per utterance."""
        by_aggregate = {s.aggregate_id: s for s in all_sentences.values()}
        texts: Dict[str, str] = {}
        for uid, u in interview.utterances.items():
            if u.get("removed"):
                continue
            parts = [
                by_aggregate[fid].text
                for fid in u["fragment_ids"]
                if fid in by_aggregate
            ]
            texts[uid] = " ".join(parts)
        return texts

    async def _emit_fragment_results(
        self, executor, embedder, to_enrich, fragment_views, contexts,
        sentence_repo, actor, correlation_id,
    ):
        entities_count = 0
        embeddings_count = 0
        if not to_enrich:
            return entities_count, embeddings_count

        enrichments = await executor.enrich_fragments(fragment_views, contexts)
        vectors = await embedder.embed([s.text for s in to_enrich])
        by_index = {s.index: s for s in to_enrich}

        for enrichment, vector in zip(enrichments, vectors):
            sentence = by_index[enrichment.index]
            if not enrichment.model:
                # Every call for this fragment failed (total provider outage).
                # Emit NOTHING so analysis_model stays None and the fragment is
                # retried on the next run instead of being sealed as analyzed.
                logger.warning(
                    f"Fragment {enrichment.index}: all extractor calls failed "
                    f"({list(enrichment.flags)}); leaving un-analyzed for retry"
                )
                continue
            confidences = list(enrichment.dimension_confidences.values())
            mean_conf = sum(confidences) / len(confidences) if confidences else None
            sentence.generate_analysis(
                model=enrichment.model,
                model_version=ENRICHMENT_MODEL_VERSION,
                classification=enrichment.classification,
                keywords=enrichment.keywords,
                domain_keywords=enrichment.domain_keywords,
                confidence=mean_conf,
                dimension_confidences=enrichment.dimension_confidences,
                flags=enrichment.flags,
                provider=enrichment.provider,
                actor=actor,
                correlation_id=correlation_id,
            )
            if enrichment.entities:
                sentence.record_entities(
                    enrichment.entities, model=enrichment.model, provider=enrichment.provider,
                    actor=actor, correlation_id=correlation_id,
                )
                entities_count += 1
            sentence.record_embedding(
                model=embedder.model_name, dim=embedder.dim, vector_b64=encode_vector(vector),
                actor=actor, correlation_id=correlation_id,
            )
            embeddings_count += 1
            await sentence_repo.save(sentence)
        return entities_count, embeddings_count

    async def _emit_utterance_results(
        self, executor, embedder, interview, utterance_texts, actor, correlation_id,
        force: bool = False,
    ):
        claims_count = 0
        utt_embeddings = 0

        # Resume-awareness (utterance side): skip utterances that already have
        # claims / embeddings unless forced — a re-run must never crash on the
        # deterministic claim_id or burn LLM calls on finished work.
        claimed_utterances = {c["utterance_id"] for c in interview.claims.values()}
        to_extract = {
            uid: text
            for uid, text in utterance_texts.items()
            if force or uid not in claimed_utterances
        }
        if not utterance_texts:
            return claims_count, utt_embeddings

        if to_extract:
            enrichments = await executor.enrich_utterances(to_extract)
            for enrichment in enrichments:
                for ordinal, claim in enumerate(enrichment.claims):
                    claim_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{interview.aggregate_id}:claim:{enrichment.utterance_id}:{ordinal}",
                        )
                    )
                    if claim_id in interview.claims:
                        # Deterministic id already recorded (forced re-run):
                        # idempotent skip, never a crash.
                        logger.info(f"Claim {claim_id} already recorded; skipping")
                        continue
                    interview.record_claim(
                        claim_id,
                        enrichment.utterance_id,
                        claim["text"],
                        claim["kind"],
                        claim["confidence"],
                        enrichment.model,
                        enrichment.provider,
                        actor=actor,
                        correlation_id=correlation_id,
                    )
                    claims_count += 1

        # Embed each utterance's joined text (skip already-embedded unless forced).
        uids = [
            uid for uid in utterance_texts
            if force or uid not in interview.utterance_embeddings
        ]
        if uids:
            vectors = await embedder.embed([utterance_texts[u] for u in uids])
            for uid, vector in zip(uids, vectors):
                interview.record_utterance_embedding(
                    uid, model=embedder.model_name, dim=embedder.dim,
                    vector_b64=encode_vector(vector), actor=actor, correlation_id=correlation_id,
                )
                utt_embeddings += 1
        return claims_count, utt_embeddings
