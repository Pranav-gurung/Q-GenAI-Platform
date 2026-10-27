"""RAGService: ground the RCA's corrective actions in retrieved SOP sections. Never raises.

  RAG disabled / KB / index / embeddings unavailable   -> UNAVAILABLE         (RCA untouched)
  no defect                                            -> NOT_APPLICABLE
  nothing above RAG_MIN_SIMILARITY                     -> NO_RELEVANT_SOURCE  (no LLM call)
  LLM disabled / failed / output rejected              -> SOURCES_ONLY        (sources listed, no SOP-backed actions)
  sources support no action                            -> NOT_SUPPORTED
  >= 1 validated SOP-backed action                     -> GROUNDED
Every status except GROUNDED sets requires_review. Part 4 AI actions that no SOP supports are listed as
`unsupported_actions` - never presented as SOP-backed.
Logs: inspection id, defect, stage, retrieval count, top score, status, latency. Never document text or prompts.
"""
from __future__ import annotations

import json
import logging
import time

from pydantic import ValidationError

from backend.core.config import Settings
from backend.core.logging import log_pipeline
from backend.rag.embeddings import EmbeddingError, EmbeddingProvider, build_embedder
from backend.rag.guardrails import check_grounding
from backend.rag.prompt import PROMPT_VERSION, build_prompt
from backend.rag.retriever import RetrievalQuery, retrieve
from backend.rag.store import IndexUnavailable, VectorIndex
from backend.rca.guardrails import GroundingViolation
from backend.rca.providers import LLMError, LLMProvider
from backend.schemas.fulfillment import Classification, FulfillmentTraceAnalysis
from backend.schemas.rag import (NO_SOURCE_MESSAGE, NOT_SUPPORTED_MESSAGE, DefectGrounding, GroundedAction,
                                 LLMGroundingOutput, RAGResult, RAGStatus, RetrievedSource, UnsupportedAction)
from backend.schemas.rca import RCAGeneration, RootCauseAnalysis

logger = logging.getLogger(__name__)
ACTION_VOCABULARY = "corrective actions immediate actions required checks verification escalation"


class RAGService:
    def __init__(self, settings: Settings, provider: LLMProvider | None = None,
                 embedder: EmbeddingProvider | None = None, index: VectorIndex | None = None) -> None:
        self.settings = settings
        self.provider = provider if settings.llm_enabled else None
        self.index: VectorIndex | None = index
        self.embedder: EmbeddingProvider | None = embedder
        self.error: tuple[str, str] | None = None
        if not settings.rag_enabled:
            return
        try:
            self.index = index or VectorIndex.load(settings.rag_index_path)
            self.embedder = embedder or build_embedder(settings)
            m = self.index.manifest
            if (m.get("embedding_provider"), m.get("embedding_model"), m.get("dim")) != (
                    self.embedder.name, self.embedder.model, self.embedder.dim):
                raise IndexUnavailable("INDEX_EMBEDDING_MISMATCH", "index was built with a different embedding "
                                                                   "provider/model/dim; re-run the ingest")
            if not self.index.chunks:
                raise IndexUnavailable("EMPTY_KNOWLEDGE_BASE", "the index contains no chunks")
        except (IndexUnavailable, EmbeddingError) as exc:
            self.index, self.error = None, (exc.code, exc.message)
            logger.warning("RAG unavailable: %s", exc.code)

    # ------------------------------------------------------------------------------------------------
    def health(self) -> dict:
        s = self.settings
        base = {"enabled": s.rag_enabled, "embedding_provider": s.rag_embedding_provider,
                "embedding_model": s.rag_embedding_model, "vector_store": "local-numpy",
                "top_k": s.rag_top_k, "min_similarity": s.rag_min_similarity, "prompt_version": PROMPT_VERSION}
        if not s.rag_enabled:
            return {**base, "status": "disabled"}
        if self.index is None:
            return {**base, "status": "unavailable", "error": self.error[0] if self.error else "UNKNOWN"}
        m = self.index.manifest
        return {**base, "status": "ready", "documents": m["documents"], "chunks": m["chunks"], "dimensions": m["dim"],
                "index_built_at": m["built_at"], "provenance": m.get("provenance", [])}

    def _kb(self) -> dict:
        if self.index is None:
            return {}
        m = self.index.manifest
        return {k: m.get(k) for k in ("documents", "chunks", "embedding_provider", "embedding_model", "dim",
                                      "built_at", "provenance")}

    # ------------------------------------------------------------------------------------------------
    def ground(self, trace: FulfillmentTraceAnalysis, rca: RootCauseAnalysis) -> RAGResult:
        iid = trace.current_inspection_id
        if not trace.defect_analyses:
            return self._log(iid, RAGResult(status=RAGStatus.NOT_APPLICABLE, requires_review=False))
        if self.index is None or self.embedder is None:
            code, msg = self.error or ("RAG_DISABLED", "RAG is disabled (RAG_ENABLED=false)")
            return self._log(iid, RAGResult(status=RAGStatus.UNAVAILABLE, requires_review=True,
                                            message=f"{code}: {msg}"))

        t0 = time.perf_counter()
        rca_defects = {d.defect_type: d for d in rca.defects}
        defects, sources, retrieved = [], {}, {}
        try:
            for a in trace.defect_analyses:
                d = rca_defects.get(a.defect_type)
                window = tuple(dict.fromkeys([a.first_observed_stage.value]
                                             + ([] if a.classification is Classification.PRE_EXISTING
                                                else [s.value for s in a.missing_stages])))
                q = " ".join([f"stage {a.first_observed_stage.value.lower()}",
                              f"defect {a.defect_type.value.replace('_', ' ').lower()}",
                              f"severity {(d.facts.severity.value if d and d.facts.severity else '')}".strip(),
                              a.classification.value.replace("_", " ").lower(),
                              *(c.hypothesis for c in (d.likely_causes if d else [])),
                              *(c.category.value.replace("_", " ").lower() for c in (d.likely_causes if d else [])),
                              *(x.action for x in (d.corrective_actions if d else [])),
                              rca.summary or "", ACTION_VOCABULARY])
                hits = retrieve(self.index, self.embedder, RetrievalQuery(a.defect_type, window, q),
                                top_k=self.settings.rag_top_k, min_similarity=self.settings.rag_min_similarity)
                retrieved[a.defect_type] = [h.chunk.chunk_id for h in hits]
                for h in hits:
                    src = sources.get(h.chunk.chunk_id)
                    if src is None:
                        c = h.chunk
                        sources[c.chunk_id] = RetrievedSource(
                            chunk_id=c.chunk_id, document_id=c.document_id, title=c.title, section=c.section,
                            version=c.version, stage=c.stage, document_type=c.document_type, provenance=c.provenance,
                            text=c.text, similarity=h.similarity, rank_score=h.score, defect_types=[a.defect_type])
                    elif a.defect_type not in src.defect_types:
                        src.defect_types.append(a.defect_type)
                defects.append(DefectGrounding(
                    defect_type=a.defect_type, query=q[:1000], source_chunk_ids=retrieved[a.defect_type],
                    status=RAGStatus.SOURCES_ONLY if hits else RAGStatus.NO_RELEVANT_SOURCE,
                    message=None if hits else NO_SOURCE_MESSAGE))
                log_pipeline(iid, "RAG_RETRIEVAL", "RETRIEVED",
                             message=f"defect={a.defect_type.value} stage={a.first_observed_stage.value} "
                                     f"retrieved={len(hits)} top_similarity={hits[0].similarity if hits else None}")
        except EmbeddingError as exc:
            return self._log(iid, RAGResult(status=RAGStatus.UNAVAILABLE, requires_review=True,
                                            message=f"{exc.code}: {exc.message}", knowledge_base=self._kb()))
        except Exception:  # retrieval bug: never fail the inspection
            logger.exception("RAG retrieval failed")
            return self._log(iid, RAGResult(status=RAGStatus.UNAVAILABLE, requires_review=True,
                                            message="RETRIEVAL_ERROR: retrieval failed", knowledge_base=self._kb()))
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

        rca_actions = {t: [x.action for x in d.corrective_actions] for t, d in rca_defects.items()}
        base = dict(defects=defects, retrieved_sources=list(sources.values()), knowledge_base=self._kb(),
                    retrieval_ms=retrieval_ms)
        all_unsupported = [UnsupportedAction(defect_type=t, action=x) for t, acts in rca_actions.items() for x in acts]
        if not sources:
            return self._log(iid, RAGResult(status=RAGStatus.NO_RELEVANT_SOURCE, requires_review=True,
                                            message=NO_SOURCE_MESSAGE, unsupported_actions=all_unsupported, **base))
        if self.provider is None:
            return self._log(iid, RAGResult(
                status=RAGStatus.SOURCES_ONLY, requires_review=True, unsupported_actions=all_unsupported,
                message="Relevant SOP sections were found, but GenAI grounding is disabled (LLM_ENABLED=false); review "
                        "the sources manually.",
                generation=RCAGeneration(ai_generated=False, prompt_version=PROMPT_VERSION, error_code="LLM_DISABLED"),
                **base))

        context = [{"defect_type": a.defect_type.value, "first_observed_stage": a.first_observed_stage.value,
                    "last_known_good_stage": a.last_known_good_stage.value if a.last_known_good_stage else None,
                    "classification": a.classification.value, "evidence_level": a.evidence_level.value,
                    "missing_stages": [s.value for s in a.missing_stages],
                    "likely_causes": [c.hypothesis for c in (rca_defects[a.defect_type].likely_causes
                                                             if a.defect_type in rca_defects else [])],
                    "suggested_actions": [{"index": i, "action": x} for i, x in enumerate(rca_actions.get(a.defect_type, []))],
                    "source_chunk_ids": retrieved[a.defect_type]} for a in trace.defect_analyses]
        src_payload = [{"chunk_id": s.chunk_id, "document": s.title, "section": s.section, "version": s.version,
                        "stage": s.stage, "document_type": s.document_type, "text": s.text} for s in sources.values()]
        prompt = build_prompt(context, src_payload, sorted(sources), sorted(t.value for t in retrieved))
        t1 = time.perf_counter()
        try:
            completion = self.provider.generate_rca(prompt)
        except LLMError as exc:
            return self._sources_only(iid, base, all_unsupported, exc.code, exc.message, _ms(t1))
        except Exception:
            logger.exception("LLM provider raised during RAG grounding")
            return self._sources_only(iid, base, all_unsupported, "LLM_PROVIDER_ERROR", "Unexpected provider error", _ms(t1))
        gen = dict(provider=self.provider.name, model=completion.model, prompt_version=PROMPT_VERSION,
                   latency_ms=_ms(t1), input_tokens=completion.input_tokens, output_tokens=completion.output_tokens)
        try:
            out = LLMGroundingOutput.model_validate(json.loads(completion.text))
            check_grounding(out, {t: set(v) for t, v in retrieved.items()}, {k: s.text for k, s in sources.items()},
                            rca_actions)
        except (json.JSONDecodeError, ValidationError):
            return self._sources_only(iid, base, all_unsupported, "LLM_INVALID_OUTPUT", "grounding output failed "
                                      "schema validation", gen["latency_ms"], gen)
        except GroundingViolation as exc:
            return self._sources_only(iid, base, all_unsupported, f"LLM_UNGROUNDED_OUTPUT:{exc.code}", exc.detail,
                                      gen["latency_ms"], gen)

        grounded = []
        for g in out.grounded_actions:
            s = sources[g.chunk_id]
            grounded.append(GroundedAction(
                defect_type=g.defect_type, action=g.action, priority=g.priority, source_chunk_id=s.chunk_id,
                source_document_id=s.document_id, source_document=s.title, source_section=s.section,
                source_version=s.version, source_provenance=s.provenance, supporting_quote=g.quote,
                supports_rca_action=(rca_actions[g.defect_type][g.rca_action_index]
                                     if g.rca_action_index is not None else None)))
        supported = {(g.defect_type, g.supports_rca_action) for g in grounded}
        unsupported = [u for u in all_unsupported if (u.defect_type, u.action) not in supported]
        for d in defects:
            if d.status is RAGStatus.SOURCES_ONLY:
                has = any(g.defect_type is d.defect_type for g in grounded)
                d.status = RAGStatus.GROUNDED if has else RAGStatus.NOT_SUPPORTED
                d.message = None if has else NOT_SUPPORTED_MESSAGE
        status = RAGStatus.GROUNDED if grounded else RAGStatus.NOT_SUPPORTED
        review = status is not RAGStatus.GROUNDED or any(d.status is not RAGStatus.GROUNDED for d in defects)
        return self._log(iid, RAGResult(
            status=status, requires_review=review, grounded_actions=grounded, unsupported_actions=unsupported,
            message=None if grounded else NOT_SUPPORTED_MESSAGE, generation=RCAGeneration(ai_generated=True, **gen),
            **base))

    def _sources_only(self, iid, base, unsupported, code, message, latency, gen=None) -> RAGResult:
        g = gen or dict(provider=self.provider.name, model=self.provider.model, prompt_version=PROMPT_VERSION,
                        latency_ms=latency)
        return self._log(iid, RAGResult(
            status=RAGStatus.SOURCES_ONLY, requires_review=True, unsupported_actions=unsupported,
            message="Relevant SOP sections were found, but no SOP-backed actions could be generated; review the sources "
                    "manually.", generation=RCAGeneration(ai_generated=False, error_code=code,
                                                          error_message=message[:300], **g), **base))

    @staticmethod
    def _log(iid: str, r: RAGResult) -> RAGResult:
        g = r.generation
        log_pipeline(iid, "RAG_GROUNDING", r.status.value,
                     level=logging.WARNING if r.status is RAGStatus.UNAVAILABLE else logging.INFO,
                     duration_ms=(r.retrieval_ms or 0) + ((g.latency_ms or 0) if g else 0),
                     error_code=g.error_code if g else None,
                     message=f"sources={len(r.retrieved_sources)} grounded={len(r.grounded_actions)} "
                             f"unsupported={len(r.unsupported_actions)} retrieval_ms={r.retrieval_ms}")
        return r


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
