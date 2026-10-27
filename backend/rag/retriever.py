"""Retrieval for one defect: query text from the deterministic facts + RCA hypotheses, cosine search, relevance gate,
then metadata-aware ranking.

  similarity = cosine(query, chunk)                                   (embedding space)
  relevant   = similarity >= RAG_MIN_SIMILARITY                       (hard gate on TEXT similarity only)
  score      = similarity + 0.10 if chunk.stage is a candidate stage of the defect (Part 3 window)
                          + 0.05 if chunk.stage is GENERAL (cross-stage SOPs: damaged product, leakage, materials)
                          + 0.10 if the defect type is in the chunk's defect_types
  at most MAX_PER_DOCUMENT chunks per document (so evidence spans documents), then the top RAG_TOP_K by
  (score desc, chunk_id) - deterministic.
Metadata only re-ranks chunks that are already textually relevant: it can never pull in an unrelated chunk, and a
relevant chunk from another stage is still eligible (soft filter, not a hard one).
"Purpose" and "Applicable Stage" sections are orientation, not procedures: they are indexed but never retrieved as
grounding for actions (CONTEXT_SECTIONS).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.domain.enums import DefectType
from backend.rag.chunking import Chunk
from backend.rag.documents import GENERAL
from backend.rag.embeddings import EmbeddingProvider
from backend.rag.store import VectorIndex

STAGE_BOOST, GENERAL_BOOST, DEFECT_BOOST = 0.10, 0.05, 0.10
MAX_PER_DOCUMENT = 2
CONTEXT_SECTIONS = frozenset({"purpose", "applicable stage"})


@dataclass(frozen=True)
class RetrievalQuery:
    defect_type: DefectType
    stages: tuple[str, ...]   # candidate stages (Part 3 window), most relevant first
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    similarity: float
    score: float


def retrieve(index: VectorIndex, embedder: EmbeddingProvider, query: RetrievalQuery, *, top_k: int,
             min_similarity: float) -> list[Hit]:
    qvec = embedder.embed([query.text])[0]
    sims = index.search(qvec)
    hits = []
    for chunk, sim in zip(index.chunks, np.asarray(sims, dtype=float)):
        if sim < min_similarity or chunk.section.lower() in CONTEXT_SECTIONS:
            continue
        score = sim + (STAGE_BOOST if chunk.stage in query.stages else 0.0) \
            + (GENERAL_BOOST if chunk.stage == GENERAL else 0.0) \
            + (DEFECT_BOOST if query.defect_type in chunk.defect_types else 0.0)
        hits.append(Hit(chunk, round(float(sim), 4), round(float(score), 4)))
    hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
    per_doc: dict[str, int] = {}
    kept = []
    for h in hits:
        if per_doc.get(h.chunk.document_id, 0) < MAX_PER_DOCUMENT:
            per_doc[h.chunk.document_id] = per_doc.get(h.chunk.document_id, 0) + 1
            kept.append(h)
    return kept[:top_k]
