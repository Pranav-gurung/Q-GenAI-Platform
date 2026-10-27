"""Build the RAG vector index from the knowledge base (synchronous, offline, idempotent).

    python -m backend.rag.ingest [--knowledge knowledge] [--index data/rag/index] [--validate-only]

load -> validate metadata -> section chunks -> embed -> write index (atomic). Exit 1 on any document error.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.core.config import Settings, get_settings
from backend.rag.chunking import chunk_document
from backend.rag.documents import Corpus, DocumentError, IngestIssue, load_corpus
from backend.rag.embeddings import build_embedder
from backend.rag.store import INDEX_FORMAT, VectorIndex


def build_index(settings: Settings, knowledge: Path | None = None) -> tuple[VectorIndex | None, Corpus, dict]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    corpus = load_corpus(knowledge or settings.rag_knowledge_path)
    chunks = []
    for doc in corpus.documents:
        try:
            chunks += chunk_document(doc, settings.rag_max_chunk_chars)
        except DocumentError as exc:
            corpus.issues.append(IngestIssue("error", exc.code, doc.path, exc.message))
    timings["load_and_chunk_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if corpus.errors:
        return None, corpus, timings
    embedder = build_embedder(settings)
    t1 = time.perf_counter()
    vectors = embedder.embed([c.embedding_text() for c in chunks])
    timings["embedding_ms"] = round((time.perf_counter() - t1) * 1000, 2)
    manifest = {
        "format": INDEX_FORMAT, "embedding_provider": embedder.name, "embedding_model": embedder.model,
        "dim": int(vectors.shape[1]) if len(chunks) else embedder.dim, "documents": len(corpus.documents),
        "chunks": len(chunks), "corpus_sha256": corpus.sha256,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document_versions": {d.metadata.document_id: d.metadata.version for d in corpus.documents},
        "provenance": sorted({d.metadata.provenance for d in corpus.documents}),
    }
    return VectorIndex(chunks=chunks, vectors=np.asarray(vectors, np.float32), manifest=manifest), corpus, timings


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--knowledge", type=Path, default=s.rag_knowledge_path)
    p.add_argument("--index", type=Path, default=s.rag_index_path)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args(argv)
    t0 = time.perf_counter()
    index, corpus, timings = build_index(s, args.knowledge)
    for i in corpus.issues:
        print(f"  [{i.level.upper():7}] {i.code}: {i.path}: {i.message}")
    if index is None:
        print(f"INVALID knowledge base ({len(corpus.errors)} error(s)); index not written")
        return 1
    print(f"documents={index.manifest['documents']} chunks={index.manifest['chunks']} "
          f"embedding={index.manifest['embedding_provider']}/{index.manifest['embedding_model']} dim={index.manifest['dim']}")
    if not args.validate_only:
        t1 = time.perf_counter()
        index.save(args.index)
        timings["write_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        print(f"index written to {args.index}")
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    print("timings:", timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
