"""Ingestion service (idempotent, re-ingestable) and retriever with relevance thresholding."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Settings
from .documents import (SUPPORTED_SUFFIXES, build_chunks, extract, load_meta, sha256_bytes, usable_pages)
from .models import Passage
from .vectorstore import Embedder, VectorStore, make_embedder, make_store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentRegistry:
    """Tracks every document seen: version, hash, status and reasons (so nothing is silently dropped)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.docs: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.docs, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class IngestResult:
    filename: str
    status: str  # indexed | unchanged | reindexed | ocr_required | empty | error | deactivated_missing
    document_id: Optional[str] = None
    chunks: int = 0
    detail: str = ""
    warnings: list = field(default_factory=list)


class IngestionService:
    def __init__(self, settings: Settings, embedder: Optional[Embedder] = None, store: Optional[VectorStore] = None):
        self.s = settings
        self.embedder = embedder or make_embedder(settings.embedding_backend, settings.embedding_model)
        self.store = store or make_store(settings.vector_store, settings.index_dir, self.embedder.name)
        self.registry = DocumentRegistry(Path(settings.index_dir) / "registry.json")

    def ingest_directory(self, directory: Optional[Path] = None, deactivate_missing: bool = True) -> list:
        directory = Path(directory or self.s.approved_docs_dir)
        results = []
        if not directory.is_dir():
            return [IngestResult(str(directory), "error", detail="approved documents directory does not exist")]
        present = set()
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and p.name.lower() != "readme.md":
                present.add(p.name)
                results.append(self.ingest_file(p))
        if deactivate_missing:
            for doc_id, d in self.registry.docs.items():
                if d["source_filename"] not in present and d["status"] == "active":
                    self.deactivate(doc_id, reason="source file no longer in approved directory")
                    results.append(IngestResult(d["source_filename"], "deactivated_missing", doc_id,
                                                detail="source file no longer in approved directory"))
        return results

    def ingest_file(self, path: Path) -> IngestResult:
        path = Path(path)
        try:
            raw = path.read_bytes()
            chash = sha256_bytes(raw)
            ext = extract(path)
            meta = load_meta(path, chash, ext.title_hint)
        except Exception as e:
            return IngestResult(path.name, "error", detail=f"{type(e).__name__}: {e}")
        warnings = []
        if meta.metadata_source == "inferred":
            warnings.append("no .meta.json sidecar: document_id/title inferred from filename, version='unspecified'")
        prev = self.registry.docs.get(meta.document_id)
        if prev and prev["content_hash"] == chash and prev["status"] == "active" and prev["version"] == meta.version \
                and prev.get("access_level") == meta.access_level:
            return IngestResult(path.name, "unchanged", meta.document_id, prev["n_chunks"], warnings=warnings)

        good, empty_pages = usable_pages(ext, self.s.min_page_chars)
        base = {"source_filename": path.name, "title": meta.title, "version": meta.version, "content_hash": chash,
                "access_level": meta.access_level, "license_note": meta.license_note, "n_pages": len(ext.pages),
                "extractor": ext.backend, "updated_at": _now(), "metadata_source": meta.metadata_source}
        if not good:
            if prev:  # a previously indexed revision is no longer valid for the new bytes
                self.store.delete_document(meta.document_id)
            status = "ocr_required" if ext.pages else "empty"
            self.registry.docs[meta.document_id] = {**base, "status": status, "n_chunks": 0,
                                                    "empty_pages": empty_pages, "chunk_ids": []}
            self.registry.save()
            msg = ("no extractable text (likely a scanned/image-only PDF): OCR is required before this document can be indexed"
                   if status == "ocr_required" else "document has no pages/text")
            return IngestResult(path.name, status, meta.document_id, 0, msg, warnings)
        if empty_pages:
            warnings.append(f"pages with no usable text (possibly scanned; OCR needed): {empty_pages}")
        chunks = build_chunks(meta, good, self.s.chunk_size, self.s.chunk_overlap)
        if prev:  # content/version changed: remove the stale revision first so old chunks cannot be cited
            self.store.delete_document(meta.document_id)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.upsert(chunks, vectors)
        self.registry.docs[meta.document_id] = {**base, "status": "active", "n_chunks": len(chunks),
                                                "empty_pages": empty_pages, "chunk_ids": [c.chunk_id for c in chunks]}
        self.registry.save()
        return IngestResult(path.name, "reindexed" if prev else "indexed", meta.document_id, len(chunks), warnings=warnings)

    def deactivate(self, document_id: str, reason: str = "") -> None:
        """Keep chunks but exclude them from retrieval (e.g. document no longer approved)."""
        if document_id not in self.registry.docs:
            raise KeyError(document_id)
        self.store.set_active(document_id, False)
        self.registry.docs[document_id].update(status="inactive", updated_at=_now(), inactive_reason=reason)
        self.registry.save()

    def reactivate(self, document_id: str) -> None:
        self.store.set_active(document_id, True)
        self.registry.docs[document_id].update(status="active", updated_at=_now())
        self.registry.save()

    def remove(self, document_id: str) -> int:
        n = self.store.delete_document(document_id)
        self.registry.docs.pop(document_id, None)
        self.registry.save()
        return n


class Retriever:
    def __init__(self, settings: Settings, embedder: Optional[Embedder] = None, store: Optional[VectorStore] = None):
        self.s = settings
        self.embedder = embedder or make_embedder(settings.embedding_backend, settings.embedding_model)
        self.store = store or make_store(settings.vector_store, settings.index_dir, self.embedder.name)

    def search(self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None) -> tuple:
        """Return (relevant_passages, n_candidates). Only passages scoring >= threshold are returned."""
        k = top_k or self.s.top_k
        thr = self.s.effective_min_score if min_score is None else min_score
        vec = self.embedder.embed([query])[0]
        hits = self.store.query(vec, k, set(self.s.allowed_access_levels))
        out = [Passage(c.chunk_id, c.document_id, c.title, c.version, c.page, c.section, round(s, 4), c.text)
               for c, s in hits if s >= thr]
        return out, len(hits)

    def has_evidence(self, passages: list) -> bool:
        return len(passages) >= self.s.min_evidence_passages


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ingest approved documents into the RAG index")
    ap.add_argument("--dir", help="approved documents directory (default: RCA_APPROVED_DOCS_DIR)")
    ap.add_argument("--deactivate", metavar="DOC_ID")
    ap.add_argument("--remove", metavar="DOC_ID")
    ap.add_argument("--query", help="run a test retrieval after ingestion")
    a = ap.parse_args(argv)
    svc = IngestionService(Settings.from_env())
    if a.deactivate:
        svc.deactivate(a.deactivate, "manual")
        print(f"deactivated {a.deactivate}")
        return 0
    if a.remove:
        print(f"removed {svc.remove(a.remove)} chunks")
        return 0
    rc = 0
    for r in svc.ingest_directory(Path(a.dir) if a.dir else None):
        print(f"[{r.status}] {r.filename} id={r.document_id} chunks={r.chunks} {r.detail}")
        for w in r.warnings:
            print(f"    warning: {w}")
        rc |= r.status in ("error", "ocr_required", "empty")
    if a.query:
        for p in Retriever(Settings.from_env(), svc.embedder, svc.store).search(a.query)[0]:
            print(f"{p.score:.3f} {p.document_id} v{p.version} p{p.page} {p.chunk_id}")
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
