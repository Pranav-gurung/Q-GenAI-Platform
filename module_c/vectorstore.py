"""Embedders and vector stores behind small interfaces so they can be swapped without touching RCA logic.

Embedders:  HashingEmbedder (dependency-free LEXICAL baseline for tests/offline use),
            SentenceTransformerEmbedder (semantic; needs sentence-transformers + model weights).
Stores:     JsonVectorStore (numpy, file-persisted, exact cosine search),
            ChromaVectorStore (needs chromadb).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .documents import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were will with".split())


class Embedder(Protocol):
    name: str

    def embed(self, texts: list) -> np.ndarray: ...


def _stem(t: str) -> str:
    """Very light suffix stripping so 'scratches'/'scratched'/'scratch' share a feature (baseline only)."""
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith(("ches", "shes", "sses", "xes")):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


class HashingEmbedder:
    """Signed hashed bag of (uni+bi)grams, L2-normalised. Lexical overlap only - NOT semantic."""

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.name = f"hashing-{dim}-v2"

    def _vec(self, text: str) -> np.ndarray:
        toks = [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += (1.0 if (h >> 63) & 1 else -1.0)
        # sublinear scaling dampens repeated terms
        v = np.sign(v) * np.log1p(np.abs(v))
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("RCA_EMBEDDING_BACKEND=sentence_transformers needs: pip install sentence-transformers") from e
        self.name = f"st:{model_name}"
        self._m = SentenceTransformer(model_name)

    def embed(self, texts: list) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)


def make_embedder(backend: str, model: str) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model)
    raise ValueError(f"unknown embedding backend {backend!r}")


class VectorStore(Protocol):
    def upsert(self, chunks: list, vectors: np.ndarray) -> None: ...
    def delete_document(self, document_id: str) -> int: ...
    def set_active(self, document_id: str, active: bool) -> None: ...
    def query(self, vector: np.ndarray, top_k: int, allowed_access: Optional[set]) -> list: ...  # [(Chunk, score)]
    def count(self, active_only: bool = True) -> int: ...


class JsonVectorStore:
    """Exact cosine search over a persisted numpy matrix. Suitable for thousands of chunks."""

    def __init__(self, directory: Path, embedder_name: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder_name = embedder_name
        self.chunks: dict = {}   # chunk_id -> Chunk
        self.active: dict = {}   # document_id -> bool
        self.vecs: dict = {}     # chunk_id -> np.ndarray
        self._load()

    def _paths(self):
        return self.dir / "chunks.json", self.dir / "vectors.npz"

    def _load(self):
        cj, vz = self._paths()
        if not cj.exists():
            return
        doc = json.loads(cj.read_text(encoding="utf-8"))
        if doc.get("embedder") != self.embedder_name:
            raise RuntimeError(f"index was built with embedder {doc.get('embedder')!r} but {self.embedder_name!r} is configured; "
                               "re-ingest with --rebuild")
        self.chunks = {c["chunk_id"]: Chunk(**c) for c in doc["chunks"]}
        self.active = doc["active"]
        data = np.load(vz)
        self.vecs = {cid: data["v"][i] for i, cid in enumerate(data["ids"].tolist())}

    def _save(self):
        cj, vz = self._paths()
        ids = sorted(self.chunks)
        tmp_json = cj.with_suffix(".tmp")
        tmp_json.write_text(json.dumps({"embedder": self.embedder_name, "active": self.active,
                                        "chunks": [asdict(self.chunks[i]) for i in ids]}), encoding="utf-8")
        tmp_npz = self.dir / "vectors.tmp.npz"
        np.savez(tmp_npz, ids=np.array(ids), v=np.stack([self.vecs[i] for i in ids]) if ids else np.zeros((0, 1), np.float32))
        os.replace(tmp_npz, vz)
        os.replace(tmp_json, cj)

    def upsert(self, chunks, vectors):
        for c, v in zip(chunks, vectors):
            self.chunks[c.chunk_id] = c  # same id => replaced, never duplicated
            self.vecs[c.chunk_id] = np.asarray(v, dtype=np.float32)
            self.active.setdefault(c.document_id, True)
        self._save()

    def delete_document(self, document_id):
        ids = [i for i, c in self.chunks.items() if c.document_id == document_id]
        for i in ids:
            del self.chunks[i], self.vecs[i]
        self.active.pop(document_id, None)
        self._save()
        return len(ids)

    def set_active(self, document_id, active):
        if document_id in self.active:
            self.active[document_id] = active
            self._save()

    def query(self, vector, top_k, allowed_access=None):
        ids = [i for i, c in self.chunks.items() if self.active.get(c.document_id, False)
               and (allowed_access is None or c.access_level in allowed_access)]
        if not ids:
            return []
        mat = np.stack([self.vecs[i] for i in ids])
        scores = mat @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.chunks[ids[j]], float(scores[j])) for j in order]

    def count(self, active_only=True):
        return sum(1 for c in self.chunks.values() if not active_only or self.active.get(c.document_id, False))


class ChromaVectorStore:
    """ChromaDB adapter (cosine). NOTE: written to the documented chromadb API but not exercised in this repo's CI sandbox."""

    def __init__(self, directory: Path, embedder_name: str):
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError("RCA_VECTOR_STORE=chroma needs: pip install chromadb") from e
        self._client = chromadb.PersistentClient(path=str(directory))
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", embedder_name)[:50]
        self._col = self._client.get_or_create_collection(f"rca_{safe}", metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _meta(c: Chunk, active: bool = True) -> dict:
        d = asdict(c)
        d.pop("text")
        d["active"] = 1 if active else 0
        return {k: ("" if v is None else v) for k, v in d.items()}

    def upsert(self, chunks, vectors):
        self._col.upsert(ids=[c.chunk_id for c in chunks], embeddings=[list(map(float, v)) for v in vectors],
                         documents=[c.text for c in chunks], metadatas=[self._meta(c) for c in chunks])

    def delete_document(self, document_id):
        got = self._col.get(where={"document_id": document_id})
        if got["ids"]:
            self._col.delete(ids=got["ids"])
        return len(got["ids"])

    def set_active(self, document_id, active):
        got = self._col.get(where={"document_id": document_id})
        if got["ids"]:
            self._col.update(ids=got["ids"], metadatas=[{**m, "active": 1 if active else 0} for m in got["metadatas"]])

    def query(self, vector, top_k, allowed_access=None):
        where = {"active": 1}
        if allowed_access is not None:
            where = {"$and": [{"active": 1}, {"access_level": {"$in": sorted(allowed_access)}}]}
        cnt = self._col.count()
        if cnt == 0:
            return []
        n_results = min(top_k, cnt)
        r = self._col.query(query_embeddings=[list(map(float, vector))], n_results=n_results, where=where)
        out = []
        for text, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
            m = {k: meta.get(k) for k in Chunk.__dataclass_fields__ if k != "text"}
            m["section"] = m["section"] or None
            out.append((Chunk(text=text, **m), 1.0 - float(dist)))  # cosine distance -> similarity
        return out

    def count(self, active_only=True):
        return len(self._col.get(where={"active": 1})["ids"]) if active_only else self._col.count()


def make_store(kind: str, directory: Path, embedder_name: str) -> VectorStore:
    if kind == "json":
        return JsonVectorStore(directory, embedder_name)
    if kind == "chroma":
        return ChromaVectorStore(directory, embedder_name)
    raise ValueError(f"unknown vector store {kind!r}")
