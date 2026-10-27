"""Embedding providers (RAG_EMBEDDING_PROVIDER).

hashing (default)        Local, deterministic, dependency-free LEXICAL embedding: normalised word stems (with a
                         small domain synonym map, e.g. leakage/leaking/seepage -> leak) plus word bigrams,
                         feature-hashed (BLAKE2b, signed) into RAG_EMBEDDING_DIM dims, sublinear tf, L2-normalised.
                         Captures shared vocabulary, not meaning: adequate for a small, consistently worded SOP corpus.
sentence_transformers    Semantic embeddings from a sentence-transformers model loaded from RAG_EMBEDDING_MODEL
                         (a local directory or a model id the machine can download). Optional dependency.
Vectors from different providers / models / dims are not comparable: the index manifest records which one built it
and the retriever refuses a mismatch.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

from backend.core.config import Settings

STOPWORDS = frozenset("""a an and are as at be before by for from has have if in into is it its may must no not of on
or that the their them then there these this to under was were when where which while with within without any all
each every other same than do does every one more only also so should can""".split())
SYNONYMS = {"leakage": "leak", "leaking": "leak", "leaks": "leak", "leaked": "leak", "seepage": "leak", "wet": "leak",
            "sealing": "seal", "sealed": "seal", "seals": "seal", "sealer": "seal", "unsealed": "seal",
            "packing": "pack", "packed": "pack", "package": "pack", "packages": "pack", "packaging": "pack",
            "crushed": "crush", "crushing": "crush", "torn": "tear", "tears": "tear", "tearing": "tear",
            "dented": "dent", "dents": "dent", "damaged": "damage", "damages": "damage",
            "labels": "label", "labelling": "label", "labeling": "label", "labelled": "label",
            "receiving": "receive", "received": "receive", "receipt": "receive", "inbound": "receive",
            "dispatch": "dispatch", "handover": "dispatch", "picked": "pick", "picking": "pick", "picker": "pick",
            "stored": "storage", "storing": "storage", "delivered": "deliver", "delivery": "deliver",
            "inspected": "inspect", "inspection": "inspect", "inspecting": "inspect", "reinspect": "inspect",
            "verify": "verify", "verified": "verify", "verification": "verify"}
TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalised rows. Raises EmbeddingError."""
        ...


def normalise(word: str) -> str:
    if word in SYNONYMS:
        return SYNONYMS[word]
    for suffix, min_len in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if word.endswith(suffix) and len(word) >= min_len:
            word = word[: -len(suffix)]
            break
    return SYNONYMS.get(word, word)


def tokens(text: str) -> list[str]:
    return [normalise(w) for w in TOKEN.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


class HashingEmbeddingProvider:
    name = "hashing"

    def __init__(self, dim: int = 2048, model: str = "hashing-lexical-v1") -> None:
        self.dim, self.model = dim, model

    def _vector(self, text: str) -> np.ndarray:
        toks = tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        counts: dict[str, int] = {}
        for f in feats:
            counts[f] = counts.get(f, 0) + 1
        v = np.zeros(self.dim, dtype=np.float32)
        for f, c in counts.items():
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + math.log(c))
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEmbeddingProvider:
    name = "sentence_transformers"

    def __init__(self, model: str) -> None:
        self.model = model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("EMBEDDING_DEPENDENCIES_MISSING", "pip install sentence-transformers") from exc
        try:
            self._model = SentenceTransformer(model, device="cpu")
        except Exception as exc:  # missing local files / no network to download
            raise EmbeddingError("EMBEDDING_MODEL_UNAVAILABLE", f"cannot load embedding model {model!r}") from exc
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)
        except Exception as exc:
            raise EmbeddingError("EMBEDDING_FAILED", "embedding model failed") from exc


def build_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.rag_embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(settings.rag_embedding_model)
    return HashingEmbeddingProvider(dim=settings.rag_embedding_dim, model=settings.rag_embedding_model)
