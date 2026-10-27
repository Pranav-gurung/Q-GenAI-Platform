"""Local vector index: a directory with vectors.npy (float32, L2-normalised), chunks.json and manifest.json.

Brute-force cosine search (one matrix-vector product): exact and fast for a corpus of hundreds of chunks. The
manifest records the embedding provider/model/dim and corpus hash so a mismatched or stale index is detected.
Written atomically (temp dir + rename).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.rag.chunking import Chunk

INDEX_FORMAT = 1


class IndexUnavailable(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class VectorIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    manifest: dict

    def search(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of the (normalised) query against every chunk."""
        return self.vectors @ query.astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".index-", dir=path.parent))
        try:
            np.save(tmp / "vectors.npy", self.vectors.astype(np.float32))
            (tmp / "chunks.json").write_text(json.dumps([c.model_dump(mode="json") for c in self.chunks], indent=1))
            (tmp / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
            if path.exists():
                shutil.rmtree(path)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        path = Path(path)
        if not (path / "manifest.json").is_file():
            raise IndexUnavailable("INDEX_MISSING", f"no vector index at {path} (run: python -m backend.rag.ingest)")
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            chunks = [Chunk.model_validate(c) for c in json.loads((path / "chunks.json").read_text())]
            vectors = np.load(path / "vectors.npy")
        except Exception as exc:
            raise IndexUnavailable("INDEX_CORRUPT", f"vector index at {path} cannot be read") from exc
        if manifest.get("format") != INDEX_FORMAT or vectors.shape != (len(chunks), manifest.get("dim")):
            raise IndexUnavailable("INDEX_CORRUPT", "vector index shape or format does not match its manifest")
        return cls(chunks=chunks, vectors=vectors, manifest=manifest)
