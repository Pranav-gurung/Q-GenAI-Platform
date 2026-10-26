"""Settings loaded from environment variables only (never hardcode secrets)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Placeholder defaults so the service can start; they MUST be aligned with Module A's class map
# (set RCA_CLASS_MAP_PATH to a module_a canonical file or JSON class map, or RCA_SUPPORTED_CLASSES).
DEFAULT_CLASSES = ("scratch", "crack", "dent", "corrosion")


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _load_classes() -> tuple:
    path = os.environ.get("RCA_CLASS_MAP_PATH")
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data = data.get("class_map", data) if isinstance(data, dict) else data
        return tuple(sorted(data) if isinstance(data, dict) else data)
    env = os.environ.get("RCA_SUPPORTED_CLASSES")
    if env:
        return tuple(c.strip().lower() for c in env.split(",") if c.strip())
    return DEFAULT_CLASSES


@dataclass
class Settings:
    approved_docs_dir: Path = Path("documents/approved")
    index_dir: Path = Path("data/rag_index")
    embedding_backend: str = "hashing"  # hashing | sentence_transformers
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "json"  # json | chroma
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    min_score: Optional[float] = None  # None -> backend default (calibrate on your corpus!)
    min_evidence_passages: int = 1
    allowed_access_levels: tuple = ("public", "internal")
    min_page_chars: int = 40  # pages with fewer characters are treated as having no usable text
    llm_provider: str = "none"  # none | anthropic | openai_compatible
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_output_retries: int = 1
    llm_max_tokens: int = 1500
    max_defects: int = 20
    supported_classes: tuple = DEFAULT_CLASSES
    audit_sink: str = "jsonl"  # jsonl | none
    audit_path: Path = Path("data/audit/rca_audit.jsonl")
    audit_fail_mode: str = "fail_closed"  # fail_closed | fail_open
    auth_mode: str = "none"  # none | static_api_key  (placeholder until Module D provides real auth)
    require_class_term: bool = True  # passage must mention the defect class (or an alias) to count as evidence
    class_aliases: dict = field(default_factory=dict)  # {"scratch": ["scratches", "abrasion"]}
    log_passages: bool = False

    @property
    def effective_min_score(self) -> float:
        if self.min_score is not None:
            return self.min_score
        return 0.30 if self.embedding_backend == "sentence_transformers" else 0.08

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        return cls(
            approved_docs_dir=Path(e("RCA_APPROVED_DOCS_DIR", "documents/approved")),
            index_dir=Path(e("RCA_INDEX_DIR", "data/rag_index")),
            embedding_backend=e("RCA_EMBEDDING_BACKEND", "hashing"),
            embedding_model=e("RCA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            vector_store=e("RCA_VECTOR_STORE", "json"),
            chunk_size=_i("RCA_CHUNK_SIZE", 900), chunk_overlap=_i("RCA_CHUNK_OVERLAP", 150),
            top_k=_i("RCA_TOP_K", 5),
            min_score=float(e("RCA_MIN_SCORE")) if e("RCA_MIN_SCORE") else None,
            min_evidence_passages=_i("RCA_MIN_EVIDENCE_PASSAGES", 1),
            allowed_access_levels=tuple(x.strip() for x in e("RCA_ALLOWED_ACCESS_LEVELS", "public,internal").split(",") if x.strip()),
            min_page_chars=_i("RCA_MIN_PAGE_CHARS", 40),
            llm_provider=e("RCA_LLM_PROVIDER", "none"), llm_model=e("RCA_LLM_MODEL"),
            llm_base_url=e("RCA_LLM_BASE_URL"), llm_timeout_s=_f("RCA_LLM_TIMEOUT_S", 30.0),
            llm_max_retries=_i("RCA_LLM_MAX_RETRIES", 2), llm_output_retries=_i("RCA_LLM_OUTPUT_RETRIES", 1),
            llm_max_tokens=_i("RCA_LLM_MAX_TOKENS", 1500),
            max_defects=_i("RCA_MAX_DEFECTS", 20), supported_classes=_load_classes(),
            audit_sink=e("RCA_AUDIT_SINK", "jsonl"), audit_path=Path(e("RCA_AUDIT_PATH", "data/audit/rca_audit.jsonl")),
            audit_fail_mode=e("RCA_AUDIT_FAIL_MODE", "fail_closed"),
            auth_mode=e("RCA_AUTH_MODE", "none"),
            require_class_term=e("RCA_REQUIRE_CLASS_TERM", "true").lower() == "true",
            class_aliases={k.lower(): [a.lower() for a in v] for k, v in json.loads(e("RCA_CLASS_ALIASES", "{}")).items()},
            log_passages=e("RCA_LOG_PASSAGES", "false").lower() == "true",
        )
    # Secrets (ANTHROPIC_API_KEY, OPENAI_API_KEY, RCA_API_KEY) are read from the environment at the
    # point of use and are never stored on Settings, so they cannot leak through repr()/logging.
