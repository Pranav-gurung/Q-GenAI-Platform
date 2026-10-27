"""Shared RAG test helpers: a session index built from the real knowledge/ corpus, and an RCA+RAG service."""
from __future__ import annotations

from pathlib import Path

from backend.core.config import Settings
from backend.rca.providers.mock_provider import MockProvider
from backend.rca.service import RCAService

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = ROOT / "knowledge"

DOC = """---
document_id: {did}
title: {title}
stage: {stage}
document_type: SOP
version: demo-1.0
defect_types: [{defect}]
provenance: DEMO_SYNTHETIC
---
# {title}

## Purpose
Demo document for tests.

## Corrective Actions
{body}
"""


def write_doc(root: Path, name: str, did="TEST-SOP", title="Test SOP", stage="PACKING", defect="LEAKING_PACKAGE",
              body="- Inspect the heat sealer before restarting the line.") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DOC.format(did=did, title=title, stage=stage, defect=defect, body=body), encoding="utf-8")
    return path


def rag_settings(index: Path, **kw) -> Settings:
    base = dict(_env_file=None, llm_enabled=True, llm_provider="mock", rag_enabled=True, rag_index_path=index,
                rag_knowledge_path=KNOWLEDGE)
    return Settings(**{**base, **kw})


def rag_service(index: Path, mode="ok", rag_mode="ok", **kw):
    mock = MockProvider(mode, rag_mode=rag_mode)
    s = rag_settings(index, **kw)
    return RCAService(s, provider=mock if s.llm_enabled else None), mock
