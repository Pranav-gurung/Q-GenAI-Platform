"""Section-aware, deterministic chunking.

One chunk per '## ' section (the front-matter banner before the first section is not chunked). A section longer
than RAG_MAX_CHUNK_CHARS is split on line boundaries into parts that keep the section heading. Every chunk keeps
the full document metadata, so a retrieved chunk is understandable and citable on its own.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from backend.domain.enums import DefectType
from backend.rag.documents import DocumentError, KnowledgeDocument


class Chunk(BaseModel):
    chunk_id: str            # "<DOCUMENT_ID>#<section-slug>[-<part>]" - stable citation id
    document_id: str
    title: str
    stage: str               # FulfillmentStage value or GENERAL
    document_type: str
    version: str
    provenance: str
    section: str
    part: int = 1
    text: str                # section content only (quotable verbatim)
    defect_types: list[DefectType]

    def embedding_text(self) -> str:
        defects = " ".join(d.value.replace("_", " ").lower() for d in self.defect_types)
        return f"{self.title}. {self.section}. stage {self.stage.lower()}. {defects}. {self.text}"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _split(lines: list[str], max_chars: int) -> list[list[str]]:
    parts, cur, size = [], [], 0
    for line in lines:
        if cur and size + len(line) + 1 > max_chars:
            parts.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        parts.append(cur)
    return parts


def chunk_document(doc: KnowledgeDocument, max_chars: int = 1200) -> list[Chunk]:
    meta = doc.metadata
    sections: list[tuple[str, list[str]]] = []
    for line in doc.body.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line.rstrip())
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for title, lines in sections:
        sid = slug(title)
        if not sid or sid in seen:
            raise DocumentError("DUPLICATE_SECTION", f"{meta.document_id}: section '{title}' is empty or repeated")
        seen.add(sid)
        content = [ln for ln in lines if ln.strip()]
        if not content:
            continue
        parts = _split(content, max_chars)
        for i, part in enumerate(parts, 1):
            chunks.append(Chunk(
                chunk_id=f"{meta.document_id}#{sid}" + (f"-{i}" if len(parts) > 1 else ""), document_id=meta.document_id,
                title=meta.title, stage=meta.stage, document_type=meta.document_type, version=meta.version,
                provenance=meta.provenance, section=title, part=i, text="\n".join(part), defect_types=meta.defect_types))
    return chunks
