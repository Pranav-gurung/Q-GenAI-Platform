"""Knowledge-base documents: Markdown with YAML front matter.

    ---
    document_id: PACKING-SOP          # stable id used in citations (A-Z, 0-9, '-')
    title: Packing and Sealing SOP
    stage: PACKING                    # a FulfillmentStage, or GENERAL (applies to all stages)
    document_type: SOP                # SOP | CHECKLIST
    version: demo-1.0
    defect_types: [LEAKING_PACKAGE]   # platform DefectType values the document addresses
    provenance: DEMO_SYNTHETIC        # DEMO_SYNTHETIC | APPROVED (APPROVED = reviewed by the operator's QA team)
    ---
    ## Section heading
    - section content ...
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

from backend.domain.enums import DefectType
from backend.domain.fulfillment import FulfillmentStage

GENERAL = "GENERAL"
STAGE_SCOPES = tuple(s.value for s in FulfillmentStage) + (GENERAL,)
SUSPICIOUS = re.compile(r"ignore (all |any )?(previous|prior|above) instructions|disregard (the )?(system|previous)|"
                        r"you are now|system prompt|<\s*/?\s*(system|evidence|sources)\s*>", re.I)


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: Annotated[str, StringConstraints(pattern=r"^[A-Z0-9][A-Z0-9-]{1,63}$")]
    title: Annotated[str, StringConstraints(min_length=3, max_length=120)]
    stage: str
    document_type: Literal["SOP", "CHECKLIST"]
    version: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")]
    defect_types: list[DefectType] = Field(min_length=1)
    provenance: Literal["DEMO_SYNTHETIC", "APPROVED"]

    @field_validator("stage")
    @classmethod
    def _stage(cls, v: str) -> str:
        if v not in STAGE_SCOPES:
            raise ValueError(f"stage must be one of {STAGE_SCOPES}")
        return v


@dataclass(frozen=True)
class KnowledgeDocument:
    metadata: DocumentMetadata
    body: str
    path: str
    content_sha256: str


@dataclass
class IngestIssue:
    level: str  # error | warning
    code: str
    path: str
    message: str


@dataclass
class Corpus:
    documents: list[KnowledgeDocument] = field(default_factory=list)
    issues: list[IngestIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[IngestIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def sha256(self) -> str:
        h = hashlib.sha256()
        for d in sorted(self.documents, key=lambda d: d.metadata.document_id):
            h.update(d.metadata.model_dump_json().encode())
            h.update(d.content_sha256.encode())
        return h.hexdigest()


class DocumentError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def parse_document(path: Path) -> KnowledgeDocument:
    raw = Path(path).read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
    if not m:
        raise DocumentError("MISSING_FRONT_MATTER", "document must start with a '---' YAML front-matter block")
    try:
        meta = DocumentMetadata.model_validate(yaml.safe_load(m.group(1)) or {})
    except (yaml.YAMLError, ValidationError) as exc:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()) \
            if isinstance(exc, ValidationError) else str(exc)
        raise DocumentError("INVALID_METADATA", detail) from exc
    body = m.group(2).strip()
    if not re.search(r"^## \S", body, re.M):
        raise DocumentError("NO_SECTIONS", "document has no '## ' sections")
    return KnowledgeDocument(meta, body, str(path), hashlib.sha256(body.encode()).hexdigest())


def load_corpus(root: Path) -> Corpus:
    """Load every *.md under root (sorted, deterministic). Errors make the corpus invalid; warnings do not."""
    corpus = Corpus()
    root = Path(root)
    if not root.is_dir():
        corpus.issues.append(IngestIssue("error", "KNOWLEDGE_BASE_MISSING", str(root), "knowledge directory not found"))
        return corpus
    ids: dict[str, str] = {}
    bodies: dict[str, str] = {}
    for path in sorted(root.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            doc = parse_document(path)
        except DocumentError as exc:
            corpus.issues.append(IngestIssue("error", exc.code, str(path), exc.message))
            continue
        did = doc.metadata.document_id
        if did in ids:
            corpus.issues.append(IngestIssue("error", "DUPLICATE_DOCUMENT_ID", str(path),
                                             f"{did} already defined in {ids[did]}"))
            continue
        if doc.content_sha256 in bodies:
            corpus.issues.append(IngestIssue("warning", "DUPLICATE_CONTENT", str(path),
                                             f"same content as {bodies[doc.content_sha256]}; skipped"))
            continue
        if SUSPICIOUS.search(doc.body):
            corpus.issues.append(IngestIssue("warning", "SUSPICIOUS_INSTRUCTION_TEXT", str(path),
                                             "contains instruction-like text; it is treated as data, never as instructions"))
        ids[did], bodies[doc.content_sha256] = str(path), str(path)
        corpus.documents.append(doc)
    if not corpus.documents and not corpus.errors:
        corpus.issues.append(IngestIssue("error", "EMPTY_KNOWLEDGE_BASE", str(root), "no documents found"))
    return corpus
