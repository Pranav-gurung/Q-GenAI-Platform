"""Approved-document extraction and chunking (PDF via PyMuPDF, falling back to pypdf; .txt/.md).

Only files that the operator placed in the approved-documents directory are read. Nothing is fetched
from the internet. Scanned/image-only PDFs are detected and reported as ``ocr_required`` (no OCR is performed).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md")
_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,80}|[A-Z][A-Z0-9 \-/&,()]{3,60})$")


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class ExtractedDocument:
    pages: list
    title_hint: Optional[str] = None
    backend: str = ""


@dataclass
class DocMeta:
    document_id: str
    title: str
    version: str
    source_filename: str
    content_hash: str
    access_level: str = "internal"
    license_note: str = ""
    metadata_source: str = "sidecar"  # sidecar | inferred


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    version: str
    source_filename: str
    page: int
    section: Optional[str]
    text: str
    content_hash: str  # hash of the source document revision
    access_level: str


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def extract_pdf(path: Path) -> ExtractedDocument:
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # PyMuPDF
        with fitz.open(str(path)) as doc:
            pages = [Page(i + 1, p.get_text("text") or "") for i, p in enumerate(doc)]
            title = (doc.metadata or {}).get("title") or None
        return ExtractedDocument(pages, title, "pymupdf")
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PDF extraction needs PyMuPDF (preferred) or pypdf: pip install pymupdf") from e
    reader = PdfReader(str(path))
    pages = [Page(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    title = None
    try:
        title = (reader.metadata.title if reader.metadata else None) or None
    except Exception:
        pass
    return ExtractedDocument(pages, title, "pypdf")


def extract_text_file(path: Path) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Pseudo-pages: form feed characters delimit pages, otherwise the whole file is page 1.
    parts = text.split("\f") if "\f" in text else [text]
    return ExtractedDocument([Page(i + 1, t) for i, t in enumerate(parts)], None, "text")


def extract(path: Path) -> ExtractedDocument:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return extract_pdf(path)
    if suf in (".txt", ".md"):
        return extract_text_file(path)
    raise ValueError(f"unsupported document type {suf!r}")


def usable_pages(doc: ExtractedDocument, min_chars: int) -> tuple:
    good = [p for p in doc.pages if len(p.text.strip()) >= min_chars]
    empty = [p.number for p in doc.pages if len(p.text.strip()) < min_chars]
    return good, empty


def load_meta(path: Path, content_hash: str, title_hint: Optional[str]) -> DocMeta:
    """Metadata comes from ``<file>.meta.json`` next to the document; otherwise it is inferred and flagged."""
    side = path.with_name(path.name + ".meta.json")
    if side.exists():
        m = json.loads(side.read_text(encoding="utf-8"))
        missing = [k for k in ("document_id", "title", "version") if not m.get(k)]
        if missing:
            raise ValueError(f"{side.name}: missing required fields {missing}")
        return DocMeta(m["document_id"], m["title"], str(m["version"]), path.name, content_hash,
                       m.get("access_level", "internal"), m.get("license_note", ""), "sidecar")
    return DocMeta(re.sub(r"[^A-Za-z0-9._\-]+", "-", path.stem), title_hint or path.stem, "unspecified", path.name,
                   content_hash, "internal", "", "inferred")


def _detect_heading(text: str, current: Optional[str]) -> Optional[str]:
    for line in text.splitlines():
        s = line.strip()
        if _HEADING_RE.match(s) and not s.endswith("."):
            current = s
    return current


def chunk_page(text: str, size: int, overlap: int) -> list:
    """Split into ~``size``-char windows with ``overlap``, preferring sentence/whitespace boundaries."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind(".\n"), window.rfind("\n\n"))
            if cut < size * 0.6:
                cut = window.rfind(" ")
            if cut >= size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def build_chunks(meta: DocMeta, pages: list, size: int, overlap: int) -> list:
    """Chunk page by page so every chunk has an exact page number (overlap never crosses pages)."""
    chunks, heading, seen = [], None, set()
    for p in pages:
        for i, piece in enumerate(chunk_page(p.text, size, overlap)):
            heading = _detect_heading(piece, heading)  # heuristic: last heading-like line seen so far
            digest = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:10]
            cid = f"{meta.document_id}:p{p.number}:c{i}:{digest}"
            if cid in seen:
                continue
            seen.add(cid)
            chunks.append(Chunk(cid, meta.document_id, meta.title, meta.version, meta.source_filename, p.number,
                                heading, piece, meta.content_hash, meta.access_level))
    return chunks
