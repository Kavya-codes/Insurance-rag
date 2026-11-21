"""Heading-aware chunking (~900 characters, ~150 overlap) preserving title, section, and source."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from insurance_rag.ingest import Document, Section

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A self-contained text chunk with full provenance metadata."""

    chunk_id: str
    text: str
    title: str
    section: str
    source: str
    chunk_index: int
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=str(data["chunk_id"]),
            text=str(data["text"]),
            title=str(data["title"]),
            section=str(data["section"]),
            source=str(data["source"]),
            chunk_index=int(data["chunk_index"]),
            char_count=int(data.get("char_count", len(str(data["text"])))),
        )


def _split_long_unit(text: str, max_chars: int) -> list[str]:
    """Split a paragraph longer than max_chars along sentence or word boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_BOUNDARY_RE.split(text.strip())
    pieces: list[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            words = sent.split()
            word_buf = ""
            for word in words:
                candidate = f"{word_buf} {word}".strip() if word_buf else word
                if len(candidate) <= max_chars:
                    word_buf = candidate
                else:
                    if word_buf:
                        pieces.append(word_buf)
                    word_buf = word
            if word_buf:
                pieces.append(word_buf)
            continue

        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sent

    if current:
        pieces.append(current)

    return pieces


def _tail_overlap(text: str, overlap: int) -> str:
    """Extract up to `overlap` trailing characters snapped cleanly to a word boundary."""
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text
    snippet = text[-overlap:]
    space_idx = snippet.find(" ")
    if space_idx != -1 and space_idx < len(snippet) - 1:
        return snippet[space_idx + 1 :].strip()
    return snippet.strip()


def chunk_section(
    doc: Document,
    section: Section,
    start_index: int,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Chunk a single document section without crossing heading boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]
    split_target = max(chunk_size - chunk_overlap - 2, chunk_size // 2)
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_split_long_unit(para, split_target))

    if not units:
        return []

    raw_chunks: list[str] = []
    current = ""

    for unit in units:
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                raw_chunks.append(current)
                available_overlap = min(chunk_overlap, max(0, chunk_size - len(unit) - 2))
                overlap_prefix = _tail_overlap(current, available_overlap)
                current = f"{overlap_prefix}\n\n{unit}" if overlap_prefix else unit
            else:
                raw_chunks.append(unit)
                current = ""

    if current:
        raw_chunks.append(current)

    chunks: list[Chunk] = []
    for offset, chunk_text in enumerate(raw_chunks):
        idx = start_index + offset
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::c{idx:04d}",
                text=chunk_text.strip(),
                title=doc.title,
                section=section.heading,
                source=doc.source,
                chunk_index=idx,
                char_count=len(chunk_text.strip()),
            )
        )
    return chunks


def chunk_document(
    doc: Document,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Chunk all sections of a single Document while preserving section headings."""
    chunks: list[Chunk] = []
    next_index = 0
    for section in doc.sections:
        sec_chunks = chunk_section(
            doc=doc,
            section=section,
            start_index=next_index,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(sec_chunks)
        next_index += len(sec_chunks)
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Chunk a corpus of Documents into heading-aware chunks."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return all_chunks
