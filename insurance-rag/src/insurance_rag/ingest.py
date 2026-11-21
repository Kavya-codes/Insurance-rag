"""Document ingestion for .pdf, .md, and .txt files with text cleaning and heading extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TXT_NUMBERED_HEADING_RE = re.compile(
    r"^(?:(?:SECTION|CHAPTER|ARTICLE|PART|CLAUSE)\s+\d+[A-Z0-9.\-]*\s*[:.\-]?\s*.+|\d+(?:\.\d+)*\.\s+[A-Z][A-Za-z0-9 ,/&()'\-]+)$"
)
_PAGE_MARKER_RE = re.compile(r"^\s*(?:Page\s+\d+\s*(?:of\s+\d+)?|[-_]{3,}\s*\d*\s*[-_]{0,})\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Section:
    """A logical section of a source document."""

    heading: str
    level: int
    text: str


@dataclass(frozen=True, slots=True)
class Document:
    """Cleaned source document broken into heading-aware sections."""

    doc_id: str
    title: str
    source: str
    file_type: str
    sections: list[Section] = field(default_factory=list)


def clean_text(raw: str) -> str:
    """Clean PDF line-break hyphenation and erratic whitespace while preserving paragraph structure."""
    if not raw:
        return ""

    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    # Rejoin words split across lines with a hyphen (common in PDF extraction)
    text = re.sub(r"([A-Za-z0-9])-\s*\n\s*([A-Za-z0-9])", r"\1\2", text)

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        if _PAGE_MARKER_RE.match(line):
            continue
        normalized_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(normalized_line)

    rejoined_paragraphs: list[str] = []
    buffer: list[str] = []

    def _flush() -> None:
        if buffer:
            rejoined_paragraphs.append(" ".join(buffer).strip())
            buffer.clear()

    for line in cleaned_lines:
        if not line:
            _flush()
            continue
        if _is_heading_line(line):
            _flush()
            rejoined_paragraphs.append(line)
            continue
        buffer.append(line)
    _flush()

    out = "\n\n".join(part for part in rejoined_paragraphs if part)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _is_heading_line(line: str) -> bool:
    if _MD_HEADING_RE.match(line):
        return True
    if len(line) <= 110 and _TXT_NUMBERED_HEADING_RE.match(line):
        return True
    if 4 <= len(line) <= 90 and line.isupper() and any(ch.isalpha() for ch in line):
        return True
    return False


def _humanize_filename(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return re.sub(r"\s+", " ", stem).title() or path.name


def _parse_markdown_sections(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    title = fallback_title
    sections: list[Section] = []
    current_heading = "Overview"
    current_level = 1
    current_blocks: list[str] = []

    for block in text.split("\n\n"):
        stripped = block.strip()
        if not stripped:
            continue
        m = _MD_HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            if level == 1 and title == fallback_title and not sections and not current_blocks:
                title = heading_text
                current_heading = heading_text
                current_level = 1
                continue
            if current_blocks:
                sections.append(
                    Section(
                        heading=current_heading,
                        level=current_level,
                        text="\n\n".join(current_blocks).strip(),
                    )
                )
                current_blocks = []
            current_heading = heading_text
            current_level = level
        else:
            current_blocks.append(stripped)

    if current_blocks:
        sections.append(
            Section(
                heading=current_heading,
                level=current_level,
                text="\n\n".join(current_blocks).strip(),
            )
        )

    if not sections and text.strip():
        sections.append(Section(heading="Overview", level=1, text=text.strip()))

    return title, sections


def _parse_text_sections(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    title = fallback_title
    sections: list[Section] = []
    current_heading = "General Provisions"
    current_blocks: list[str] = []

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    for idx, block in enumerate(blocks):
        is_all_caps_title = (
            idx == 0
            and len(block) <= 120
            and "\n" not in block
            and block.isupper()
            and not _TXT_NUMBERED_HEADING_RE.match(block)
        )
        if is_all_caps_title:
            title = block.title()
            continue

        if len(block) <= 120 and (
            _TXT_NUMBERED_HEADING_RE.match(block)
            or (block.isupper() and any(ch.isalpha() for ch in block))
        ):
            if current_blocks:
                sections.append(
                    Section(
                        heading=current_heading,
                        level=2,
                        text="\n\n".join(current_blocks).strip(),
                    )
                )
                current_blocks = []
            current_heading = block.strip()
        else:
            current_blocks.append(block)

    if current_blocks:
        sections.append(
            Section(
                heading=current_heading,
                level=2,
                text="\n\n".join(current_blocks).strip(),
            )
        )

    if not sections and text.strip():
        sections.append(Section(heading="General Provisions", level=1, text=text.strip()))

    return title, sections


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to ingest .pdf files. Install dependencies via `pip install -r requirements.txt`."
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages.append(extracted)
    return "\n\n".join(pages)


def load_document(path: Path, base_dir: Path | None = None) -> Document:
    """Load and clean a single .pdf, .md, or .txt document."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}' for {path}")

    source_name = str(path.relative_to(base_dir)) if base_dir else path.name
    fallback_title = _humanize_filename(path)

    if ext == ".pdf":
        raw_text = _read_pdf_text(path)
        cleaned = clean_text(raw_text)
        title, sections = _parse_text_sections(cleaned, fallback_title)
    elif ext == ".md":
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(raw_text)
        title, sections = _parse_markdown_sections(cleaned, fallback_title)
    else:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(raw_text)
        title, sections = _parse_text_sections(cleaned, fallback_title)

    doc_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", source_name).strip("_")
    return Document(
        doc_id=doc_id,
        title=title,
        source=source_name.replace("\\", "/"),
        file_type=ext.lstrip("."),
        sections=sections,
    )


def load_documents(raw_dir: Path) -> list[Document]:
    """Recursively load all supported documents from raw_dir in deterministic order."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw documents directory does not exist: {raw_dir}")

    files = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    documents: list[Document] = []
    for file_path in files:
        doc = load_document(file_path, base_dir=raw_dir)
        if doc.sections:
            documents.append(doc)
    return documents
