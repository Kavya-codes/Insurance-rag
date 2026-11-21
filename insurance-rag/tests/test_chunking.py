"""Offline unit tests for document cleaning and heading-aware chunking."""

from __future__ import annotations

from pathlib import Path

from insurance_rag.chunking import chunk_document, chunk_documents
from insurance_rag.ingest import Document, Section, clean_text, load_documents


def test_clean_text_fixes_pdf_hyphenation_and_whitespace() -> None:
    raw = (
        "Pre-exist-\ning diseases are cov-\nered after   36 months.\n"
        "Page 2 of 10\n"
        "---\n"
        "Cashless   authori-\nzation takes   1 hour."
    )
    cleaned = clean_text(raw)
    assert "Pre-existing" in cleaned or "Preexisting" in cleaned
    assert "covered after 36 months." in cleaned
    assert "authorization takes 1 hour." in cleaned
    assert "Page 2 of 10" not in cleaned


def test_heading_aware_chunking_never_crosses_sections() -> None:
    doc = Document(
        doc_id="irdai_doc",
        title="IRDAI Circular",
        source="irdai_doc.md",
        file_type="md",
        sections=[
            Section(
                heading="1. Free Look Period",
                level=2,
                text="Policyholders receive a 30-day free look period to cancel.",
            ),
            Section(
                heading="2. Moratorium Period",
                level=2,
                text="After 60 continuous months, claims cannot be contested except for fraud.",
            ),
        ],
    )
    chunks = chunk_document(doc, chunk_size=900, chunk_overlap=150)
    assert len(chunks) == 2
    assert chunks[0].section == "1. Free Look Period"
    assert chunks[1].section == "2. Moratorium Period"
    assert chunks[0].title == "IRDAI Circular"
    assert chunks[0].source == "irdai_doc.md"
    assert "60 continuous months" not in chunks[0].text


def test_long_section_splits_around_chunk_size_with_overlap() -> None:
    para_a = " ".join(["grace"] * 90) + "."
    para_b = " ".join(["continuity"] * 90) + "."
    para_c = " ".join(["renewal"] * 90) + "."
    doc = Document(
        doc_id="long_doc",
        title="Long Policy",
        source="long_policy.md",
        file_type="md",
        sections=[
            Section(
                heading="3. Renewal Terms",
                level=2,
                text=f"{para_a}\n\n{para_b}\n\n{para_c}",
            )
        ],
    )
    chunks = chunk_document(doc, chunk_size=900, chunk_overlap=150)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= 900
        assert chunk.section == "3. Renewal Terms"
        assert chunk.source == "long_policy.md"

    # Verify trailing overlap from chunk 0 appears at start of chunk 1
    tail_word = chunks[0].text.split()[-1]
    assert chunks[1].text.startswith(tail_word) or tail_word in chunks[1].text[:180]


def test_load_and_chunk_raw_directory(sample_raw_dir: Path) -> None:
    docs = load_documents(sample_raw_dir)
    chunks = chunk_documents(docs, chunk_size=900, chunk_overlap=150)
    assert len(docs) == 2
    assert len(chunks) >= 4
    sources = {c.source for c in chunks}
    assert sources == {"irdai_rules.md", "pmjay_guide.txt"}
