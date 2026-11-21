"""Offline pytest fixtures with deterministic embedder, reranker, and LLM stubs."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

import pytest

from insurance_rag.chunking import Chunk, chunk_documents
from insurance_rag.ingest import Document, Section, load_documents
from insurance_rag.pipeline import InsuranceRAGPipeline
from insurance_rag.retriever import HybridRetriever, RetrievedChunk, expand_query, tokenize


class DeterministicHashEmbedder:
    """Offline bag-of-hashed-tokens embedder (zero network or model downloads)."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:2], "big") % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


class LexicalOverlapReranker:
    """Offline cross-encoder stub scoring query-passage overlap cleanly into [0, 1]."""

    _STOPWORDS = {
        "the", "is", "are", "a", "an", "in", "on", "of", "for", "to", "and",
        "or", "what", "when", "how", "many", "do", "does", "my", "i", "it",
    }

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        expanded = expand_query(query)
        q_tokens = {t for t in tokenize(expanded) if t not in self._STOPWORDS}
        if not q_tokens:
            return [0.0 for _ in passages]

        scores: list[float] = []
        for passage in passages:
            p_tokens = set(tokenize(passage))
            overlap = len(q_tokens & p_tokens)
            # Smooth ratio in [0, 1]
            raw = overlap / max(len(q_tokens), 1)
            scores.append(min(1.0, round(raw * 1.15, 4)))
        return scores


class StubGenerator:
    """Offline answer generator producing grounded inline citations from top passage."""

    def generate(
        self,
        question: str,
        passages: Sequence[RetrievedChunk],
        history: Sequence[dict[str, str]] | None = None,
    ) -> str:
        if not passages:
            return "I could not find this in the indexed documents."
        top = passages[0].chunk
        first_sentence = top.text.split(".")[0].strip()
        return f"Based on {top.title} ({top.section}), {first_sentence} [1]."


@pytest.fixture()
def sample_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "irdai_rules.md").write_text(
        "# IRDAI Health Insurance Rules\n\n"
        "## 1. Moratorium Period\n"
        "After completion of 60 continuous months of coverage, no claim can be contested "
        "on grounds of non-disclosure except established fraud.\n\n"
        "## 2. Health Insurance Portability\n"
        "A policyholder may apply for portability to switch to another insurer at least "
        "45 days before renewal, carrying forward waiting period credits.\n\n"
        "## 3. Claim Repudiation Review\n"
        "No claim shall be repudiated without formal approval by a three-member Claims "
        "Review Committee within 15 days of receiving the last document.\n",
        encoding="utf-8",
    )
    (raw_dir / "pmjay_guide.txt").write_text(
        "PM-JAY GUIDELINES\n\n"
        "1. COVERAGE LIMIT\n"
        "AB PM-JAY provides cashless coverage of Rs. 5,00,000 per family per year "
        "with zero waiting period for pre-existing diseases and zero co-payment.\n",
        encoding="utf-8",
    )
    return raw_dir


@pytest.fixture()
def sample_chunks(sample_raw_dir: Path) -> list[Chunk]:
    docs: list[Document] = load_documents(sample_raw_dir)
    return chunk_documents(docs, chunk_size=900, chunk_overlap=150)


@pytest.fixture()
def offline_retriever(sample_chunks: list[Chunk]) -> HybridRetriever:
    retriever = HybridRetriever(
        embedder=DeterministicHashEmbedder(),
        reranker=LexicalOverlapReranker(),
        rrf_k=60,
        candidate_top_k=20,
    )
    retriever.index_chunks(sample_chunks)
    return retriever


@pytest.fixture()
def offline_pipeline(offline_retriever: HybridRetriever) -> InsuranceRAGPipeline:
    return InsuranceRAGPipeline(
        retriever=offline_retriever,
        generator=StubGenerator(),
        abstention_threshold=0.25,
        final_top_k=5,
    )
