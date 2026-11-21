"""Offline unit tests for query expansion, hybrid RRF fusion, and reranking."""

from __future__ import annotations

from pathlib import Path

from insurance_rag.retriever import HybridRetriever, expand_query, reciprocal_rank_fusion
from tests.conftest import DeterministicHashEmbedder, LexicalOverlapReranker


def test_query_expansion_adds_insurance_synonyms() -> None:
    q1 = expand_query("Why would an insurer reject my claim?")
    assert "repudiation" in q1.lower()

    q2 = expand_query("How do I switch insurer before renewal?")
    assert "portability" in q2.lower()


def test_reciprocal_rank_fusion_combines_ranked_lists() -> None:
    fused = reciprocal_rank_fusion([[2, 0, 1], [2, 1, 0]], k=60)
    top_idx, top_score = fused[0]
    assert top_idx == 2
    assert top_score > fused[1][1]


def test_hybrid_retriever_finds_expected_sections(offline_retriever: HybridRetriever) -> None:
    results = offline_retriever.retrieve("How do I switch insurer at renewal?", top_k=3)
    assert results
    assert results[0].chunk.section == "2. Health Insurance Portability"

    reject_results = offline_retriever.retrieve("Can a company reject my claim?", top_k=3)
    assert reject_results
    assert reject_results[0].chunk.section == "3. Claim Repudiation Review"


def test_retriever_passes_top_20_candidates_to_reranker(sample_chunks) -> None:
    observed_counts: list[int] = []

    class SpyReranker(LexicalOverlapReranker):
        def score(self, query: str, passages):
            observed_counts.append(len(passages))
            return super().score(query, passages)

    # Duplicate sample chunks so corpus > 20
    big_corpus = []
    for rep in range(6):
        for c in sample_chunks:
            big_corpus.append(
                type(c)(
                    chunk_id=f"{c.chunk_id}_r{rep}",
                    text=c.text,
                    title=c.title,
                    section=c.section,
                    source=c.source,
                    chunk_index=c.chunk_index,
                    char_count=c.char_count,
                )
            )

    retriever = HybridRetriever(
        embedder=DeterministicHashEmbedder(),
        reranker=SpyReranker(),
        candidate_top_k=20,
    )
    retriever.index_chunks(big_corpus)
    results = retriever.retrieve("What is the PM-JAY coverage limit?", top_k=5)
    assert len(results) == 5
    assert observed_counts == [20]


def test_index_save_and_load_roundtrip(tmp_path: Path, offline_retriever: HybridRetriever) -> None:
    index_dir = tmp_path / "index"
    offline_retriever.save(index_dir)

    reloaded = HybridRetriever.load(
        index_dir=index_dir,
        embedder=DeterministicHashEmbedder(),
        reranker=LexicalOverlapReranker(),
    )
    results = reloaded.retrieve("What is the moratorium period?", top_k=2)
    assert results
    assert results[0].chunk.section == "1. Moratorium Period"
