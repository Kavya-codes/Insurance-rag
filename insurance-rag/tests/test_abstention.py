"""Offline unit tests for abstention gating, grounded citations, and disclaimers."""

from __future__ import annotations

from insurance_rag.generate import ABSTENTION_TEXT, DISCLAIMER_TEXT
from insurance_rag.pipeline import InsuranceRAGPipeline


def test_pipeline_answers_with_inline_citations_and_sources(
    offline_pipeline: InsuranceRAGPipeline,
) -> None:
    resp = offline_pipeline.ask("What is the moratorium period after 60 continuous months?")
    assert resp.abstained is False
    assert "[1]" in resp.answer
    assert resp.sources
    assert resp.sources[0].section == "1. Moratorium Period"
    assert resp.sources[0].source == "irdai_rules.md"
    assert resp.disclaimer == DISCLAIMER_TEXT


def test_pipeline_abstains_on_out_of_scope_query(
    offline_pipeline: InsuranceRAGPipeline,
) -> None:
    resp = offline_pipeline.ask("What is the monthly premium for US Medicare Part D?")
    assert resp.abstained is True
    assert resp.answer == ABSTENTION_TEXT
    assert resp.sources == []
    assert resp.best_score < offline_pipeline.abstention_threshold
    assert resp.disclaimer == DISCLAIMER_TEXT


def test_pipeline_abstains_when_generator_refuses_even_if_score_passes(
    offline_retriever,
) -> None:
    class RefusingGenerator:
        def generate(self, question, passages, history=None) -> str:
            return ABSTENTION_TEXT

    pipeline = InsuranceRAGPipeline(
        retriever=offline_retriever,
        generator=RefusingGenerator(),
        abstention_threshold=0.01,
    )
    resp = pipeline.ask("What is the moratorium period?")
    assert resp.abstained is True
    assert resp.answer == ABSTENTION_TEXT
    assert resp.sources == []


def test_answer_response_to_markdown_includes_citations_and_sources(
    offline_pipeline: InsuranceRAGPipeline,
) -> None:
    resp = offline_pipeline.ask("What is the moratorium period after 60 continuous months?")
    md = resp.to_markdown(generated_at="2026-11-20T15:00:00+00:00")
    assert "# Indian Insurance RAG — Documentation Export" in md
    assert "## Question" in md
    assert "## Answer" in md
    assert "## Cited Sources" in md
    assert "### [1]" in md
    assert "irdai_rules.md" in md
    assert DISCLAIMER_TEXT in md

