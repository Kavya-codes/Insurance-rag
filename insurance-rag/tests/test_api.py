"""Offline unit tests for the FastAPI endpoints (/health, /ask, /)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from insurance_rag.api import create_app
from insurance_rag.generate import ABSTENTION_TEXT, DISCLAIMER_TEXT
from insurance_rag.pipeline import InsuranceRAGPipeline


def test_health_endpoint_with_loaded_pipeline(offline_pipeline: InsuranceRAGPipeline) -> None:
    app = create_app(pipeline=offline_pipeline)
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["chunks_count"] == len(offline_pipeline.retriever.chunks)


def test_ask_endpoint_returns_cited_answer_and_sources(
    offline_pipeline: InsuranceRAGPipeline,
) -> None:
    app = create_app(pipeline=offline_pipeline)
    client = TestClient(app)

    res = client.post("/ask", json={"question": "What is the AB PM-JAY cashless coverage?"})
    assert res.status_code == 200
    body = res.json()
    assert body["abstained"] is False
    assert "[1]" in body["answer"]
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["source"] == "pmjay_guide.txt"
    assert body["disclaimer"] == DISCLAIMER_TEXT


def test_ask_endpoint_abstains_on_unrelated_question(
    offline_pipeline: InsuranceRAGPipeline,
) -> None:
    app = create_app(pipeline=offline_pipeline)
    client = TestClient(app)

    res = client.post(
        "/ask",
        json={"question": "What is the current Schengen visa processing fee in euros?"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["abstained"] is True
    assert body["answer"] == ABSTENTION_TEXT
    assert body["sources"] == []
    assert body["disclaimer"] == DISCLAIMER_TEXT


def test_ask_endpoint_rejects_blank_question(offline_pipeline: InsuranceRAGPipeline) -> None:
    app = create_app(pipeline=offline_pipeline)
    client = TestClient(app)

    res = client.post("/ask", json={"question": "   "})
    assert res.status_code == 422


def test_ui_homepage_renders(offline_pipeline: InsuranceRAGPipeline) -> None:
    app = create_app(pipeline=offline_pipeline)
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    assert "Indian Insurance RAG" in res.text
    assert "/ask" in res.text
