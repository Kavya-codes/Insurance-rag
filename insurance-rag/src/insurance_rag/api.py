"""FastAPI server exposing GET /health, POST /ask, and the mobile-friendly web UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from insurance_rag.config import Settings, load_settings
from insurance_rag.generate import DISCLAIMER_TEXT
from insurance_rag.pipeline import InsuranceRAGPipeline
from insurance_rag.retriever import CrossEncoderReranker, HybridRetriever, SentenceTransformerEmbedder

_STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question must not be blank.")
        return stripped


class SourceOut(BaseModel):
    """Source passage metadata returned in /ask responses."""

    index: int
    title: str
    section: str
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    """Response body for POST /ask."""

    answer: str
    sources: list[SourceOut]
    abstained: bool
    best_score: float = 0.0
    disclaimer: str = DISCLAIMER_TEXT


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    index_loaded: bool
    chunks_count: int


def create_app(
    pipeline: InsuranceRAGPipeline | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create a FastAPI application with optional injected pipeline for offline testing."""
    active_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if getattr(application.state, "pipeline", None) is None:
            try:
                retriever = HybridRetriever.load(
                    index_dir=active_settings.index_dir,
                    embedder=SentenceTransformerEmbedder(
                        model_name=active_settings.embedding_model
                    ),
                    reranker=CrossEncoderReranker(
                        model_name=active_settings.reranker_model
                    ),
                    rrf_k=active_settings.rrf_k,
                    candidate_top_k=active_settings.candidate_top_k,
                )
                application.state.pipeline = InsuranceRAGPipeline.from_settings(
                    active_settings, retriever=retriever
                )
            except FileNotFoundError:
                application.state.pipeline = None
        yield

    application = FastAPI(
        title="insurance-rag API",
        version="0.1.0",
        description="Grounded Indian insurance Q&A API with inline citations and strict abstention.",
        lifespan=lifespan,
    )
    application.state.pipeline = pipeline

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def serve_ui() -> HTMLResponse:
        if not _STATIC_INDEX.exists():
            raise HTTPException(status_code=404, detail="Web UI file not found.")
        return HTMLResponse(_STATIC_INDEX.read_text(encoding="utf-8"))

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current_pipeline: InsuranceRAGPipeline | None = getattr(
            application.state, "pipeline", None
        )
        if current_pipeline is None:
            return HealthResponse(status="ok", index_loaded=False, chunks_count=0)
        return HealthResponse(
            status="ok",
            index_loaded=True,
            chunks_count=len(current_pipeline.retriever.chunks),
        )

    @application.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        current_pipeline: InsuranceRAGPipeline | None = getattr(
            application.state, "pipeline", None
        )
        if current_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Search index is not loaded. Run `insurance-rag ingest` first.",
            )

        try:
            result = current_pipeline.ask(payload.question, top_k=payload.top_k)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return AskResponse(
            answer=result.answer,
            sources=[SourceOut(**s.to_dict()) for s in result.sources],
            abstained=result.abstained,
            best_score=result.best_score,
            disclaimer=result.disclaimer,
        )

    return application


app = create_app()
