"""End-to-end RAG pipeline orchestrating retrieval, abstention gating, and cited generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from insurance_rag.config import Settings
from insurance_rag.generate import (
    ABSTENTION_TEXT,
    DISCLAIMER_TEXT,
    AnswerGenerator,
    GeminiAnswerGenerator,
)
from insurance_rag.retriever import HybridRetriever, RetrievedChunk


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """Cited source passage returned alongside an answer."""

    index: int
    title: str
    section: str
    source: str
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnswerResponse:
    """Structured response produced by the RAG pipeline."""

    question: str
    answer: str
    sources: list[SourceCitation]
    abstained: bool
    best_score: float
    disclaimer: str = DISCLAIMER_TEXT

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "abstained": self.abstained,
            "best_score": self.best_score,
            "disclaimer": self.disclaimer,
        }

    def to_markdown(self, generated_at: str | None = None) -> str:
        """Format the answer and its cited sources as a documentation-ready Markdown document."""
        timestamp = generated_at or datetime.now(timezone.utc).isoformat()
        status = "Abstained (insufficient evidence)" if self.abstained else "Grounded Answer"
        lines: list[str] = [
            "# Indian Insurance RAG — Documentation Export",
            "",
            f"- **Generated:** {timestamp}",
            f"- **Status:** {status}",
            f"- **Best Retrieval Score:** {self.best_score:.3f}",
            "",
            "## Question",
            "",
            self.question,
            "",
            "## Answer",
            "",
            self.answer,
            "",
        ]
        if self.sources:
            lines.extend(["## Cited Sources", ""])
            for src in self.sources:
                lines.extend(
                    [
                        f"### [{src.index}] {src.title}",
                        "",
                        f"- **Section:** {src.section}",
                        f"- **File:** `{src.source}`",
                        f"- **Retrieval Score:** {src.score:.3f}",
                        "",
                    ]
                )
                quoted = "\n".join(
                    ">" if line.strip() == "" else f"> {line}"
                    for line in src.text.splitlines()
                )
                lines.extend([quoted, ""])
        lines.extend(["---", "", f"*{self.disclaimer}*", ""])
        return "\n".join(lines)


def is_abstention_answer(answer: str) -> bool:
    """Detect whether the generated text is an abstention response."""
    normalized = answer.strip().lower()
    return "could not find this in the indexed documents" in normalized


class InsuranceRAGPipeline:
    """Coordinates hybrid retrieval, score-based abstention, and grounded generation."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator | None = None,
        abstention_threshold: float = 0.25,
        final_top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or GeminiAnswerGenerator()
        self.abstention_threshold = abstention_threshold
        self.final_top_k = final_top_k

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        retriever: HybridRetriever | None = None,
        generator: AnswerGenerator | None = None,
    ) -> InsuranceRAGPipeline:
        loaded_retriever = retriever or HybridRetriever.load(
            index_dir=settings.index_dir,
            rrf_k=settings.rrf_k,
            candidate_top_k=settings.candidate_top_k,
        )
        active_generator = generator or GeminiAnswerGenerator(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            judge_model=settings.gemini_judge_model,
        )
        return cls(
            retriever=loaded_retriever,
            generator=active_generator,
            abstention_threshold=settings.abstention_threshold,
            final_top_k=settings.final_top_k,
        )

    @staticmethod
    def _build_citations(retrieved: Sequence[RetrievedChunk]) -> list[SourceCitation]:
        return [
            SourceCitation(
                index=idx,
                title=item.chunk.title,
                section=item.chunk.section,
                source=item.chunk.source,
                score=round(item.score, 4),
                text=item.chunk.text,
            )
            for idx, item in enumerate(retrieved, start=1)
        ]

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> AnswerResponse:
        cleaned_q = question.strip()
        if not cleaned_q:
            raise ValueError("Question must not be empty.")

        k = top_k or self.final_top_k
        retrieved = self.retriever.retrieve(cleaned_q, top_k=k)
        best_score = max((r.score for r in retrieved), default=0.0)

        if not retrieved or best_score < self.abstention_threshold:
            return AnswerResponse(
                question=cleaned_q,
                answer=ABSTENTION_TEXT,
                sources=[],
                abstained=True,
                best_score=round(best_score, 4),
                disclaimer=DISCLAIMER_TEXT,
            )

        raw_answer = self.generator.generate(
            question=cleaned_q,
            passages=retrieved,
            history=history,
        )
        abstained = is_abstention_answer(raw_answer)
        sources = [] if abstained else self._build_citations(retrieved)

        return AnswerResponse(
            question=cleaned_q,
            answer=ABSTENTION_TEXT if abstained else raw_answer.strip(),
            sources=sources,
            abstained=abstained,
            best_score=round(best_score, 4),
            disclaimer=DISCLAIMER_TEXT,
        )
