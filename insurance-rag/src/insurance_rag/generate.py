"""Grounded answer generation and LLM-as-judge faithfulness scoring via Gemini API."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from insurance_rag.retriever import RetrievedChunk

ABSTENTION_TEXT = "I could not find this in the indexed documents."
DISCLAIMER_TEXT = (
    "Disclaimer: This response provides general information based only on the indexed "
    "insurance documents and is not legal or financial advice."
)

_SYSTEM_INSTRUCTION = (
    "You are an Indian insurance information assistant. Answer the user's question "
    "STRICTLY and ONLY from the numbered context passages provided.\n"
    "Rules:\n"
    "1. Keep the answer concise, plain-English, and factual.\n"
    "2. Cite supporting passages inline using bracketed numbers like [1], [2].\n"
    "3. Never invent numbers, waiting periods, limits, dates, or policy clauses not explicitly stated in the passages.\n"
    "4. If the passages do not contain enough information to answer the question, respond with exact text:\n"
    f'"{ABSTENTION_TEXT}"'
)

_JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict RAG faithfulness evaluator. Given a question, an answer, and the "
    "numbered context passages, check whether EVERY factual claim in the answer is directly "
    "supported by the cited passages.\n"
    "Return valid JSON only with keys:\n"
    '- "total_claims": integer\n'
    '- "supported_claims": integer\n'
    '- "faithfulness_score": float between 0.0 and 1.0 (supported_claims / total_claims, or 1.0 if abstained appropriately)\n'
    '- "unsupported_claims": list of strings\n'
    '- "reasoning": short one-sentence explanation'
)


@dataclass(frozen=True, slots=True)
class FaithfulnessJudgement:
    """Structured LLM-as-judge faithfulness evaluation output."""

    faithfulness_score: float
    supported_claims: int
    total_claims: int
    unsupported_claims: list[str]
    reasoning: str


class AnswerGenerator(Protocol):
    """Protocol for answer generation backends (enables offline test mocks)."""

    def generate(
        self,
        question: str,
        passages: Sequence[RetrievedChunk],
        history: Sequence[dict[str, str]] | None = None,
    ) -> str:
        ...


class FaithfulnessJudge(Protocol):
    """Protocol for LLM-as-judge faithfulness evaluators."""

    def judge(
        self,
        question: str,
        answer: str,
        passages: Sequence[RetrievedChunk],
    ) -> FaithfulnessJudgement:
        ...


def format_numbered_passages(passages: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks into numbered context blocks for prompting."""
    blocks: list[str] = []
    for idx, item in enumerate(passages, start=1):
        c = item.chunk
        blocks.append(
            f"[{idx}] Document: {c.title} | Section: {c.section} | File: {c.source}\n{c.text}"
        )
    return "\n\n".join(blocks)


class GeminiAnswerGenerator:
    """Answer generator backed by the official google-genai SDK."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-4-hydrogen",
        judge_model: str = "gemini-4-hydrogen",
    ) -> None:
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.model = model
        self.judge_model = judge_model
        self._client: object | None = None

    def _get_client(self) -> object:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export GEMINI_API_KEY in your environment or .env file."
            )
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _call_with_retry(self, model_name: str, contents: str, config: object) -> object:
        import time

        client = self._get_client()
        candidates = (
            [model_name, "gemini-3.1-flash-lite"]
            if model_name == "gemini-4-hydrogen"
            else [model_name]
        )
        last_exc: Exception | None = None
        for candidate in candidates:
            for delay in (0.0, 0.5, 1.0):
                if delay > 0:
                    time.sleep(delay)
                try:
                    return client.models.generate_content(  # type: ignore[attr-defined]
                        model=candidate,
                        contents=contents,
                        config=config,
                    )
                except Exception as exc:  # pragma: no cover - network path
                    last_exc = exc
                    msg = str(exc).upper()
                    transient = any(
                        token in msg
                        for token in ("503", "429", "UNAVAILABLE", "HIGH DEMAND", "RESOURCE_EXHAUSTED", "TIMEOUT")
                    )
                    if not transient:
                        raise
        assert last_exc is not None
        raise last_exc

    def generate(
        self,
        question: str,
        passages: Sequence[RetrievedChunk],
        history: Sequence[dict[str, str]] | None = None,
    ) -> str:
        if not passages:
            return ABSTENTION_TEXT

        from google.genai import types

        context_block = format_numbered_passages(passages)
        history_block = ""
        if history:
            turns = [f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]]
            history_block = "Recent conversation:\n" + "\n".join(turns) + "\n\n"

        prompt = (
            f"{history_block}"
            f"Context passages:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely with inline citations [1], [2] using ONLY the context passages above."
        )

        response = self._call_with_retry(
            self.model,
            prompt,
            types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
            ),
        )
        text = (getattr(response, "text", None) or "").strip()
        return text or ABSTENTION_TEXT

    def judge(
        self,
        question: str,
        answer: str,
        passages: Sequence[RetrievedChunk],
    ) -> FaithfulnessJudgement:
        from google.genai import types

        context_block = format_numbered_passages(passages)
        prompt = (
            f"Question: {question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Context passages:\n{context_block}\n\n"
            "Verify every factual claim in the Answer against the Context passages."
        )

        response = self._call_with_retry(
            self.judge_model,
            prompt,
            types.GenerateContentConfig(
                system_instruction=_JUDGE_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            ),
        )
        raw_text = (getattr(response, "text", None) or "{}").strip()
        # Strip accidental markdown fences if any
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return FaithfulnessJudgement(
                faithfulness_score=0.0,
                supported_claims=0,
                total_claims=1,
                unsupported_claims=["Judge response was not valid JSON."],
                reasoning=cleaned[:200],
            )

        total = max(int(payload.get("total_claims", 1)), 1)
        supported = max(0, min(int(payload.get("supported_claims", 0)), total))
        score = float(payload.get("faithfulness_score", supported / total))
        score = max(0.0, min(1.0, score))
        unsupported = [str(u) for u in payload.get("unsupported_claims", [])]
        reasoning = str(payload.get("reasoning", ""))
        return FaithfulnessJudgement(
            faithfulness_score=score,
            supported_claims=supported,
            total_claims=total,
            unsupported_claims=unsupported,
            reasoning=reasoning,
        )
