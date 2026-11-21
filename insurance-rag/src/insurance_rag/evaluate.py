"""Golden-set evaluation: hit@1/3/5, MRR, keyword coverage, abstention accuracy, and LLM faithfulness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from insurance_rag.generate import FaithfulnessJudge
from insurance_rag.pipeline import InsuranceRAGPipeline


@dataclass(frozen=True, slots=True)
class GoldenExample:
    """Single golden evaluation record from golden.jsonl."""

    id: str
    question: str
    expected_doc: str | None
    expected_section: str | None
    expected_keywords: list[str]


@dataclass(frozen=True, slots=True)
class QuestionEvalResult:
    """Per-question evaluation details."""

    id: str
    question: str
    expected_doc: str | None
    expected_section: str | None
    retrieved_docs: list[str]
    retrieved_sections: list[str]
    best_score: float
    abstained: bool
    hit_at_1: bool | None
    hit_at_3: bool | None
    hit_at_5: bool | None
    reciprocal_rank: float | None
    keyword_coverage: float | None
    abstention_correct: bool | None
    faithfulness_score: float | None
    answer: str


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """Aggregate evaluation metrics across the golden dataset."""

    total_questions: int
    answerable_questions: int
    out_of_scope_questions: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    keyword_coverage: float
    abstention_accuracy: float
    false_abstention_rate: float
    faithfulness_score: float | None
    generated_at: str


def load_golden_set(path: Path) -> list[GoldenExample]:
    """Load golden evaluation cases from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {path}")

    examples: list[GoldenExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        raw = json.loads(stripped)
        examples.append(
            GoldenExample(
                id=str(raw["id"]),
                question=str(raw["question"]),
                expected_doc=raw.get("expected_doc"),
                expected_section=raw.get("expected_section"),
                expected_keywords=[str(k) for k in raw.get("expected_keywords", [])],
            )
        )
    return examples


def _matches_expected(source: str, section: str, ex: GoldenExample) -> bool:
    if ex.expected_doc is None:
        return False
    doc_match = source == ex.expected_doc or source.endswith(f"/{ex.expected_doc}")
    if not doc_match:
        return False
    if ex.expected_section:
        return section.strip().lower() == ex.expected_section.strip().lower()
    return True


def evaluate_pipeline(
    pipeline: InsuranceRAGPipeline,
    golden_examples: list[GoldenExample],
    judge: FaithfulnessJudge | None = None,
    run_generation: bool = True,
) -> tuple[EvalSummary, list[QuestionEvalResult]]:
    """Run retrieval, abstention, keyword coverage, and optional LLM-as-judge evaluation."""
    results: list[QuestionEvalResult] = []

    answerable_count = 0
    oos_count = 0
    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    rr_sum = 0.0
    kw_scores: list[float] = []
    oos_correct = 0
    false_abstentions = 0
    faith_scores: list[float] = []

    for ex in golden_examples:
        retrieved = pipeline.retriever.retrieve(ex.question, top_k=5)
        best_score = max((r.score for r in retrieved), default=0.0)
        retrieved_docs = [r.chunk.source for r in retrieved]
        retrieved_sections = [r.chunk.section for r in retrieved]

        if run_generation:
            response = pipeline.ask(ex.question, top_k=5)
            abstained = response.abstained
            answer = response.answer
        else:
            abstained = (not retrieved) or (best_score < pipeline.abstention_threshold)
            answer = (
                "I could not find this in the indexed documents."
                if abstained
                else " ".join(r.chunk.text for r in retrieved)
            )

        if ex.expected_doc is not None:
            answerable_count += 1
            first_match_rank: int | None = None
            for idx, r in enumerate(retrieved, start=1):
                if _matches_expected(r.chunk.source, r.chunk.section, ex):
                    first_match_rank = idx
                    break

            h1 = first_match_rank == 1
            h3 = first_match_rank is not None and first_match_rank <= 3
            h5 = first_match_rank is not None and first_match_rank <= 5
            rr = (1.0 / first_match_rank) if first_match_rank is not None else 0.0

            hits_1 += int(h1)
            hits_3 += int(h3)
            hits_5 += int(h5)
            rr_sum += rr

            if abstained:
                false_abstentions += 1

            combined_text = (
                (answer + " " + " ".join(r.chunk.text for r in retrieved)).lower()
            )
            if ex.expected_keywords:
                matched_kw = sum(
                    1 for kw in ex.expected_keywords if kw.lower() in combined_text
                )
                kw_cov = matched_kw / len(ex.expected_keywords)
            else:
                kw_cov = 1.0
            kw_scores.append(kw_cov)

            faith_score: float | None = None
            if run_generation and judge is not None and not abstained and retrieved:
                judgement = judge.judge(ex.question, answer, retrieved)
                faith_score = judgement.faithfulness_score
                faith_scores.append(faith_score)

            results.append(
                QuestionEvalResult(
                    id=ex.id,
                    question=ex.question,
                    expected_doc=ex.expected_doc,
                    expected_section=ex.expected_section,
                    retrieved_docs=retrieved_docs,
                    retrieved_sections=retrieved_sections,
                    best_score=round(best_score, 4),
                    abstained=abstained,
                    hit_at_1=h1,
                    hit_at_3=h3,
                    hit_at_5=h5,
                    reciprocal_rank=round(rr, 4),
                    keyword_coverage=round(kw_cov, 4),
                    abstention_correct=None,
                    faithfulness_score=round(faith_score, 4) if faith_score is not None else None,
                    answer=answer,
                )
            )
        else:
            oos_count += 1
            abstain_ok = abstained
            if abstain_ok:
                oos_correct += 1

            results.append(
                QuestionEvalResult(
                    id=ex.id,
                    question=ex.question,
                    expected_doc=None,
                    expected_section=None,
                    retrieved_docs=retrieved_docs,
                    retrieved_sections=retrieved_sections,
                    best_score=round(best_score, 4),
                    abstained=abstained,
                    hit_at_1=None,
                    hit_at_3=None,
                    hit_at_5=None,
                    reciprocal_rank=None,
                    keyword_coverage=None,
                    abstention_correct=abstain_ok,
                    faithfulness_score=None,
                    answer=answer,
                )
            )

    summary = EvalSummary(
        total_questions=len(golden_examples),
        answerable_questions=answerable_count,
        out_of_scope_questions=oos_count,
        hit_at_1=round(hits_1 / answerable_count, 4) if answerable_count else 0.0,
        hit_at_3=round(hits_3 / answerable_count, 4) if answerable_count else 0.0,
        hit_at_5=round(hits_5 / answerable_count, 4) if answerable_count else 0.0,
        mrr=round(rr_sum / answerable_count, 4) if answerable_count else 0.0,
        keyword_coverage=round(sum(kw_scores) / len(kw_scores), 4) if kw_scores else 0.0,
        abstention_accuracy=round(oos_correct / oos_count, 4) if oos_count else 0.0,
        false_abstention_rate=(
            round(false_abstentions / answerable_count, 4) if answerable_count else 0.0
        ),
        faithfulness_score=(
            round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    return summary, results


def write_reports(
    reports_dir: Path,
    summary: EvalSummary,
    results: list[QuestionEvalResult],
) -> tuple[Path, Path]:
    """Write evaluation results to reports/latest.json and reports/latest.md."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "latest.json"
    md_path = reports_dir / "latest.md"

    payload: dict[str, Any] = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    faith_display = (
        f"{summary.faithfulness_score:.1%}"
        if summary.faithfulness_score is not None
        else "Skipped (set GEMINI_API_KEY and omit --skip-judge)"
    )

    lines = [
        "# Evaluation Report — insurance-rag",
        "",
        f"Generated at: `{summary.generated_at}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total Questions | {summary.total_questions} ({summary.answerable_questions} answerable, {summary.out_of_scope_questions} out-of-scope) |",
        f"| Hit@1 | {summary.hit_at_1:.1%} |",
        f"| Hit@3 | {summary.hit_at_3:.1%} |",
        f"| Hit@5 | {summary.hit_at_5:.1%} |",
        f"| MRR | {summary.mrr:.4f} |",
        f"| Keyword Coverage | {summary.keyword_coverage:.1%} |",
        f"| Abstention Accuracy (Out-of-Scope) | {summary.abstention_accuracy:.1%} |",
        f"| False Abstention Rate (Answerable) | {summary.false_abstention_rate:.1%} |",
        f"| LLM-as-Judge Faithfulness | {faith_display} |",
        "",
        "## Per-Question Breakdown",
        "",
        "| ID | Expected Doc | Best Score | Hit@1 | Hit@3 | Hit@5 | KW Cov | Abstained | Faithfulness |",
        "| --- | --- | ---: | :---: | :---: | :---: | ---: | :---: | ---: |",
    ]

    for r in results:
        doc_label = r.expected_doc or "OUT_OF_SCOPE"
        h1 = "-" if r.hit_at_1 is None else ("Y" if r.hit_at_1 else "N")
        h3 = "-" if r.hit_at_3 is None else ("Y" if r.hit_at_3 else "N")
        h5 = "-" if r.hit_at_5 is None else ("Y" if r.hit_at_5 else "N")
        kw = "-" if r.keyword_coverage is None else f"{r.keyword_coverage:.0%}"
        ab = "Yes" if r.abstained else "No"
        fa = "-" if r.faithfulness_score is None else f"{r.faithfulness_score:.2f}"
        lines.append(
            f"| {r.id} | {doc_label} | {r.best_score:.3f} | {h1} | {h3} | {h5} | {kw} | {ab} | {fa} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
