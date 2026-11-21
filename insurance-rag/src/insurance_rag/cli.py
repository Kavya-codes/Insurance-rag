"""Command-line interface for insurance-rag: ingest, ask, chat, eval, serve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from insurance_rag.chunking import chunk_documents
from insurance_rag.config import Settings, load_settings
from insurance_rag.evaluate import evaluate_pipeline, load_golden_set, write_reports
from insurance_rag.generate import GeminiAnswerGenerator
from insurance_rag.ingest import load_documents
from insurance_rag.pipeline import AnswerResponse, InsuranceRAGPipeline
from insurance_rag.retriever import CrossEncoderReranker, HybridRetriever, SentenceTransformerEmbedder


def _build_retriever_from_settings(settings: Settings) -> HybridRetriever:
    embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    return HybridRetriever(
        embedder=embedder,
        reranker=reranker,
        rrf_k=settings.rrf_k,
        candidate_top_k=settings.candidate_top_k,
    )


def _print_response(resp: AnswerResponse) -> None:
    print()
    print(resp.answer)
    if resp.sources:
        print("\nSources:")
        for src in resp.sources:
            print(
                f"  [{src.index}] {src.title} — {src.section} "
                f"(file: {src.source}, score: {src.score:.3f})"
            )
    print(f"\n{resp.disclaimer}\n")


def cmd_ingest(settings: Settings) -> int:
    docs = load_documents(settings.raw_data_dir)
    if not docs:
        print(f"No supported documents (.pdf, .md, .txt) found in {settings.raw_data_dir}")
        return 1

    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retriever = _build_retriever_from_settings(settings)
    retriever.index_chunks(chunks)
    retriever.save(settings.index_dir)

    print(
        f"Ingested {len(docs)} documents -> {len(chunks)} chunks "
        f"(saved index to {settings.index_dir})"
    )
    return 0


def cmd_ask(
    settings: Settings,
    question: str,
    top_k: int | None = None,
    export_path: Path | None = None,
) -> int:
    retriever = HybridRetriever.load(
        index_dir=settings.index_dir,
        embedder=SentenceTransformerEmbedder(model_name=settings.embedding_model),
        reranker=CrossEncoderReranker(model_name=settings.reranker_model),
        rrf_k=settings.rrf_k,
        candidate_top_k=settings.candidate_top_k,
    )
    pipeline = InsuranceRAGPipeline.from_settings(settings, retriever=retriever)
    resp = pipeline.ask(question, top_k=top_k)
    _print_response(resp)
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(resp.to_markdown(), encoding="utf-8")
        print(f"Markdown documentation written to {export_path}")
    return 0


def cmd_chat(settings: Settings) -> int:
    retriever = HybridRetriever.load(
        index_dir=settings.index_dir,
        embedder=SentenceTransformerEmbedder(model_name=settings.embedding_model),
        reranker=CrossEncoderReranker(model_name=settings.reranker_model),
        rrf_k=settings.rrf_k,
        candidate_top_k=settings.candidate_top_k,
    )
    pipeline = InsuranceRAGPipeline.from_settings(settings, retriever=retriever)
    history: list[dict[str, str]] = []

    print("insurance-rag interactive chat (type 'exit' or 'quit' to stop)")
    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            print("Goodbye.")
            break

        resp = pipeline.ask(user_input, history=history)
        _print_response(resp)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": resp.answer})
    return 0


def cmd_eval(settings: Settings, skip_llm: bool = False, skip_judge: bool = False) -> int:
    retriever = HybridRetriever.load(
        index_dir=settings.index_dir,
        embedder=SentenceTransformerEmbedder(model_name=settings.embedding_model),
        reranker=CrossEncoderReranker(model_name=settings.reranker_model),
        rrf_k=settings.rrf_k,
        candidate_top_k=settings.candidate_top_k,
    )
    generator = GeminiAnswerGenerator(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        judge_model=settings.gemini_judge_model,
    )
    pipeline = InsuranceRAGPipeline.from_settings(
        settings, retriever=retriever, generator=generator
    )

    has_api_key = bool(settings.gemini_api_key)
    run_generation = (not skip_llm) and has_api_key
    judge = generator if (run_generation and not skip_judge and has_api_key) else None

    if not has_api_key and not skip_llm:
        print(
            "GEMINI_API_KEY not set; running retrieval & abstention evaluation without LLM generation/judge."
        )

    examples = load_golden_set(settings.golden_path)
    summary, results = evaluate_pipeline(
        pipeline=pipeline,
        golden_examples=examples,
        judge=judge,
        run_generation=run_generation,
    )
    json_path, md_path = write_reports(settings.reports_dir, summary, results)

    print(
        f"\nEvaluation Summary ({summary.total_questions} questions):\n"
        f"  Hit@1:               {summary.hit_at_1:.1%}\n"
        f"  Hit@3:               {summary.hit_at_3:.1%}\n"
        f"  Hit@5:               {summary.hit_at_5:.1%}\n"
        f"  MRR:                 {summary.mrr:.4f}\n"
        f"  Keyword Coverage:    {summary.keyword_coverage:.1%}\n"
        f"  Abstention Accuracy: {summary.abstention_accuracy:.1%}\n"
        f"  Faithfulness:        "
        + (f"{summary.faithfulness_score:.1%}" if summary.faithfulness_score is not None else "Skipped")
        + f"\n\nReports written to {md_path} and {json_path}"
    )
    return 0


def cmd_serve(settings: Settings, host: str | None = None, port: int | None = None) -> int:
    import uvicorn

    uvicorn.run(
        "insurance_rag.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insurance-rag",
        description="Indian Insurance RAG CLI (ingest, ask, chat, eval, serve)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Optional project root directory (defaults to current working directory).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Ingest documents from data/raw/ and build hybrid index.")

    ask_p = sub.add_parser("ask", help="Ask a single question against the indexed documents.")
    ask_p.add_argument("question", type=str, help="Question to answer.")
    ask_p.add_argument("--top-k", type=int, default=None, help="Number of passages to retrieve.")
    ask_p.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional output .md file path to export the answer and cited sources.",
    )

    sub.add_parser("chat", help="Start an interactive multi-turn Q&A session.")

    eval_p = sub.add_parser("eval", help="Run golden-set evaluation and write reports/latest.*.")
    eval_p.add_argument("--skip-llm", action="store_true", help="Evaluate retrieval & abstention only.")
    eval_p.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge faithfulness scoring.")

    serve_p = sub.add_parser("serve", help="Start the FastAPI web server.")
    serve_p.add_argument("--host", type=str, default=None, help="Bind host.")
    serve_p.add_argument("--port", type=int, default=None, help="Bind port.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(base_dir=args.base_dir)

    if args.command == "ingest":
        return cmd_ingest(settings)
    if args.command == "ask":
        return cmd_ask(
            settings,
            question=args.question,
            top_k=args.top_k,
            export_path=args.export,
        )
    if args.command == "chat":
        return cmd_chat(settings)
    if args.command == "eval":
        return cmd_eval(settings, skip_llm=args.skip_llm, skip_judge=args.skip_judge)
    if args.command == "serve":
        return cmd_serve(settings, host=args.host, port=args.port)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
