"""Hybrid retriever: query expansion, BM25 + dense embeddings, RRF fusion, and cross-encoder reranking."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from insurance_rag.chunking import Chunk

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore[assignment]


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# Domain vocabulary expansion for Indian health & general insurance queries
INSURANCE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "reject": ("repudiation", "repudiated", "claim rejection"),
    "rejected": ("repudiation", "repudiated", "claim rejection"),
    "rejection": ("repudiation", "repudiated"),
    "deny": ("repudiation", "rejection", "denial"),
    "denied": ("repudiation", "rejection", "denial"),
    "switch insurer": ("portability", "migration"),
    "switch": ("portability", "migration"),
    "change insurer": ("portability", "migration"),
    "transfer policy": ("portability", "migration"),
    "port": ("portability", "migration"),
    "waiting": ("waiting period", "pre-existing disease", "PED"),
    "ped": ("pre-existing disease", "36 months"),
    "preexisting": ("pre-existing disease", "PED"),
    "cancel": ("free look period", "policy cancellation", "refund"),
    "cancellation": ("free look period", "policy cancellation", "refund"),
    "return policy": ("free look period", "cancellation"),
    "grace": ("grace period", "continuity"),
    "ncb": ("no claim bonus", "cumulative bonus"),
    "no claim": ("no claim bonus", "cumulative bonus"),
    "bonus": ("cumulative bonus", "no claim bonus"),
    "pmjay": ("PM-JAY", "Ayushman Bharat", "HBP"),
    "pm-jay": ("PM-JAY", "Ayushman Bharat", "HBP"),
    "ayushman": ("PM-JAY", "Ayushman Bharat"),
    "ombudsman": ("Insurance Ombudsman", "grievance redressal"),
    "complaint": ("grievance redressal", "Bima Bharosa", "IGMS", "Insurance Ombudsman"),
    "grievance": ("grievance redressal", "Bima Bharosa", "IGMS"),
    "contest": ("moratorium period", "60 continuous months"),
    "moratorium": ("moratorium period", "60 continuous months", "5 years"),
    "cashless": ("cashless", "pre-authorization", "discharge authorization"),
    "discharge": ("discharge authorization", "cashless", "3 hours"),
    "ayush": ("AYUSH", "Ayurveda", "Homeopathy", "Unani", "Siddha"),
    "ayurveda": ("AYUSH", "Ayurveda"),
    "homeopathy": ("AYUSH", "Homeopathy"),
    "cis": ("Customer Information Sheet",),
    "nominee": ("nomination", "beneficial nominee", "Section 39"),
    "room rent": ("room rent", "proportionate deduction", "ICU"),
    "copay": ("co-payment", "co-pay"),
    "co-pay": ("co-payment",),
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for BM25."""
    return _TOKEN_RE.findall(text.lower())


def expand_query(query: str) -> str:
    """Append domain-specific Indian insurance synonyms without duplicating terms."""
    lowered = query.lower()
    existing_tokens = set(tokenize(lowered))
    additions: list[str] = []

    # Match multi-word phrases first, then single tokens
    for trigger in sorted(INSURANCE_SYNONYMS, key=len, reverse=True):
        if " " in trigger or "-" in trigger:
            matched = trigger in lowered
        else:
            matched = trigger in existing_tokens
        if matched:
            for expansion in INSURANCE_SYNONYMS[trigger]:
                exp_lower = expansion.lower()
                if exp_lower not in lowered and exp_lower not in {a.lower() for a in additions}:
                    additions.append(expansion)

    if not additions:
        return query.strip()
    return f"{query.strip()} {' '.join(additions)}"


class Embedder(Protocol):
    """Protocol for dense text embedding backends."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class Reranker(Protocol):
    """Protocol for query-passage cross-encoder rerankers."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        ...


class SentenceTransformerEmbedder:
    """Lazy-loading dense embedder backed by sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(  # type: ignore[attr-defined]
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in vectors]


class CrossEncoderReranker:
    """Lazy-loading cross-encoder reranker with sigmoid score normalization into [0, 1]."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        model = self._get_model()
        pairs = [(query, passage) for passage in passages]
        raw_scores = model.predict(pairs)  # type: ignore[attr-defined]
        return [self._sigmoid(float(s)) for s in raw_scores]


class _PurePythonBM25:
    """Lightweight Okapi BM25 fallback when rank_bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0

        df: Counter[str] = Counter()
        for counts in self.doc_freqs:
            for term in counts:
                df[term] += 1
        self.idf: dict[str, float] = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        if self.n_docs == 0:
            return []
        scores = [0.0] * self.n_docs
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for idx, counts in enumerate(self.doc_freqs):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                scores[idx] += idf * ((tf * (self.k1 + 1.0)) / denom)
        return scores


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieved chunk with lexical, dense, RRF, and cross-encoder scores."""

    chunk: Chunk
    score: float
    bm25_score: float
    dense_score: float
    rrf_score: float
    rerank_score: float


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine multiple ranked index lists using Reciprocal Rank Fusion (1 / (k + rank))."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank_zero, doc_idx in enumerate(ranked):
            rank = rank_zero + 1
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def _normalize_vector(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return [0.0 for _ in vec]
    return [float(v) / norm for v in vec]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class HybridRetriever:
    """Hybrid BM25 + dense retriever with RRF fusion and top-20 cross-encoder reranking."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidate_top_k: int = 20,
    ) -> None:
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder()
        self.reranker: Reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k
        self.candidate_top_k = candidate_top_k

        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []
        self._bm25: object | None = None

    @staticmethod
    def _chunk_index_text(chunk: Chunk) -> str:
        return f"{chunk.title}\n{chunk.section}\n{chunk.text}"

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Build in-memory BM25 and dense vector indexes from chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.embeddings = []
            self._bm25 = None
            return

        corpus_texts = [self._chunk_index_text(c) for c in self.chunks]
        tokenized_corpus = [tokenize(t) for t in corpus_texts]
        if BM25Okapi is not None:
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = _PurePythonBM25(tokenized_corpus)

        raw_vectors = self.embedder.encode(corpus_texts)
        self.embeddings = [_normalize_vector(v) for v in raw_vectors]

    def save(self, index_dir: Path) -> None:
        """Persist chunks and dense embeddings to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = index_dir / "chunks.json"
        meta_path = index_dir / "index_meta.json"

        chunks_payload = [c.to_dict() for c in self.chunks]
        chunks_path.write_text(json.dumps(chunks_payload, indent=2), encoding="utf-8")

        if np is not None:
            arr = np.asarray(self.embeddings, dtype=np.float32)
            np.save(index_dir / "embeddings.npy", arr)
            emb_format = "npy"
        else:
            (index_dir / "embeddings.json").write_text(
                json.dumps(self.embeddings), encoding="utf-8"
            )
            emb_format = "json"

        meta = {
            "chunks_count": len(self.chunks),
            "embedding_dim": len(self.embeddings[0]) if self.embeddings else 0,
            "embeddings_format": emb_format,
            "rrf_k": self.rrf_k,
            "candidate_top_k": self.candidate_top_k,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        index_dir: Path,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidate_top_k: int = 20,
    ) -> HybridRetriever:
        """Load a persisted hybrid index from disk and rebuild in-memory BM25."""
        chunks_path = index_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Index not found at {chunks_path}. Run `insurance-rag ingest` first."
            )

        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(item) for item in raw_chunks]

        npy_path = index_dir / "embeddings.npy"
        json_emb_path = index_dir / "embeddings.json"

        embeddings: list[list[float]] = []
        if npy_path.exists() and np is not None:
            arr = np.load(npy_path)
            embeddings = [list(map(float, row)) for row in arr.tolist()]
        elif json_emb_path.exists():
            embeddings = [
                list(map(float, row))
                for row in json.loads(json_emb_path.read_text(encoding="utf-8"))
            ]
        else:
            raise FileNotFoundError(
                f"Embeddings file not found in {index_dir}. Run `insurance-rag ingest`."
            )

        retriever = cls(
            embedder=embedder,
            reranker=reranker,
            rrf_k=rrf_k,
            candidate_top_k=candidate_top_k,
        )
        retriever.chunks = chunks
        retriever.embeddings = [_normalize_vector(v) for v in embeddings]
        tokenized_corpus = [tokenize(cls._chunk_index_text(c)) for c in chunks]
        if chunks:
            retriever._bm25 = (
                BM25Okapi(tokenized_corpus)
                if BM25Okapi is not None
                else _PurePythonBM25(tokenized_corpus)
            )
        return retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Run query expansion -> BM25 + dense -> RRF -> cross-encoder rerank over top 20."""
        if not query.strip() or not self.chunks or self._bm25 is None:
            return []

        expanded_query = expand_query(query)
        query_tokens = tokenize(expanded_query)

        bm25_raw = self._bm25.get_scores(query_tokens)  # type: ignore[attr-defined]
        bm25_scores = [float(s) for s in bm25_raw]
        bm25_ranked = sorted(
            range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True
        )

        query_vec = _normalize_vector(self.embedder.encode([expanded_query])[0])
        if np is not None and self.embeddings:
            mat = np.asarray(self.embeddings, dtype=np.float32)
            q_arr = np.asarray(query_vec, dtype=np.float32)
            dense_scores = [float(s) for s in (mat @ q_arr).tolist()]
        else:
            dense_scores = [_dot(vec, query_vec) for vec in self.embeddings]

        dense_ranked = sorted(
            range(len(self.chunks)), key=lambda i: dense_scores[i], reverse=True
        )

        rrf_results = reciprocal_rank_fusion(
            [bm25_ranked, dense_ranked], k=self.rrf_k
        )
        pool_size = min(candidate_top_k or self.candidate_top_k, len(rrf_results))
        candidates = rrf_results[:pool_size]

        passages = [self._chunk_index_text(self.chunks[idx]) for idx, _ in candidates]
        # Rerank against the original user query so exact user intent governs relevance
        rerank_scores = self.reranker.score(query.strip(), passages)

        combined: list[RetrievedChunk] = []
        for (idx, rrf_score), rerank_score in zip(candidates, rerank_scores):
            combined.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(rerank_score),
                    bm25_score=bm25_scores[idx],
                    dense_score=dense_scores[idx],
                    rrf_score=float(rrf_score),
                    rerank_score=float(rerank_score),
                )
            )

        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[:top_k]
