"""Environment-based configuration for insurance-rag."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in bare stdlib runs
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings loaded from environment variables."""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-4-hydrogen"
    gemini_judge_model: str = "gemini-4-hydrogen"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    raw_data_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")
    golden_path: Path = Path("data/golden/golden.jsonl")
    reports_dir: Path = Path("reports")

    chunk_size: int = 900
    chunk_overlap: int = 150
    rrf_k: int = 60
    candidate_top_k: int = 20
    final_top_k: int = 5
    abstention_threshold: float = 0.25

    api_host: str = "0.0.0.0"
    api_port: int = 8000


def load_settings(base_dir: Path | None = None, load_env_file: bool = True) -> Settings:
    """Load Settings from environment variables and optional .env file."""
    if load_env_file:
        env_path = (base_dir / ".env") if base_dir else Path(".env")
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            load_dotenv(override=False)

    root = base_dir or Path.cwd()

    def _resolve_path(env_key: str, default_rel: str) -> Path:
        val = os.environ.get(env_key, default_rel).strip() or default_rel
        p = Path(val)
        return p if p.is_absolute() else (root / p)

    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen").strip() or "gemini-4-hydrogen",
        gemini_judge_model=os.environ.get("GEMINI_JUDGE_MODEL", "gemini-4-hydrogen").strip() or "gemini-4-hydrogen",
        embedding_model=(
            os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
            or "sentence-transformers/all-MiniLM-L6-v2"
        ),
        reranker_model=(
            os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        raw_data_dir=_resolve_path("RAW_DATA_DIR", "data/raw"),
        index_dir=_resolve_path("INDEX_DIR", "data/index"),
        golden_path=_resolve_path("GOLDEN_PATH", "data/golden/golden.jsonl"),
        reports_dir=_resolve_path("REPORTS_DIR", "reports"),
        chunk_size=_get_int("CHUNK_SIZE", 900),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        rrf_k=_get_int("RRF_K", 60),
        candidate_top_k=_get_int("CANDIDATE_TOP_K", 20),
        final_top_k=_get_int("FINAL_TOP_K", 5),
        abstention_threshold=_get_float("ABSTENTION_THRESHOLD", 0.25),
        api_host=os.environ.get("API_HOST", "0.0.0.0").strip() or "0.0.0.0",
        api_port=_get_int("API_PORT", 8000),
    )
