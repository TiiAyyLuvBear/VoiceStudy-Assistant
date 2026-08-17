"""Application speaker verification against a claimed user centroid."""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
import yaml

from src.audio.preprocessing import preprocess_audio
from src.speaker.embedding import ECAPAEmbeddingExtractor, get_embedding_extractor
from src.utils.config import threshold_from_metrics_document


CONFIG_PATH = Path("config.yaml")
_USER_ID_RE = re.compile(r"^user_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class VerificationResult(TypedDict):
    candidate_user_id: str
    similarity: float | None
    verification_threshold: float | None
    verified: bool
    latency_ms: float
    success: bool
    error: str | None


@lru_cache(maxsize=1)
def _get_extractor() -> ECAPAEmbeddingExtractor:
    return get_embedding_extractor()


def _load_settings(config_path: Path = CONFIG_PATH) -> tuple[Path, Path | float]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    speaker = config.get("speaker", {})
    centroid_dir = Path(
        speaker.get(
            "application_centroid_dir",
            "models/application/user_embeddings",
        )
    )
    direct_threshold = speaker.get("application_verification_threshold")
    threshold_source: Path | float
    if direct_threshold is not None:
        threshold_source = float(direct_threshold)
    else:
        threshold_source = Path(
            speaker.get(
                "application_verification_threshold_path",
                "reports/results/evaluation/finetuned/verification_metrics.json",
            )
        )
    if not centroid_dir.is_absolute():
        centroid_dir = config_path.resolve().parent / centroid_dir
    if isinstance(threshold_source, Path) and not threshold_source.is_absolute():
        threshold_source = config_path.resolve().parent / threshold_source
    return centroid_dir, threshold_source


def _load_threshold(path: Path | float) -> float:
    if isinstance(path, (int, float)):
        if not np.isfinite(path):
            raise ValueError("Verification threshold is missing or invalid")
        return float(path)
    if not path.is_file():
        raise FileNotFoundError(f"Verification threshold does not exist: {path}")
    value = threshold_from_metrics_document(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if value is None or not np.isfinite(value):
        raise ValueError("Verification threshold is missing or invalid")
    return float(value)


def _load_candidate_centroid(directory: Path, candidate_user_id: str) -> np.ndarray:
    if not _USER_ID_RE.fullmatch(candidate_user_id):
        raise ValueError("candidate_user_id must start with 'user_'")
    path = directory / f"{candidate_user_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"Enrollment centroid does not exist for application user: "
            f"{candidate_user_id}"
        )
    vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"Invalid enrollment centroid for {candidate_user_id}")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError(f"Zero enrollment centroid for {candidate_user_id}")
    return vector / norm


def _failure(
    started_at: float,
    candidate_user_id: str,
    error: Exception | str,
    threshold: float | None = None,
) -> VerificationResult:
    return {
        "candidate_user_id": candidate_user_id,
        "similarity": None,
        "verification_threshold": threshold,
        "verified": False,
        "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        "success": False,
        "error": str(error),
    }


def verify_speaker(audio_path: str, candidate_user_id: str) -> VerificationResult:
    """Verify audio against the named application user's enrollment centroid."""

    started_at = time.perf_counter()
    threshold: float | None = None
    try:
        if not isinstance(candidate_user_id, str):
            raise TypeError("candidate_user_id must be a string")
        path = resolve_audio_path(audio_path)

        # Initialize Librosa before SpeechBrain optional lazy imports.
        audio, sample_rate = preprocess_audio(str(path))
        centroid_dir, threshold_path = _load_settings()
        threshold = _load_threshold(threshold_path)
        centroid = _load_candidate_centroid(centroid_dir, candidate_user_id)
        embedding, dimension, _ = _get_extractor().extract(
            audio,
            sample_rate=sample_rate,
        )
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if dimension != centroid.size or query.size != centroid.size:
            raise ValueError(
                f"Query dimension {query.size} does not match candidate "
                f"centroid dimension {centroid.size}"
            )
        norm = float(np.linalg.norm(query))
        if not np.isfinite(query).all() or norm <= np.finfo(np.float32).eps:
            raise ValueError("Query embedding is zero or non-finite")
        similarity = float((query / norm) @ centroid)
        return {
            "candidate_user_id": candidate_user_id,
            "similarity": similarity,
            "verification_threshold": threshold,
            "verified": similarity >= threshold,
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            "success": True,
            "error": None,
        }
    except Exception as error:  # Stable application boundary.
        user_id = candidate_user_id if isinstance(candidate_user_id, str) else ""
        return _failure(started_at, user_id, error, threshold)
