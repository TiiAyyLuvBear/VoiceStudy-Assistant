"""Open-set application user identification with cosine centroids."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
import yaml

from src.audio.preprocessing import preprocess_audio
from src.speaker.embedding import ECAPAEmbeddingExtractor


CONFIG_PATH = Path("config.yaml")
PROTOCOL = "APPLICATION_SID"
_USER_ID_RE = __import__("re").compile(r"^user_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ApplicationIdentificationResult(TypedDict):
    protocol: str
    candidate_user_id: str | None
    cosine_similarity: float | None
    unknown_threshold: float | None
    status: str
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
    direct_threshold = speaker.get("application_sid_threshold")
    threshold_source: Path | float
    if direct_threshold is not None:
        threshold_source = float(direct_threshold)
    else:
        threshold_source = Path(
            speaker.get(
                "application_sid_threshold_path",
                "reports/results/evaluation/finetuned/open_set_metrics.json",
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
            raise ValueError("Application SID threshold is missing or invalid")
        return float(path)
    if not path.is_file():
        raise FileNotFoundError(f"Application SID threshold does not exist: {path}")
    value = threshold_from_metrics_document(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if value is None or not np.isfinite(value):
        raise ValueError("Application SID threshold is missing or invalid")
    return float(value)


def _load_gallery(directory: Path) -> tuple[list[str], np.ndarray]:
    paths = sorted(directory.glob("*.npy")) if directory.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"No application user centroids found in: {directory}")
    user_ids: list[str] = []
    vectors: list[np.ndarray] = []
    expected_dimension: int | None = None
    for path in paths:
        vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"Invalid application centroid: {path}")
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float32).eps:
            raise ValueError(f"Zero application centroid: {path}")
        vector = vector / norm
        if expected_dimension is None:
            expected_dimension = vector.size
        elif vector.size != expected_dimension:
            raise ValueError("Application centroid dimensions are inconsistent")
        if not _USER_ID_RE.fullmatch(path.stem):
            continue
        user_ids.append(path.stem)
        vectors.append(vector)
    if not vectors:
        raise FileNotFoundError(f"No application user centroids found in: {directory}")
    return user_ids, np.stack(vectors)


def _failure(
    started_at: float,
    error: Exception | str,
    threshold: float | None = None,
) -> ApplicationIdentificationResult:
    return {
        "protocol": PROTOCOL,
        "candidate_user_id": None,
        "cosine_similarity": None,
        "unknown_threshold": threshold,
        "status": "ERROR",
        "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        "success": False,
        "error": str(error),
    }


def identify_application_user(audio_path: str) -> ApplicationIdentificationResult:
    """Identify an application user by cosine similarity or return UNKNOWN."""

    started_at = time.perf_counter()
    threshold: float | None = None
    try:
        path = resolve_audio_path(audio_path)

        # This module intentionally uses no SVM model.
        audio, sample_rate = preprocess_audio(str(path))
        centroid_dir, threshold_path = _load_settings()
        threshold = _load_threshold(threshold_path)
        user_ids, gallery = _load_gallery(centroid_dir)
        embedding, dimension, _ = _get_extractor().extract(
            audio,
            sample_rate=sample_rate,
        )
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if dimension != gallery.shape[1] or query.size != gallery.shape[1]:
            raise ValueError(
                f"Query dimension {query.size} does not match gallery "
                f"dimension {gallery.shape[1]}"
            )
        norm = float(np.linalg.norm(query))
        if not np.isfinite(query).all() or norm <= np.finfo(np.float32).eps:
            raise ValueError("Query embedding is zero or non-finite")
        similarities = gallery @ (query / norm)
        winner = int(np.argmax(similarities))
        similarity = float(similarities[winner])
        known = similarity >= threshold
        return {
            "protocol": PROTOCOL,
            "candidate_user_id": user_ids[winner] if known else None,
            "cosine_similarity": similarity,
            "unknown_threshold": threshold,
            "status": "KNOWN" if known else "UNKNOWN",
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            "success": True,
            "error": None,
        }
    except Exception as error:  # Stable application boundary.
        return _failure(started_at, error, threshold)
