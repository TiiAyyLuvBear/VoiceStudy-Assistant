"""Enrollment centroids for application users."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np

from src.audio.preprocessing import preprocess_audio
from src.audio.source import resolve_audio_path
from src.speaker.embedding import ECAPAEmbeddingExtractor


APPLICATION_EMBEDDING_DIR = Path("models/application/user_embeddings")
REQUIRED_AUDIO_COUNT = 5
_USER_ID_RE = re.compile(r"^user_[A-Za-z0-9][A-Za-z0-9_-]{0,58}$")


class EnrollmentResult(TypedDict):
    user_id: str
    audio_count: int
    embedding_path: str | None
    embedding_dim: int | None
    l2_norm: float | None
    success: bool
    error: str | None


@lru_cache(maxsize=1)
def _get_extractor() -> ECAPAEmbeddingExtractor:
    return ECAPAEmbeddingExtractor.from_config()


def _failure(user_id: str, audio_count: int, error: Exception | str) -> EnrollmentResult:
    return {
        "user_id": user_id,
        "audio_count": audio_count,
        "embedding_path": None,
        "embedding_dim": None,
        "l2_norm": None,
        "success": False,
        "error": str(error),
    }


def enroll_user(user_id: str, audio_paths: list[str]) -> EnrollmentResult:
    """Create an L2-normalized centroid from exactly five valid audio files."""

    try:
        if not isinstance(user_id, str) or not _USER_ID_RE.fullmatch(user_id):
            raise ValueError(
                "user_id must match user_<ascii letters/digits/underscore/hyphen>"
            )
        if not isinstance(audio_paths, list):
            raise TypeError("audio_paths must be a list of strings")
        if len(audio_paths) != REQUIRED_AUDIO_COUNT:
            raise ValueError(f"Exactly {REQUIRED_AUDIO_COUNT} audio files are required")
        if any(not isinstance(value, str) or not value.strip() for value in audio_paths):
            raise ValueError("Every audio path must be a non-empty string")

        paths = [resolve_audio_path(value) for value in audio_paths]
        if len({str(path.resolve()) for path in paths}) != REQUIRED_AUDIO_COUNT:
            raise ValueError("The five enrollment audio files must be distinct")

        embeddings: list[np.ndarray] = []
        expected_dimension: int | None = None
        for path in paths:
            # Initialize Librosa before SpeechBrain optional lazy imports.
            audio, sample_rate = preprocess_audio(str(path))
            embedding, dimension, _ = _get_extractor().extract(
                audio,
                sample_rate=sample_rate,
            )
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if dimension != vector.size or not np.isfinite(vector).all():
                raise ValueError(f"Invalid embedding extracted from {path}")
            if expected_dimension is None:
                expected_dimension = dimension
            elif dimension != expected_dimension:
                raise ValueError("Enrollment embedding dimensions are inconsistent")
            embeddings.append(vector)

        centroid = np.mean(np.stack(embeddings), axis=0)
        norm = float(np.linalg.norm(centroid))
        if not np.isfinite(norm) or norm <= np.finfo(np.float32).eps:
            raise ValueError("Enrollment centroid is zero or non-finite")
        centroid = np.asarray(centroid / norm, dtype=np.float32)
        final_norm = float(np.linalg.norm(centroid))

        output_path = APPLICATION_EMBEDDING_DIR / f"{user_id}.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, centroid, allow_pickle=False)
        temporary.replace(output_path)
        return {
            "user_id": user_id,
            "audio_count": len(paths),
            "embedding_path": output_path.as_posix(),
            "embedding_dim": int(centroid.size),
            "l2_norm": final_norm,
            "success": True,
            "error": None,
        }
    except Exception as error:  # Stable application boundary.
        count = len(audio_paths) if isinstance(audio_paths, list) else 0
        return _failure(user_id if isinstance(user_id, str) else "", count, error)
