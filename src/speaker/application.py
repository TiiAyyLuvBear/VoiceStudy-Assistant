"""Application speaker enrollment, identification, and verification APIs."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.audio.preprocessing import preprocess_audio
from src.database.user_repository import create_user, get_user, list_users, update_embedding_path
from src.speaker.embedding import EmbeddingError, extract_embedding
from src.utils.config import load_yaml_mapping, resolve_path

AudioLoader = Callable[[str | Path], tuple[np.ndarray, int]]
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_ENROLLMENT_AUDIO_COUNT = 5


def _result(**values: object) -> dict:
    result = {
        "success": False,
        "user_id": None,
        "candidate_user_id": None,
        "embedding_count": None,
        "centroid_path": None,
        "similarity": None,
        "identified": False,
        "verified": None,
        "error": None,
    }
    result.update(values)
    return result


def _speaker_config(config_path: str | Path) -> tuple[dict, Path]:
    config, root = load_yaml_mapping(config_path)
    return config.get("speaker", {}), root


def _centroid_dir(config_path: str | Path) -> Path:
    speaker, root = _speaker_config(config_path)
    return resolve_path(speaker.get("application_centroid_dir", "models/user_embeddings"), root)


def _threshold(
    config_path: str | Path,
    setting: str,
    explicit_value: float | None,
) -> float | None:
    if explicit_value is not None:
        return float(explicit_value)
    speaker, root = _speaker_config(config_path)
    threshold_path = speaker.get(setting)
    if not threshold_path:
        return None
    try:
        document, _ = load_yaml_mapping(resolve_path(threshold_path, root))
        value = document.get("threshold")
        return float(value) if value is not None else None
    except (OSError, TypeError, ValueError):
        return None


def _embedding_for_audio(
    audio_path: str | Path,
    *,
    extractor: object | None,
    audio_loader: AudioLoader,
) -> np.ndarray:
    audio, sample_rate = audio_loader(audio_path)
    embedding, _, _ = extract_embedding(audio, sample_rate=sample_rate, extractor=extractor)
    return _normalise(embedding)


def _normalise(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if values.size == 0 or not np.isfinite(values).all() or norm == 0.0:
        raise EmbeddingError("Speaker embedding is empty, non-finite, or zero-norm")
    return values / norm


def _load_centroid(path: Path) -> np.ndarray:
    return _normalise(np.load(path, allow_pickle=False))


def enroll_user(
    user_id: str,
    name: str,
    audio_paths: Sequence[str | Path],
    *,
    database_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
    extractor: object | None = None,
    audio_loader: AudioLoader = preprocess_audio,
) -> dict:
    """Create/update enrollment from exactly five audio recordings."""
    if not isinstance(user_id, str) or not _USER_ID_PATTERN.fullmatch(user_id):
        return _result(error="INVALID_USER_ID")
    if not isinstance(name, str) or not name.strip():
        return _result(user_id=user_id, error="INVALID_NAME")
    if isinstance(audio_paths, (str, bytes)) or len(audio_paths) != _ENROLLMENT_AUDIO_COUNT:
        return _result(user_id=user_id, error="INVALID_ENROLLMENT_AUDIO_COUNT")

    try:
        embeddings = [
            _embedding_for_audio(path, extractor=extractor, audio_loader=audio_loader)
            for path in audio_paths
        ]
        centroid = _normalise(np.mean(embeddings, axis=0))
    except (OSError, ValueError, EmbeddingError):
        return _result(user_id=user_id, error="INVALID_AUDIO")

    try:
        centroid_dir = _centroid_dir(config_path)
        centroid_dir.mkdir(parents=True, exist_ok=True)
        centroid_path = centroid_dir / f"{user_id}.npy"
        np.save(centroid_path, centroid)
        if get_user(user_id, database_path):
            update_embedding_path(user_id, str(centroid_path), database_path)
        else:
            create_user(user_id, name.strip(), str(centroid_path), database_path)
    except (OSError, sqlite3.Error):
        return _result(user_id=user_id, error="CENTROID_WRITE_FAILED")

    return _result(
        success=True,
        user_id=user_id,
        embedding_count=len(embeddings),
        centroid_path=str(centroid_path),
        error=None,
    )


def identify_application_user(
    audio_path: str | Path,
    *,
    database_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
    extractor: object | None = None,
    audio_loader: AudioLoader = preprocess_audio,
    identification_threshold: float | None = None,
) -> dict:
    """Identify highest-scoring enrolled application user from audio."""
    users = list_users(database_path)
    if not users:
        return _result(error="NO_ENROLLMENT")
    try:
        query = _embedding_for_audio(audio_path, extractor=extractor, audio_loader=audio_loader)
    except (OSError, ValueError, EmbeddingError):
        return _result(error="INVALID_AUDIO")

    scores: list[tuple[str, float, Path]] = []
    missing_centroid = False
    for user in users:
        path_value = user.get("embedding_path")
        if not path_value:
            missing_centroid = True
            continue
        path = Path(path_value)
        try:
            centroid = _load_centroid(path)
            scores.append((user["user_id"], float(np.dot(query, centroid)), path))
        except (OSError, ValueError, EmbeddingError):
            missing_centroid = True

    if not scores:
        return _result(error="CENTROID_NOT_FOUND" if missing_centroid else "NO_ENROLLMENT")
    threshold = _threshold(config_path, "application_sid_threshold_path", identification_threshold)
    if threshold is None:
        return _result(error="THRESHOLD_NOT_CONFIGURED")
    user_id, similarity, centroid_path = max(scores, key=lambda item: item[1])
    identified = similarity >= threshold
    return _result(
        success=True,
        candidate_user_id=user_id if identified else None,
        centroid_path=str(centroid_path),
        similarity=similarity,
        identified=identified,
        error=None,
    )


def verify_speaker(
    audio_path: str | Path,
    candidate_user_id: str,
    *,
    database_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
    extractor: object | None = None,
    audio_loader: AudioLoader = preprocess_audio,
    verification_threshold: float | None = None,
) -> dict:
    """Verify audio against enrolled centroid for one application user."""
    user = get_user(candidate_user_id, database_path)
    if not user:
        return _result(candidate_user_id=candidate_user_id, error="NO_ENROLLMENT")
    path_value = user.get("embedding_path")
    if not path_value:
        return _result(candidate_user_id=candidate_user_id, error="CENTROID_NOT_FOUND")
    try:
        query = _embedding_for_audio(audio_path, extractor=extractor, audio_loader=audio_loader)
    except (OSError, ValueError, EmbeddingError):
        return _result(candidate_user_id=candidate_user_id, error="INVALID_AUDIO")
    try:
        centroid_path = Path(path_value)
        similarity = float(np.dot(query, _load_centroid(centroid_path)))
    except (OSError, ValueError, EmbeddingError):
        return _result(candidate_user_id=candidate_user_id, error="CENTROID_NOT_FOUND")
    threshold = _threshold(config_path, "application_verification_threshold_path", verification_threshold)
    if threshold is None:
        return _result(candidate_user_id=candidate_user_id, error="THRESHOLD_NOT_CONFIGURED")
    return _result(
        success=True,
        candidate_user_id=candidate_user_id,
        centroid_path=str(centroid_path),
        similarity=similarity,
        verified=similarity >= threshold,
        error=None,
    )
