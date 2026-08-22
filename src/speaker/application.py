"""Application speaker enrollment, identification, and verification APIs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.audio.preprocessing import SpeakerPreprocessResult, preprocess_audio, preprocess_audio_with_metrics
from src.database.user_repository import (
    create_user,
    delete_user,
    get_user,
    list_users,
    update_user_enrollment,
)
from src.security.secret_phrase import hash_secret_phrase, validate_secret_phrase
from src.speaker.embedding import EmbeddingError, extract_embedding
from src.speaker.enrollment_quality import (
    analyze_audio_quality,
    centroid_similarities,
    embedding_consistency,
    validate_enrollment_prompts,
)
from src.utils.config import (
    load_yaml_mapping,
    resolve_path,
    threshold_from_metrics_document,
)

AudioLoader = Callable[[str | Path], object]
_USER_ID_PATTERN = re.compile(r"^user_[A-Za-z0-9][A-Za-z0-9_-]{0,58}$")
_ENROLLMENT_AUDIO_COUNT = 5
_MIN_ENROLLMENT_AUDIO_COUNT = 3
_MAX_ENROLLMENT_AUDIO_COUNT = 10


class CentroidModelMismatchError(ValueError):
    """Raised when a centroid belongs to another embedding model version."""


def _result(**values: object) -> dict:
    result = {
        "protocol": None,
        "success": False,
        "user_id": None,
        "candidate_user_id": None,
        "embedding_count": None,
        "audio_count": None,
        "embedding_path": None,
        "centroid_path": None,
        "embedding_dim": None,
        "l2_norm": None,
        "cosine_similarity": None,
        "similarity": None,
        "identified": None,
        "verified": None,
        "latency_ms": 0.0,
        "file_results": [],
        "quality": None,
        "embedding_consistency": None,
        "secret_phrase_configured": None,
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


def _resolve_centroid_path(path_value: str | Path, config_path: str | Path) -> Path:
    """Resolve database-stored centroid paths against config directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    _, root = _speaker_config(config_path)
    return resolve_path(path, root)


def _portable_centroid_path(path: str | Path, config_path: str | Path) -> str:
    """Return centroid path relative to config directory when possible."""
    centroid = Path(path).resolve()
    _, root = _speaker_config(config_path)
    try:
        return centroid.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(centroid)


def _model_version(config_path: str | Path) -> str:
    speaker, _ = _speaker_config(config_path)
    return str(speaker.get("model_version", "unknown")).strip() or "unknown"


def _enrollment_audio_count(config_path: str | Path) -> int:
    speaker, _ = _speaker_config(config_path)
    value = int(speaker.get("enrollment_audio_count", _ENROLLMENT_AUDIO_COUNT))
    if value <= 0:
        raise ValueError("speaker.enrollment_audio_count must be positive")
    return value


def _enrollment_audio_limits(config_path: str | Path) -> tuple[int, int]:
    speaker, _ = _speaker_config(config_path)
    default_count = int(speaker.get("enrollment_audio_count", _ENROLLMENT_AUDIO_COUNT))
    minimum = int(speaker.get("min_enrollment_audio_count", min(_MIN_ENROLLMENT_AUDIO_COUNT, default_count)))
    maximum = int(speaker.get("max_enrollment_audio_count", max(_MAX_ENROLLMENT_AUDIO_COUNT, default_count)))
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Invalid enrollment audio count limits")
    return minimum, maximum


def _enrollment_quality_settings(config_path: str | Path) -> dict:
    speaker, _ = _speaker_config(config_path)
    settings = speaker.get("enrollment_quality", {})
    return dict(settings) if isinstance(settings, dict) else {}


def _threshold(
    config_path: str | Path,
    setting: str,
    explicit_value: float | None,
) -> float | None:
    if explicit_value is not None:
        return float(explicit_value)
    speaker, root = _speaker_config(config_path)
    direct_setting = setting.removesuffix("_path")
    direct_value = speaker.get(direct_setting)
    if direct_value is not None:
        try:
            value = float(direct_value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None
    threshold_path = speaker.get(setting)
    if not threshold_path:
        return None
    try:
        document, _ = load_yaml_mapping(resolve_path(threshold_path, root))
        return threshold_from_metrics_document(document)
    except (OSError, TypeError, ValueError):
        return None


def _embedding_for_audio(
    audio_path: str | Path,
    *,
    extractor: object | None,
    audio_loader: AudioLoader,
) -> np.ndarray:
    loaded = audio_loader(audio_path)
    audio, sample_rate, _ = _coerce_loaded_audio(loaded)
    embedding, _, _ = extract_embedding(audio, sample_rate=sample_rate, extractor=extractor)
    return _normalise(embedding)


def _coerce_loaded_audio(loaded: object) -> tuple[np.ndarray, int, dict]:
    if isinstance(loaded, SpeakerPreprocessResult):
        return loaded.audio, loaded.sample_rate, dict(loaded.metrics)
    if not isinstance(loaded, tuple):
        raise ValueError("audio_loader must return (audio, sample_rate)")
    if len(loaded) == 3:
        audio, sample_rate, metrics = loaded
        return np.asarray(audio, dtype=np.float32), int(sample_rate), dict(metrics or {})
    if len(loaded) == 2:
        audio, sample_rate = loaded
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = float(values.size / int(sample_rate)) if int(sample_rate) > 0 else 0.0
        return values, int(sample_rate), {
            "duration_seconds": duration,
            "speech_duration_seconds": duration,
            "speech_ratio": 1.0 if duration > 0.0 else 0.0,
            "sample_rate": int(sample_rate),
            "num_channels": 1,
        }
    raise ValueError("audio_loader must return (audio, sample_rate)")


def _load_enrollment_audio(
    path: str | Path,
    *,
    audio_loader: AudioLoader,
    quality_settings: dict,
) -> tuple[np.ndarray, int, dict]:
    if audio_loader is preprocess_audio:
        result = preprocess_audio_with_metrics(
            path,
            vad_peak_ratio=float(quality_settings.get("silence_peak_ratio", 0.03)),
        )
        return result.audio, result.sample_rate, dict(result.metrics)
    return _coerce_loaded_audio(audio_loader(path))


def _normalise(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if values.size == 0 or not np.isfinite(values).all() or norm == 0.0:
        raise EmbeddingError("Speaker embedding is empty, non-finite, or zero-norm")
    return values / norm


def _centroid_metadata_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _write_centroid_metadata(
    path: Path,
    *,
    model_version: str,
    embedding_dimension: int,
    enrollment_audio_count: int,
) -> None:
    document = {
        "model_version": model_version,
        "embedding_dimension": int(embedding_dimension),
        "enrollment_audio_count": int(enrollment_audio_count),
    }
    destination = _centroid_metadata_path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _load_centroid(path: Path, expected_model_version: str) -> np.ndarray:
    metadata_path = _centroid_metadata_path(path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CentroidModelMismatchError("CENTROID_MODEL_MISMATCH") from error
    if metadata.get("model_version") != expected_model_version:
        raise CentroidModelMismatchError("CENTROID_MODEL_MISMATCH")
    vector = _normalise(np.load(path, allow_pickle=False))
    if metadata.get("embedding_dimension") != vector.size:
        raise CentroidModelMismatchError("CENTROID_MODEL_MISMATCH")
    return vector


def enroll_user(
    user_id: str,
    name: str,
    audio_paths: Sequence[str | Path],
    *,
    secret_phrase: str | None = None,
    enrollment_prompts: Sequence[str] | None = None,
    database_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
    extractor: object | None = None,
    audio_loader: AudioLoader = preprocess_audio,
) -> dict:
    """Create/update enrollment from guided audio recordings and a secret phrase."""
    started_at = time.perf_counter()
    if not isinstance(user_id, str) or not _USER_ID_PATTERN.fullmatch(user_id):
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="INVALID_USER_ID",
        )
    if not isinstance(name, str) or not name.strip():
        return _result(protocol="APPLICATION_ENROLLMENT", user_id=user_id, error="INVALID_NAME")
    valid_secret, secret_error = validate_secret_phrase(secret_phrase or "")
    if not valid_secret:
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            user_id=user_id,
            error=secret_error or "INVALID_SECRET_PHRASE",
        )
    prompts_valid, prompts_error = validate_enrollment_prompts(enrollment_prompts)
    if not prompts_valid:
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            user_id=user_id,
            error=prompts_error,
        )
    try:
        min_audio_count, max_audio_count = _enrollment_audio_limits(config_path)
        model_version = _model_version(config_path)
        quality_settings = _enrollment_quality_settings(config_path)
    except (OSError, TypeError, ValueError):
        return _result(protocol="APPLICATION_ENROLLMENT", user_id=user_id, error="INVALID_SPEAKER_CONFIG")
    if isinstance(audio_paths, (str, bytes)) or not (min_audio_count <= len(audio_paths) <= max_audio_count):
        return _result(protocol="APPLICATION_ENROLLMENT", user_id=user_id, error="INVALID_ENROLLMENT_AUDIO_COUNT")
    if any(not isinstance(path, (str, Path)) for path in audio_paths):
        return _result(protocol="APPLICATION_ENROLLMENT", user_id=user_id, error="INVALID_AUDIO")
    resolved_audio_paths = [Path(path).resolve() for path in audio_paths]
    if len(set(resolved_audio_paths)) != len(resolved_audio_paths):
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            user_id=user_id,
            error="DUPLICATE_ENROLLMENT_AUDIO",
        )
    content_hashes: list[str] = []
    try:
        for path in resolved_audio_paths:
            content_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    except OSError as exc:
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            user_id=user_id,
            audio_count=len(audio_paths),
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error=f"INVALID_AUDIO: {exc}",
        )
    if len(set(content_hashes)) != len(content_hashes):
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            user_id=user_id,
            audio_count=len(audio_paths),
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="DUPLICATE_ENROLLMENT_AUDIO",
        )

    embeddings: list[np.ndarray] = []
    embedding_file_indexes: list[int] = []
    file_results: list[dict] = []
    quality_results: list[dict] = []
    total_speech_duration = 0.0
    min_accepted_samples = int(quality_settings.get("min_accepted_enrollment_samples", 3))
    min_samples_for_outlier = int(quality_settings.get("min_samples_for_outlier_detection", 3))
    min_centroid_similarity = float(quality_settings.get("min_centroid_similarity", 0.45))
    for index, path in enumerate(audio_paths, start=1):
        try:
            audio, sample_rate, preprocess_metrics = _load_enrollment_audio(
                path,
                audio_loader=audio_loader,
                quality_settings=quality_settings,
            )
            quality = analyze_audio_quality(
                audio,
                sample_rate,
                quality_settings,
                metrics=preprocess_metrics,
            )
            quality_results.append(quality)
            total_speech_duration += float(quality.get("metrics", {}).get("speech_duration_seconds") or 0.0)
            if not quality["valid"]:
                file_results.append(
                    {
                        "sample_id": f"sample_{index:02d}",
                        "audio_path": str(path),
                        "valid": False,
                        "accepted": False,
                        "error": "AUDIO_QUALITY_FAILED",
                        "message_vi": quality.get("message_vi"),
                        "rejection_reasons": quality.get("issues", []),
                        "quality": quality,
                    }
                )
                continue
            embedding, _, _ = extract_embedding(
                audio,
                sample_rate=sample_rate,
                extractor=extractor,
            )
            embedding = _normalise(embedding)
        except (OSError, ValueError, EmbeddingError) as exc:
            file_results.append(
                {
                    "sample_id": f"sample_{index:02d}",
                    "audio_path": str(path),
                    "valid": False,
                    "accepted": False,
                    "error": "INVALID_EMBEDDING",
                    "message_vi": "Không tạo được speaker embedding từ mẫu này. Hãy thu lại mẫu rõ hơn.",
                    "rejection_reasons": ["invalid_embedding"],
                    "details": str(exc),
                }
            )
            continue
        embeddings.append(embedding)
        embedding_file_indexes.append(len(file_results))
        file_results.append(
            {
                "sample_id": f"sample_{index:02d}",
                "audio_path": str(path),
                "valid": True,
                "accepted": True,
                "error": None,
                "rejection_reasons": [],
                "quality": quality,
            }
        )
    if len(embeddings) < min_accepted_samples:
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            user_id=user_id,
            audio_count=len(audio_paths),
            accepted_samples=len(embeddings),
            rejected_samples=len(audio_paths) - len(embeddings),
            requires_more_samples=True,
            total_speech_duration_sec=total_speech_duration,
            file_results=file_results,
            quality=quality_results,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="INSUFFICIENT_ACCEPTED_SAMPLES",
        )
    consistency = embedding_consistency(embeddings, quality_settings)
    accepted_embedding_flags = [True] * len(embeddings)
    if len(embeddings) >= min_samples_for_outlier:
        _, sample_centroid_similarities = centroid_similarities(embeddings)
        consistency["centroid_similarities"] = sample_centroid_similarities
        for local_index, score in enumerate(sample_centroid_similarities):
            file_index = embedding_file_indexes[local_index]
            file_results[file_index]["centroid_similarity"] = score
            if score < min_centroid_similarity:
                accepted_embedding_flags[local_index] = False
                file_results[file_index]["valid"] = False
                file_results[file_index]["accepted"] = False
                file_results[file_index]["error"] = "VOICE_INCONSISTENT_WITH_OTHER_SAMPLES"
                file_results[file_index]["message_vi"] = (
                    "Mẫu voice lệch nhiều so với các mẫu còn lại. Hãy nghe lại và thu lại mẫu này."
                )
                file_results[file_index]["rejection_reasons"].append("embedding_outlier")
    accepted_embeddings = [
        embedding
        for embedding, accepted in zip(embeddings, accepted_embedding_flags)
        if accepted
    ]
    if len(accepted_embeddings) < min_accepted_samples:
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            user_id=user_id,
            audio_count=len(audio_paths),
            accepted_samples=len(accepted_embeddings),
            rejected_samples=len(audio_paths) - len(accepted_embeddings),
            requires_more_samples=True,
            total_speech_duration_sec=total_speech_duration,
            file_results=file_results,
            quality=quality_results,
            embedding_consistency=consistency,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="INSUFFICIENT_ACCEPTED_SAMPLES",
        )
    accepted_consistency = embedding_consistency(accepted_embeddings, quality_settings)
    consistency["accepted"] = accepted_consistency
    if not accepted_consistency["valid"]:
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            user_id=user_id,
            audio_count=len(audio_paths),
            accepted_samples=len(accepted_embeddings),
            rejected_samples=len(audio_paths) - len(accepted_embeddings),
            requires_more_samples=False,
            total_speech_duration_sec=total_speech_duration,
            file_results=file_results,
            quality=quality_results,
            embedding_consistency=consistency,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="EMBEDDING_CONSISTENCY_FAILED",
        )
    centroid = _normalise(np.mean(accepted_embeddings, axis=0))

    centroid_path: Path | None = None
    try:
        centroid_dir = _centroid_dir(config_path)
        centroid_dir.mkdir(parents=True, exist_ok=True)
        centroid_path = centroid_dir / f"{user_id}.npy"
        np.save(centroid_path, centroid)
        _write_centroid_metadata(
            centroid_path,
            model_version=model_version,
            embedding_dimension=centroid.size,
            enrollment_audio_count=len(accepted_embeddings),
        )
        stored_centroid_path = _portable_centroid_path(centroid_path, config_path)
        secret_digest, secret_salt = hash_secret_phrase(secret_phrase or "")
        if get_user(user_id, database_path):
            update_user_enrollment(
                user_id,
                name=name.strip(),
                embedding_path=stored_centroid_path,
                secret_phrase_hash=secret_digest,
                secret_phrase_salt=secret_salt,
                database_path=database_path,
            )
        else:
            create_user(
                user_id,
                name.strip(),
                stored_centroid_path,
                database_path,
                secret_phrase_hash=secret_digest,
                secret_phrase_salt=secret_salt,
            )
    except (OSError, sqlite3.Error):
        if centroid_path is not None and centroid_path.is_file():
            try:
                centroid_path.unlink()
                metadata_path = _centroid_metadata_path(centroid_path)
                if metadata_path.is_file():
                    metadata_path.unlink()
            except OSError:
                pass
        return _result(
            protocol="APPLICATION_ENROLLMENT",
            user_id=user_id,
            audio_count=len(audio_paths),
            file_results=file_results,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="CENTROID_WRITE_FAILED",
        )

    return _result(
        protocol="APPLICATION_ENROLLMENT",
        success=True,
        status="ENROLLED",
        user_id=user_id,
        embedding_count=len(accepted_embeddings),
        audio_count=len(audio_paths),
        accepted_samples=len(accepted_embeddings),
        rejected_samples=len(audio_paths) - len(accepted_embeddings),
        requires_more_samples=False,
        total_speech_duration_sec=total_speech_duration,
        enrollment_quality_score=float(len(accepted_embeddings) / len(audio_paths)),
        embedding_path=stored_centroid_path,
        centroid_path=str(centroid_path),
        embedding_dim=int(centroid.size),
        l2_norm=float(np.linalg.norm(centroid)),
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
        file_results=file_results,
        quality=quality_results,
        embedding_consistency=consistency,
        metadata={
            "model_version": model_version,
            "min_accepted_enrollment_samples": min_accepted_samples,
            "min_samples_for_outlier_detection": min_samples_for_outlier,
            "min_centroid_similarity": min_centroid_similarity,
        },
        secret_phrase_configured=True,
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
    started_at = time.perf_counter()
    try:
        users = list_users(database_path)
    except sqlite3.Error:
        return _result(
            protocol="APPLICATION_SID",
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="DATABASE_ERROR",
        )
    if not users:
        return _result(protocol="APPLICATION_SID", identified=False, error="NO_ENROLLMENT")
    try:
        query = _embedding_for_audio(audio_path, extractor=extractor, audio_loader=audio_loader)
    except (OSError, ValueError, EmbeddingError):
        return _result(protocol="APPLICATION_SID", identified=False, error="INVALID_AUDIO")

    try:
        model_version = _model_version(config_path)
    except (OSError, TypeError, ValueError):
        return _result(protocol="APPLICATION_SID", identified=False, error="INVALID_SPEAKER_CONFIG")
    scores: list[tuple[str, float, Path]] = []
    missing_centroid = False
    mismatched_centroid = False
    for user in users:
        user_id = str(user.get("user_id", ""))
        if not _USER_ID_PATTERN.fullmatch(user_id):
            continue
        path_value = user.get("embedding_path")
        if not path_value:
            missing_centroid = True
            continue
        path = _resolve_centroid_path(path_value, config_path)
        try:
            centroid = _load_centroid(path, model_version)
            scores.append((user_id, float(np.dot(query, centroid)), path))
        except CentroidModelMismatchError:
            mismatched_centroid = True
        except (OSError, ValueError, EmbeddingError):
            missing_centroid = True

    if not scores:
        error = "CENTROID_MODEL_MISMATCH" if mismatched_centroid else (
            "CENTROID_NOT_FOUND" if missing_centroid else "NO_ENROLLMENT"
        )
        return _result(protocol="APPLICATION_SID", identified=False, error=error)
    threshold = _threshold(config_path, "application_sid_threshold_path", identification_threshold)
    if threshold is None:
        return _result(protocol="APPLICATION_SID", identified=False, error="THRESHOLD_NOT_CONFIGURED")
    user_id, similarity, centroid_path = max(scores, key=lambda item: item[1])
    identified = similarity >= threshold
    return _result(
        protocol="APPLICATION_SID",
        success=True,
        status="KNOWN" if identified else "UNKNOWN",
        candidate_user_id=user_id if identified else None,
        centroid_path=(
            _portable_centroid_path(centroid_path, config_path)
            if identified
            else None
        ),
        cosine_similarity=similarity,
        similarity=similarity,
        unknown_threshold=threshold,
        identified=identified,
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
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
    started_at = time.perf_counter()
    if not isinstance(candidate_user_id, str) or not _USER_ID_PATTERN.fullmatch(candidate_user_id):
        return _result(
            protocol="APPLICATION_SV",
            candidate_user_id=candidate_user_id,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="INVALID_USER_ID",
        )
    user = get_user(candidate_user_id, database_path)
    if not user:
        return _result(
            protocol="APPLICATION_SV",
            candidate_user_id=candidate_user_id,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="NO_ENROLLMENT",
        )
    path_value = user.get("embedding_path")
    if not path_value:
        return _result(
            protocol="APPLICATION_SV",
            candidate_user_id=candidate_user_id,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="CENTROID_NOT_FOUND",
        )
    try:
        query = _embedding_for_audio(audio_path, extractor=extractor, audio_loader=audio_loader)
    except (OSError, ValueError, EmbeddingError):
        return _result(
            protocol="APPLICATION_SV",
            candidate_user_id=candidate_user_id,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="INVALID_AUDIO",
        )
    try:
        centroid_path = _resolve_centroid_path(path_value, config_path)
        model_version = _model_version(config_path)
        similarity = float(np.dot(query, _load_centroid(centroid_path, model_version)))
    except CentroidModelMismatchError:
        return _result(protocol="APPLICATION_SV", candidate_user_id=candidate_user_id, error="CENTROID_MODEL_MISMATCH")
    except (OSError, ValueError, EmbeddingError):
        return _result(
            protocol="APPLICATION_SV",
            candidate_user_id=candidate_user_id,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="CENTROID_NOT_FOUND",
        )
    threshold = _threshold(config_path, "application_verification_threshold_path", verification_threshold)
    if threshold is None:
        return _result(
            protocol="APPLICATION_SV",
            candidate_user_id=candidate_user_id,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="THRESHOLD_NOT_CONFIGURED",
        )
    verified = similarity >= threshold
    return _result(
        protocol="APPLICATION_SV",
        success=True,
        status="VERIFIED" if verified else "REJECTED",
        candidate_user_id=candidate_user_id,
        centroid_path=_portable_centroid_path(centroid_path, config_path),
        cosine_similarity=similarity,
        similarity=similarity,
        verification_threshold=threshold,
        verified=verified,
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
        error=None,
    )


def delete_application_user(
    user_id: str,
    *,
    database_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
) -> dict:
    """Delete an application user and only that user's managed centroid."""

    started_at = time.perf_counter()
    if not isinstance(user_id, str) or not _USER_ID_PATTERN.fullmatch(user_id):
        return _result(
            protocol="APPLICATION_DELETE",
            user_id=user_id,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="INVALID_USER_ID",
        )
    user = get_user(user_id, database_path)
    if not user:
        return _result(
            protocol="APPLICATION_DELETE",
            user_id=user_id,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="USER_NOT_FOUND",
        )

    centroid_dir = _centroid_dir(config_path).resolve()
    managed_centroid = (centroid_dir / f"{user_id}.npy").resolve()
    try:
        if managed_centroid.is_file():
            managed_centroid.unlink()
        metadata_path = _centroid_metadata_path(managed_centroid)
        if metadata_path.is_file():
            metadata_path.unlink()
        if not delete_user(user_id, database_path):
            raise sqlite3.DatabaseError("User deletion did not affect one row")
    except (OSError, sqlite3.Error):
        return _result(
            protocol="APPLICATION_DELETE",
            user_id=user_id,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error="DELETE_FAILED",
        )
    return _result(
        protocol="APPLICATION_DELETE",
        success=True,
        status="DELETED",
        user_id=user_id,
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
        error=None,
    )
