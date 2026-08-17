"""Closed-set experimental speaker identification with a Linear SVM."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import joblib
import numpy as np

from src.audio.preprocessing import preprocess_audio
from src.speaker.embedding import ECAPAEmbeddingExtractor, get_embedding_extractor


MODEL_PATH = Path("models/experimental/speaker_svm_linear.pkl")
PROTOCOL = "SVM_CLOSED_SET"


class ClosedSetIdentificationResult(TypedDict):
    protocol: str
    candidate_speaker_id: str | None
    decision_score: float | None
    latency_ms: float
    success: bool
    error: str | None


@lru_cache(maxsize=1)
def _load_model_bundle(model_path: str = str(MODEL_PATH)) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"SVM model does not exist: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise ValueError("Invalid SVM closed-set model payload")
    if "model" not in payload or "embedding_dim" not in payload:
        raise ValueError("SVM model payload is missing required fields")
    return payload


@lru_cache(maxsize=1)
def _get_extractor() -> ECAPAEmbeddingExtractor:
    return get_embedding_extractor()


def _failure(started_at: float, error: Exception | str) -> ClosedSetIdentificationResult:
    return {
        "protocol": PROTOCOL,
        "candidate_speaker_id": None,
        "decision_score": None,
        "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        "success": False,
        "error": str(error),
    }


def identify_closed_set_svm(audio_path: str) -> ClosedSetIdentificationResult:
    """Identify one experimental SVM speaker; this protocol never returns UNKNOWN."""

    started_at = time.perf_counter()
    try:
        path = resolve_audio_path(audio_path)

        # Initialize Librosa before SpeechBrain registers optional lazy imports.
        audio, sample_rate = preprocess_audio(str(path))
        bundle = _load_model_bundle()
        extractor = _get_extractor()
        embedding, dimension, _ = extractor.extract(audio, sample_rate=sample_rate)
        expected_dimension = int(bundle["embedding_dim"])
        if dimension != expected_dimension:
            raise ValueError(
                f"Expected {expected_dimension}-D embedding; received {dimension}-D"
            )

        model = bundle["model"]
        features = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        decision = np.asarray(model.decision_function(features))
        classes = np.asarray(model.classes_)
        if decision.ndim == 1:
            decision = decision.reshape(1, -1)
        winner = int(np.argmax(decision[0]))
        candidate = str(classes[winner])
        return {
            "protocol": PROTOCOL,
            "candidate_speaker_id": candidate,
            "decision_score": float(decision[0, winner]),
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            "success": True,
            "error": None,
        }
    except Exception as error:  # Stable application boundary.
        return _failure(started_at, error)
