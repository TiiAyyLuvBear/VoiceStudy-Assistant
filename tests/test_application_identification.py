"""Tests for cosine-only application enrollment and identification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.speaker import application_identification, enrollment


class _SequenceExtractor:
    def __init__(self, vectors: list[np.ndarray]) -> None:
        self.vectors = iter(vectors)

    def extract(self, audio, *, sample_rate):
        vector = np.asarray(next(self.vectors), dtype=np.float32)
        return vector, vector.size, 1.0


def test_enroll_user_requires_exactly_five_valid_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_paths = []
    for index in range(5):
        path = tmp_path / f"audio-{index}.wav"
        path.write_bytes(b"audio")
        audio_paths.append(str(path))
    vectors = [
        np.asarray([1.0, 0.0], dtype=np.float32)
        for _ in range(5)
    ]
    monkeypatch.setattr(enrollment, "APPLICATION_EMBEDDING_DIR", tmp_path / "users")
    monkeypatch.setattr(
        enrollment,
        "preprocess_audio",
        lambda path: (np.ones(10, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        enrollment,
        "_get_extractor",
        lambda: _SequenceExtractor(vectors),
    )

    result = enrollment.enroll_user("user_003", audio_paths)

    assert result["success"] is True
    assert result["audio_count"] == 5
    assert result["embedding_dim"] == 2
    assert np.isclose(result["l2_norm"], 1.0)
    assert np.allclose(
        np.load(result["embedding_path"], allow_pickle=False),
        [1.0, 0.0],
    )

    rejected = enrollment.enroll_user("user_004", audio_paths[:4])
    assert rejected["success"] is False
    assert "Exactly 5" in (rejected["error"] or "")


def _application_setup(
    tmp_path: Path,
    monkeypatch,
    query: np.ndarray,
) -> None:
    centroid_dir = tmp_path / "centroids"
    centroid_dir.mkdir()
    np.save(centroid_dir / "user_003.npy", [1.0, 0.0])
    np.save(centroid_dir / "user_004.npy", [0.0, 1.0])
    threshold = tmp_path / "threshold.json"
    threshold.write_text(json.dumps({"threshold": 0.68}), encoding="utf-8")
    monkeypatch.setattr(
        application_identification,
        "_load_settings",
        lambda: (centroid_dir, threshold),
    )
    monkeypatch.setattr(
        application_identification,
        "preprocess_audio",
        lambda path: (np.ones(10, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        application_identification,
        "_get_extractor",
        lambda: _SequenceExtractor([query]),
    )


def test_application_identification_known_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "query.wav"
    audio.write_bytes(b"audio")
    _application_setup(
        tmp_path,
        monkeypatch,
        np.asarray([0.8, 0.6], dtype=np.float32),
    )

    result = application_identification.identify_application_user(str(audio))

    assert result["protocol"] == "APPLICATION_SID"
    assert result["candidate_user_id"] == "user_003"
    assert np.isclose(result["cosine_similarity"], 0.8)
    assert result["unknown_threshold"] == 0.68
    assert result["status"] == "KNOWN"
    assert result["success"] is True
    assert result["error"] is None


def test_application_identification_unknown_hides_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "query.wav"
    audio.write_bytes(b"audio")
    _application_setup(
        tmp_path,
        monkeypatch,
        np.asarray([-1.0, -1.0], dtype=np.float32),
    )

    result = application_identification.identify_application_user(str(audio))

    assert result["candidate_user_id"] is None
    assert result["cosine_similarity"] < result["unknown_threshold"]
    assert result["status"] == "UNKNOWN"
    assert result["success"] is True


def test_application_identification_missing_audio_is_stable(tmp_path: Path) -> None:
    result = application_identification.identify_application_user(
        str(tmp_path / "missing.wav")
    )
    assert result["status"] == "ERROR"
    assert result["success"] is False
    assert result["candidate_user_id"] is None


def test_application_identification_without_enrolled_users(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "query.wav"
    audio.write_bytes(b"audio")
    empty_gallery = tmp_path / "empty"
    empty_gallery.mkdir()
    threshold = tmp_path / "threshold.json"
    threshold.write_text(json.dumps({"threshold": 0.68}), encoding="utf-8")
    monkeypatch.setattr(
        application_identification,
        "_load_settings",
        lambda: (empty_gallery, threshold),
    )
    monkeypatch.setattr(
        application_identification,
        "preprocess_audio",
        lambda path: (np.ones(10, dtype=np.float32), 16000),
    )

    result = application_identification.identify_application_user(str(audio))

    assert result["status"] == "ERROR"
    assert result["success"] is False
    assert result["candidate_user_id"] is None
    assert "No application user centroids" in (result["error"] or "")

def test_enroll_user_rejects_non_application_prefix(
    tmp_path: Path,
) -> None:
    audio_paths = []
    for index in range(5):
        path = tmp_path / f"audio-{index}.wav"
        path.write_bytes(b"audio")
        audio_paths.append(str(path))

    result = enrollment.enroll_user("spk0003", audio_paths)

    assert result["success"] is False
    assert "user_" in (result["error"] or "")