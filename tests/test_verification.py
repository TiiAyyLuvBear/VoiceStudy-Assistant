"""Tests for application speaker verification."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np

from src.speaker import verification


class _FakeExtractor:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def extract(self, audio, *, sample_rate):
        return self.vector, self.vector.size, 1.0


def _setup(
    tmp_path: Path,
    monkeypatch,
    query: np.ndarray,
) -> tuple[Path, Path]:
    audio = tmp_path / "query.wav"
    audio.write_bytes(b"audio")
    centroids = tmp_path / "users"
    centroids.mkdir()
    np.save(centroids / "user_003.npy", [1.0, 0.0])
    threshold = tmp_path / "threshold.json"
    threshold.write_text(json.dumps({"threshold": 0.72}), encoding="utf-8")
    monkeypatch.setattr(
        verification,
        "_load_settings",
        lambda: (centroids, threshold),
    )
    monkeypatch.setattr(
        verification,
        "preprocess_audio",
        lambda path: (np.ones(20, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        verification,
        "_get_extractor",
        lambda: _FakeExtractor(query),
    )
    return audio, centroids


def test_genuine_verification(tmp_path: Path, monkeypatch) -> None:
    audio, _ = _setup(tmp_path, monkeypatch, np.asarray([0.8, 0.6]))
    result = verification.verify_speaker(str(audio), "user_003")

    assert result["candidate_user_id"] == "user_003"
    assert np.isclose(result["similarity"], 0.8)
    assert result["verification_threshold"] == 0.72
    assert result["verified"] is True
    assert result["success"] is True
    assert result["error"] is None


def test_impostor_verification(tmp_path: Path, monkeypatch) -> None:
    audio, _ = _setup(tmp_path, monkeypatch, np.asarray([0.0, 1.0]))
    result = verification.verify_speaker(str(audio), "user_003")

    assert np.isclose(result["similarity"], 0.0)
    assert result["verified"] is False
    assert result["success"] is True


def test_missing_candidate_centroid_and_non_application_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio, _ = _setup(tmp_path, monkeypatch, np.asarray([1.0, 0.0]))
    missing = verification.verify_speaker(str(audio), "user_004")
    legacy = verification.verify_speaker(str(audio), "spk0003")

    assert missing["success"] is False
    assert "does not exist" in (missing["error"] or "")
    assert legacy["success"] is False
    assert legacy["candidate_user_id"] == "spk0003"


def test_verification_audio_error(tmp_path: Path) -> None:
    result = verification.verify_speaker(
        str(tmp_path / "bad.wav"),
        "user_003",
    )
    assert result["success"] is False
    assert result["verified"] is False
    assert "does not exist" in (result["error"] or "")


def test_verification_validation_artifacts_are_consistent() -> None:
    with Path(
        "experiments/validation/"
        "speaker_disjoint_verification_validation_scores.csv"
    ).open("r", encoding="utf-8-sig", newline="") as stream:
        scores = list(csv.DictReader(stream))
    with Path(
        "experiments/validation/verification_threshold_results.csv"
    ).open("r", encoding="utf-8-sig", newline="") as stream:
        results = list(csv.DictReader(stream))
    config = json.loads(
        Path("models/experimental/verification_threshold.json").read_text(
            encoding="utf-8"
        )
    )
    selected = [row for row in results if row["selected"] == "true"]

    assert len(scores) == 308
    assert sum(row["label"] == "1" for row in scores) == 10
    assert sum(row["label"] == "0" for row in scores) == 298
    assert len(selected) == 1
    assert float(selected[0]["threshold"]) == config["threshold"]
    assert float(selected[0]["eer_gap"]) == abs(
        float(selected[0]["FAR"]) - float(selected[0]["FRR"])
    )
