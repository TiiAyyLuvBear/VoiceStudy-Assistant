"""Tests for the closed-set SVM identification API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.speaker import identification


class _FakeExtractor:
    def extract(self, audio, *, sample_rate):
        assert sample_rate == 16000
        return np.ones(192, dtype=np.float32) / np.sqrt(192), 192, 1.0


class _FakeModel:
    classes_ = np.asarray(
        ["exp_svm_spk_0001", "exp_svm_spk_0002", "exp_svm_spk_0003"]
    )

    def decision_function(self, features):
        assert features.shape == (1, 192)
        return np.asarray([[0.1, 2.41, -0.2]])


def test_closed_set_identification_contract(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        identification,
        "preprocess_audio",
        lambda path: (np.ones(100, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        identification,
        "_load_model_bundle",
        lambda: {"model": _FakeModel(), "embedding_dim": 192},
    )
    monkeypatch.setattr(identification, "_get_extractor", lambda: _FakeExtractor())

    result = identification.identify_closed_set_svm(str(audio))

    assert result["protocol"] == "SVM_CLOSED_SET"
    assert result["candidate_speaker_id"] == "exp_svm_spk_0002"
    assert result["decision_score"] == 2.41
    assert result["latency_ms"] >= 0
    assert result["success"] is True
    assert result["error"] is None


def test_closed_set_missing_audio_is_stable(tmp_path: Path) -> None:
    result = identification.identify_closed_set_svm(str(tmp_path / "missing.wav"))

    assert result["protocol"] == "SVM_CLOSED_SET"
    assert result["candidate_speaker_id"] is None
    assert result["decision_score"] is None
    assert result["success"] is False
    assert "does not exist" in (result["error"] or "")
