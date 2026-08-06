"""Regression checks for leakage-safe speaker experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np


def _metadata() -> list[dict[str, str]]:
    with Path("data/metadata/embedding_metadata.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def test_svm_feature_archive_contains_train_only() -> None:
    expected = {
        row["audio_id"]
        for row in _metadata()
        if row["protocol"] == "SVM_CLOSED_SET"
        and row["split"] == "svm_closed_set_train"
    }
    enrollment = {
        row["audio_id"]
        for row in _metadata()
        if row["split"] == "svm_closed_set_enrollment"
    }
    archive = np.load(
        "experiments/svm/svm_closed_set_train_features.npz",
        allow_pickle=False,
    )
    actual = set(archive["audio_ids"].tolist())

    assert archive["X"].shape == (100, 192)
    assert archive["protocol"].tolist() == ["SVM_CLOSED_SET"]
    assert archive["split"].tolist() == ["svm_closed_set_train"]
    assert actual == expected
    assert not actual & enrollment


def test_saved_model_and_selection_config_are_train_validation_only() -> None:
    config = json.loads(
        Path("models/experimental/svm_best_config.json").read_text(encoding="utf-8")
    )
    payload = joblib.load("models/experimental/speaker_svm_linear.pkl")
    train_ids = {
        row["audio_id"]
        for row in _metadata()
        if row["protocol"] == "SVM_CLOSED_SET"
        and row["split"] == "svm_closed_set_train"
    }

    assert config["selection_split"] == "svm_closed_set_validation"
    assert config["training_split"] == "svm_closed_set_train"
    assert config["selected_C"] in (0.1, 1.0, 10.0)
    assert set(payload["training_audio_ids"]) == train_ids
    assert payload["training_split"] == "svm_closed_set_train"
    assert len(payload["classes"]) == 10


def test_experimental_centroids_are_192d_and_l2_normalized() -> None:
    svm = sorted(
        Path("models/experimental/svm_closed_set_centroids").glob("*.npy")
    )
    cosine = sorted(
        Path("models/experimental/cosine_validation_centroids").glob("*.npy")
    )

    assert len(svm) == 10
    assert len(cosine) == 2
    for path in svm + cosine:
        vector = np.load(path, allow_pickle=False)
        assert vector.shape == (192,)
        assert np.isfinite(vector).all()
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
