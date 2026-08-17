"""Regression checks for unknown-threshold and SV validation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.utils.files import sha256_file


def _rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_selected_unknown_threshold_metrics_are_consistent() -> None:
    results = _rows(
        "experiments/validation/cosine_unknown_threshold_results.csv"
    )
    selected = [row for row in results if row["selected"] == "true"]
    config = json.loads(
        Path("models/experimental/cosine_unknown_threshold.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(selected) == 1
    assert float(selected[0]["threshold"]) == config["threshold"]
    assert int(selected[0]["known_count"]) == 10
    assert int(selected[0]["unknown_count"]) == 144
    assert config["metrics"]["overall_detection_accuracy"] >= 0.99


def test_sv_trials_use_cosine_validation_speakers_only() -> None:
    trials = _rows(
        "experiments/validation/"
        "speaker_disjoint_verification_validation_trials.csv"
    )
    genuine = [row for row in trials if row["trial_type"] == "GENUINE"]
    impostor = [row for row in trials if row["trial_type"] == "IMPOSTOR"]
    unknown = [
        row for row in trials if row["trial_type"] == "UNKNOWN_IMPOSTOR"
    ]

    assert len(genuine) == 10
    assert len(impostor) == 10
    assert len(unknown) == 288
    assert all(row["protocol"] == "COSINE_VALIDATION" for row in trials)
    assert all(
        row["query_speaker_id"] == row["claimed_speaker_id"]
        for row in genuine
    )
    assert all(
        row["query_speaker_id"] != row["claimed_speaker_id"]
        for row in impostor + unknown
    )


def test_speaker_test_config_pins_model_and_frozen_manifest() -> None:
    config = json.loads(
        Path("experiments/test/speaker_test_config.json").read_text(
            encoding="utf-8-sig"
        )
    )
    ecapa = config["ecapa"]
    manifest = config["split_manifest"]

    assert ecapa["frozen"] is True
    assert sha256_file(ecapa["checkpoint"]) == ecapa["checkpoint_sha256"]
    assert (
        sha256_file(ecapa["hyperparameters"])
        == ecapa["hyperparameters_sha256"]
    )
    assert sha256_file(manifest["path"]) == manifest["sha256"]
    assert manifest["dataset_version"] == "v1"
    assert manifest["freeze_status"] == "FROZEN"


def test_application_config_pins_verification_threshold_and_ecapa() -> None:
    config = json.loads(
        Path("models/application/application_sid_config.json").read_text(
            encoding="utf-8"
        )
    )
    verification = json.loads(
        Path("models/application/application_verification_threshold.json").read_text(
            encoding="utf-8"
        )
    )
    model = config["model"]

    assert (
        config["speaker_verification"]["threshold"]
        == verification["threshold"]
    )
    assert (
        config["speaker_verification"]["threshold_source"]["source"]
        == "models/application/application_verification_threshold.json"
    )
    assert verification["application_audio_calibrated"] is True
    assert verification["threshold_tuned_on_v2_test"] is False
    assert config["application_enrollment"]["embedding_count"] >= 3
    assert sha256_file(model["checkpoint"]) == model["checkpoint_sha256"]
    assert (
        sha256_file(model["hyperparameters"])
        == model["hyperparameters_sha256"]
    )
