"""Regression checks for unknown-threshold and SV validation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


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
