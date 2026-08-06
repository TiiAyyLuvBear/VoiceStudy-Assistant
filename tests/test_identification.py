"""Protocol-level identification regression tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_svm_closed_set_validation_results_cover_all_c_values() -> None:
    rows = _rows("experiments/svm/svm_training_results.csv")
    assert [float(row["C"]) for row in rows] == [0.1, 1.0, 10.0]
    assert all(int(row["train_sample_count"]) == 100 for row in rows)
    assert all(int(row["validation_sample_count"]) == 50 for row in rows)
    assert all(float(row["validation_accuracy"]) == 1.0 for row in rows)


def test_validation_cosine_identification_and_unknown_rejection() -> None:
    known = _rows(
        "experiments/validation/cosine_validation_known_scores.csv"
    )
    unknown = _rows(
        "experiments/validation/cosine_validation_unknown_scores.csv"
    )
    threshold = json.loads(
        Path("models/experimental/cosine_unknown_threshold.json").read_text(
            encoding="utf-8"
        )
    )["threshold"]

    assert len(known) == 10
    assert all(row["correct"] == "true" for row in known)
    assert len(unknown) == 144
    assert all(float(row["max_similarity"]) < threshold for row in unknown)
