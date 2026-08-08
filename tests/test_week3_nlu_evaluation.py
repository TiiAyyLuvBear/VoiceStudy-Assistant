"""Tests for the frozen Week 3 ASR/NLU evaluation artifacts."""

import csv
from pathlib import Path

from scripts.evaluate_week3_nlu import (
    _intent_metrics,
    evaluate_audio_commands,
    evaluate_text_commands,
)


def _command_test_rows() -> list[dict[str, str]]:
    with Path("data/metadata/command_test.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def test_ground_truth_text_covers_all_five_application_labels() -> None:
    rows = evaluate_text_commands(_command_test_rows())

    assert len(rows) == 30
    assert {row["expected_intent"] for row in rows} == {
        "GET_TIME",
        "VIEW_SCHEDULE",
        "ADD_SCHEDULE",
        "VIEW_PRIVATE_NOTE",
        "OUT_OF_SCOPE",
    }
    assert all(row["intent_correct"] == "true" for row in rows)


def test_missing_audio_remains_explicit_and_is_not_scored(tmp_path: Path) -> None:
    command = _command_test_rows()[0]
    manifest = {
        "recording_id": "REC_PENDING",
        "command_id": command["command_id"],
        "speaker_id": "cmdspk01",
        "status": "pending",
        "audio_path": "",
    }

    rows = evaluate_audio_commands(
        [command],
        [manifest],
        config_path=Path("config.yaml"),
        lock_hash="test-lock",
        checkpoint_path=tmp_path / "checkpoint.csv",
    )

    assert rows[0]["evaluated"] == "false"
    assert rows[0]["intent_correct"] == ""
    assert rows[0]["error"] == "not_recorded"


def test_intent_metric_schema_keeps_requested_accuracy_fields() -> None:
    rows = [
        {"expected_intent": "GET_TIME", "intent_correct": "true"},
        {"expected_intent": "OUT_OF_SCOPE", "intent_correct": "false"},
    ]

    metrics = _intent_metrics(rows)

    assert metrics["overall_5_label_accuracy"] == 0.5
    assert metrics["functional_4_intent_accuracy"] == 1.0
    assert metrics["out_of_scope_rejection_accuracy"] == 0.0
    assert "per_intent_accuracy" in metrics
