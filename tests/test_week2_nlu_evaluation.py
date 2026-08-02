"""Tests for Week 2 text-level validation report generation."""

import csv
from pathlib import Path

from scripts.evaluate_week2_nlu import (
    _optional_ratio,
    evaluate_audio_rows,
    evaluate_text_rows,
)


def test_text_reports_cover_intent_oos_and_entities() -> None:
    path = Path("data/metadata/command_validation.csv")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        commands = list(csv.DictReader(stream))

    ground_truth, out_of_scope, entities = evaluate_text_rows(commands)

    assert len(ground_truth) == 30
    assert all(row["intent_correct"] == "true" for row in ground_truth)
    assert len(out_of_scope) == 10
    assert all(row["correctly_rejected"] == "true" for row in out_of_scope)
    assert len(entities) == 30
    assert all(row["entity_exact_match"] == "true" for row in entities)
    add_rows = [row for row in ground_truth if row["expected_intent"] == "ADD_SCHEDULE"]
    assert add_rows
    assert all(row["can_write_database"] == "true" for row in add_rows)


def test_audio_report_keeps_unrecorded_commands_explicitly_pending() -> None:
    path = Path("data/metadata/command_validation.csv")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        command = next(csv.DictReader(stream))
    manifest = {
        "recording_id": "REC_PENDING",
        "command_id": command["command_id"],
        "split": "validation",
        "status": "pending",
        "audio_path": "",
        "speaker_id": "",
    }

    rows = evaluate_audio_rows(
        [command],
        [manifest],
        config_path=Path("config.yaml"),
        previous={},
    )

    assert len(rows) == 1
    assert rows[0]["evaluated"] == "false"
    assert rows[0]["asr_success"] == "false"
    assert rows[0]["error"] == "not_recorded"


def test_empty_audio_subset_has_no_measured_rate() -> None:
    assert _optional_ratio([], "intent_correct") is None
