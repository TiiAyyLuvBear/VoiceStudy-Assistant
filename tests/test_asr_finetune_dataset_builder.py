"""Tests for the full-source ASR fine-tuning dataset builder."""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

import pytest

from scripts.build_asr_finetune_dataset import build_dataset


FIELDS = (
    "audio_id",
    "audio_path",
    "original_split",
    "project_split",
    "locale",
    "speaker_id",
    "transcript",
    "intent",
    "duration_sec",
    "sample_rate",
    "num_channels",
    "is_valid",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_row(
    audio_id: str,
    project_split: str,
    speaker_id: str,
    *,
    is_valid: str = "True",
    transcript: str = "ha\u0303y thử",
) -> dict[str, str]:
    return {
        "audio_id": audio_id,
        "audio_path": f"dev/{audio_id}.wav",
        "original_split": "validation",
        "project_split": project_split,
        "locale": "vi-VN",
        "speaker_id": speaker_id,
        "transcript": transcript,
        "intent": "sample",
        "duration_sec": "1.5",
        "sample_rate": "16000",
        "num_channels": "1",
        "is_valid": is_valid,
    }


def test_builds_all_splits_and_accounts_for_rejected_source(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    rows = [
        _source_row("train_svm", "SVM", "speaker_train_1"),
        _source_row("train_unused", "UNUSED", "speaker_train_2"),
        _source_row("validation", "VALIDATION", "speaker_validation"),
        _source_row("test", "TEST", "speaker_test"),
        _source_row(
            "invalid", "UNUSED", "speaker_train_2", is_valid="False"
        ),
    ]
    for row in rows:
        audio_path = audio_root / row["audio_path"]
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(f"audio-{row['audio_id']}".encode())

    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, FIELDS, rows)
    invalid_report = tmp_path / "invalid.csv"
    _write_csv(
        invalid_report,
        ("audio_id", "reason"),
        [{"audio_id": "invalid", "reason": "low_volume"}],
    )

    output_root = tmp_path / "processed" / "v3"
    manifest = build_dataset(
        inventory_path=inventory,
        audio_root=audio_root,
        invalid_audio_path=invalid_report,
        output_root=output_root,
    )

    assert manifest["freeze_status"] == "DEVELOPMENT"
    assert manifest["source"]["source_row_count"] == 5
    assert manifest["source"]["accepted_row_count"] == 4
    assert manifest["source"]["rejected_row_count"] == 1
    assert manifest["source"]["all_source_rows_accounted_for"] is True
    assert manifest["datasets"]["train"]["row_count"] == 2
    assert manifest["datasets"]["validation"]["row_count"] == 1
    assert manifest["datasets"]["test"]["row_count"] == 1
    assert manifest["rejected"]["reason_counts"] == {"low_volume": 1}

    with (output_root / "metadata/asr_finetune_train.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        train_rows = list(csv.DictReader(stream))
    assert {row["source_project_split"] for row in train_rows} == {
        "SVM",
        "UNUSED",
    }
    assert all(unicodedata.is_normalized("NFC", row["transcript"]) for row in train_rows)
    assert all(len(row["audio_sha256"]) == 64 for row in train_rows)


def test_rejects_speaker_leakage_between_splits(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    rows = [
        _source_row("train", "SVM", "same_speaker"),
        _source_row("test", "TEST", "same_speaker"),
    ]
    for row in rows:
        audio_path = audio_root / row["audio_path"]
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(row["audio_id"].encode())
    inventory = tmp_path / "inventory.csv"
    _write_csv(inventory, FIELDS, rows)

    with pytest.raises(ValueError, match="Speaker leakage"):
        build_dataset(
            inventory_path=inventory,
            audio_root=audio_root,
            invalid_audio_path=None,
            output_root=tmp_path / "out",
        )
