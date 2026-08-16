"""Unit tests for dependency-light ASR fine-tuning helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.asr.finetuning import audio_bucket_seconds, epoch_rows, load_finetune_rows
from scripts.finalize_asr_v3 import metric_comparison


def test_audio_bucket_seconds_uses_smallest_covering_window() -> None:
    assert audio_bucket_seconds(16_000, 16_000, maximum_seconds=16) == 1
    assert audio_bucket_seconds(16_001, 16_000, maximum_seconds=16) == 2
    assert audio_bucket_seconds(999_999, 16_000, maximum_seconds=16) == 16


def test_epoch_shuffle_is_reproducible_without_mutating_source() -> None:
    rows = [{"audio_id": str(index)} for index in range(10)]
    original = list(rows)
    assert epoch_rows(rows, seed=42, epoch=1) == epoch_rows(rows, seed=42, epoch=1)
    assert epoch_rows(rows, seed=42, epoch=1) != epoch_rows(rows, seed=42, epoch=2)
    assert rows == original


def test_load_finetune_rows_requires_existing_audio(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("audio_id", "audio_path", "transcript")
        )
        writer.writeheader()
        writer.writerow(
            {"audio_id": "a", "audio_path": str(tmp_path / "missing.wav"), "transcript": "xin chào"}
        )
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_finetune_rows(csv_path)


def test_metric_comparison_reports_absolute_and_relative_improvement() -> None:
    result = metric_comparison(0.25, 0.20)
    assert result["absolute_change"] == pytest.approx(-0.05)
    assert result["absolute_percentage_point_change"] == pytest.approx(-5.0)
    assert result["relative_improvement"] == pytest.approx(0.20)
