"""Test command audio manifest và validator bằng WAV fixture."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np

from scripts.prepare_command_audio_manifest import build_rows
from scripts.record_command_audio import _record_until_enter
from scripts.validate_command_audio import validate_manifest


def _write_tone(path: Path, *, sample_rate: int = 16000) -> None:
    time_axis = np.arange(sample_rate, dtype=np.float64) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * 440 * time_axis) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples.tobytes())


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "recording_id",
        "command_id",
        "split",
        "expected_transcript",
        "intent",
        "speaker_id",
        "audio_path",
        "sample_rate",
        "channels",
        "duration_sec",
        "recorded_at",
        "recording_device",
        "status",
        "notes",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _recorded_row(audio_path: Path, recording_id: str = "REC_VAL0001") -> dict[str, str]:
    return {
        "recording_id": recording_id,
        "command_id": recording_id.removeprefix("REC_"),
        "split": "validation",
        "expected_transcript": "bây giờ là mấy giờ",
        "intent": "GET_TIME",
        "speaker_id": "cmdspk01",
        "audio_path": str(audio_path),
        "sample_rate": "16000",
        "channels": "1",
        "duration_sec": "1.0",
        "recorded_at": "2026-07-28T10:00:00+07:00",
        "recording_device": "test",
        "status": "recorded",
        "notes": "",
    }


def test_prepare_manifest_has_60_prompts() -> None:
    rows = build_rows(
        [
            Path("data/metadata/command_validation.csv"),
            Path("data/metadata/command_test.csv"),
        ]
    )
    assert len(rows) == 60
    assert {row["split"] for row in rows} == {"validation", "test"}
    assert all(row["status"] == "pending" for row in rows)


def test_valid_command_audio(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    manifest = tmp_path / "manifest.csv"
    _write_tone(audio)
    _write_manifest(manifest, [_recorded_row(audio)])

    results = validate_manifest(manifest, metadata_dir=tmp_path)

    assert results[0]["issues"] == []
    assert results[0]["sample_rate"] == 16000
    assert results[0]["channels"] == 1


def test_duplicate_audio_is_reported(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    manifest = tmp_path / "manifest.csv"
    _write_tone(audio)
    _write_manifest(
        manifest,
        [_recorded_row(audio), _recorded_row(audio, "REC_VAL0002")],
    )

    results = validate_manifest(manifest, metadata_dir=tmp_path)

    assert all("duplicate_content" in result["issues"] for result in results)


def test_pending_recording_is_reported(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    row = _recorded_row(tmp_path / "missing.wav")
    row.update({"status": "pending", "audio_path": ""})
    _write_manifest(manifest, [row])

    results = validate_manifest(manifest, metadata_dir=tmp_path)

    assert results[0]["issues"] == ["not_recorded"]


def test_validator_can_limit_results_to_one_split(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    manifest = tmp_path / "manifest.csv"
    _write_tone(audio)
    validation = _recorded_row(audio)
    test = _recorded_row(audio, "REC_TST0001")
    test["split"] = "test"
    _write_manifest(manifest, [validation, test])

    results = validate_manifest(
        manifest,
        metadata_dir=tmp_path,
        split_name="validation",
    )

    assert len(results) == 1
    assert results[0]["split"] == "validation"


def test_speaker_training_leakage_is_reported(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    manifest = tmp_path / "manifest.csv"
    _write_tone(audio)
    _write_manifest(manifest, [_recorded_row(audio)])
    with (tmp_path / "speaker_train.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio_path"])
        writer.writeheader()
        writer.writerow({"audio_path": str(audio)})

    results = validate_manifest(manifest, metadata_dir=tmp_path)

    assert "speaker_training_leakage" in results[0]["issues"]


def test_manual_recording_stops_on_enter(monkeypatch) -> None:
    class FakeInputStream:
        def __init__(self, **kwargs) -> None:
            self.callback = kwargs["callback"]

        def __enter__(self):
            samples = np.full((320, 1), 0.25, dtype=np.float32)
            self.callback(samples, 320, None, None)
            return self

        def __exit__(self, *_args) -> None:
            return None

    class FakeSoundDevice:
        InputStream = FakeInputStream

    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    samples = _record_until_enter(
        FakeSoundDevice(),
        sample_rate=16000,
        device=1,
    )

    assert samples.shape == (320, 1)
    assert samples.dtype == np.float32
