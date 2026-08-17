"""Tests for portable ECAPA Kaggle dataset builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_ecapa_kaggle_dataset import SPEAKER_SPLITS, build_kaggle_dataset
from src.utils.files import sha256_file


FIELDS = ("audio_id", "audio_path", "normalized_speaker_id", "checksum")


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path) -> tuple[Path, Path]:
    source = root / "v2"
    metadata = source / "metadata"
    audio_root = root / "audio"
    split_speakers = {
        **{name: "svm" for name in SPEAKER_SPLITS[:4]},
        "cosine_test_enrollment.csv": "enrolled",
        "cosine_test_query.csv": "enrolled",
        "cosine_test_unknown.csv": "unknown",
    }
    for index, (name, speaker) in enumerate(split_speakers.items()):
        relative = Path("dev") / f"{index}.wav"
        audio = audio_root / relative
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(f"audio-{index}".encode())
        _write_rows(metadata / name, [{
            "audio_id": str(index),
            "audio_path": relative.as_posix(),
            "normalized_speaker_id": speaker,
            "checksum": sha256_file(audio),
        }])
    for name in (
        "selected_svm_experimental_speakers.csv",
        "selected_test_enrolled_speakers.csv",
        "selected_test_unknown_speakers.csv",
    ):
        (metadata / name).write_text("speaker_id\nexample\n", encoding="utf-8")
    for name in ("asr_validation.csv", "asr_test.csv"):
        _write_rows(metadata / name, [{
            "audio_id": f"asr-{name}",
            "audio_path": f"data/audio/dev/{name}.wav",
            "normalized_speaker_id": "asr",
            "checksum": "unused",
        }])
    (source / "split_manifest.json").write_text("{}\n", encoding="utf-8")
    return source, audio_root


def test_builds_portable_dataset_with_audio(tmp_path: Path) -> None:
    source, audio_root = _fixture(tmp_path)
    output = tmp_path / "package"
    manifest = build_kaggle_dataset(
        source_root=source, audio_root=audio_root, output_root=output
    )

    assert manifest["speaker_audio"] == 7
    assert manifest["asr_audio_included"] is False
    for name in SPEAKER_SPLITS:
        with (output / "metadata" / name).open(encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))
        copied = output / row["audio_path"]
        assert copied.is_file()
        assert sha256_file(copied) == row["checksum"]
    assert len(list((output / "audio").rglob("*.wav"))) == 7
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_refuses_overwrite(tmp_path: Path) -> None:
    source, audio_root = _fixture(tmp_path)
    output = tmp_path / "package"
    output.mkdir()
    try:
        build_kaggle_dataset(
            source_root=source, audio_root=audio_root, output_root=output
        )
    except FileExistsError as error:
        assert "Refusing to overwrite" in str(error)
    else:
        raise AssertionError("Expected overwrite refusal")
