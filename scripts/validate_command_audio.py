"""Kiểm tra command audio, checksum và rò rỉ sang Speaker datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np


REPORT_FIELDS = (
    "recording_id",
    "command_id",
    "split",
    "speaker_id",
    "audio_path",
    "manifest_status",
    "exists",
    "readable",
    "sample_rate",
    "channels",
    "duration_sec",
    "rms",
    "checksum_sha256",
    "issues",
)

SPEAKER_SPLIT_NAMES = (
    "speaker_enrollment.csv",
    "speaker_train.csv",
    "speaker_validation.csv",
    "speaker_test.csv",
    "unknown_validation.csv",
    "unknown_test.csv",
    "unknown_speaker_test.csv",
)


def _speaker_audio_paths(metadata_dir: Path) -> set[str]:
    paths: set[str] = set()
    for name in SPEAKER_SPLIT_NAMES:
        source = metadata_dir / name
        if not source.is_file():
            source = Path(name)
        if not source.is_file():
            continue
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            path_column = next(
                (
                    field
                    for field in ("audio_path", "file_path", "path")
                    if field in (reader.fieldnames or ())
                ),
                None,
            )
            if path_column:
                paths.update(
                    str(Path(row[path_column])).replace("\\", "/").casefold()
                    for row in reader
                    if row.get(path_column)
                )
    return paths


def _inspect_wav(
    path: Path,
    *,
    expected_rate: int,
    min_duration: float,
    max_duration: float,
    min_rms: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "exists": path.is_file(),
        "readable": False,
        "sample_rate": "",
        "channels": "",
        "duration_sec": "",
        "rms": "",
        "checksum_sha256": "",
        "issues": [],
    }
    issues: list[str] = result["issues"]  # type: ignore[assignment]
    if not path.is_file():
        issues.append("missing_file")
        return result

    try:
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frame_count = stream.getnframes()
            frames = stream.readframes(frame_count)
        duration = frame_count / sample_rate if sample_rate else 0.0
        samples = np.frombuffer(frames, dtype="<i2") if sample_width == 2 else np.array([])
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
        result.update(
            {
                "readable": True,
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_sec": round(duration, 3),
                "rms": round(rms, 3),
                "checksum_sha256": checksum,
            }
        )
        if sample_rate != expected_rate:
            issues.append("wrong_sample_rate")
        if channels != 1:
            issues.append("not_mono")
        if sample_width != 2:
            issues.append("not_pcm16")
        if duration < min_duration:
            issues.append("too_short")
        if duration > max_duration:
            issues.append("too_long")
        if rms < min_rms:
            issues.append("signal_too_quiet")
    except (wave.Error, OSError, ValueError) as exc:
        issues.append(f"unreadable:{exc}")
    return result


def validate_manifest(
    manifest: Path,
    *,
    metadata_dir: Path,
    expected_rate: int = 16000,
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    min_rms: float = 50.0,
) -> list[dict[str, object]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))

    speaker_paths = _speaker_audio_paths(metadata_dir)
    results: list[dict[str, object]] = []
    checksums: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        path_value = row.get("audio_path", "")
        status = row.get("status", "pending")
        issues: list[str] = []
        if status != "recorded" or not path_value:
            inspected: dict[str, object] = {
                "exists": False,
                "readable": False,
                "sample_rate": "",
                "channels": "",
                "duration_sec": "",
                "rms": "",
                "checksum_sha256": "",
                "issues": ["not_recorded"],
            }
        else:
            inspected = _inspect_wav(
                Path(path_value),
                expected_rate=expected_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                min_rms=min_rms,
            )
        issues.extend(inspected.pop("issues"))  # type: ignore[arg-type]
        normalized_path = str(Path(path_value)).replace("\\", "/").casefold()
        if path_value and normalized_path in speaker_paths:
            issues.append("speaker_training_leakage")

        result = {
            "recording_id": row.get("recording_id", ""),
            "command_id": row.get("command_id", ""),
            "split": row.get("split", ""),
            "speaker_id": row.get("speaker_id", ""),
            "audio_path": path_value,
            "manifest_status": status,
            **inspected,
            "issues": issues,
        }
        results.append(result)
        checksum = str(result["checksum_sha256"])
        if checksum:
            checksums[checksum].append(result)

    for duplicates in checksums.values():
        if len(duplicates) > 1:
            for result in duplicates:
                result["issues"].append("duplicate_content")  # type: ignore[union-attr]
    return results


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/commands/command_audio_manifest.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/command_audio_validation.csv"),
    )
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--min-rms", type=float, default=50.0)
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    results = validate_manifest(
        args.manifest,
        metadata_dir=args.metadata_dir,
        expected_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_rms=args.min_rms,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({**result, "issues": ";".join(result["issues"])})

    failures = sum(bool(result["issues"]) for result in results)
    recorded = sum(result["manifest_status"] == "recorded" for result in results)
    print(f"Recorded: {recorded}/{len(results)}; rows with issues: {failures}")
    print(f"Report: {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
