"""Build large ECAPA split metadata that references unprocessed data/audio files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.utils.files import sha256_file


SPLIT_COUNTS = {"train": 20, "validation": 5, "test": 5}
SPLIT_NAMES = {
    "train": "svm_closed_set_train",
    "validation": "svm_closed_set_validation",
    "test": "svm_closed_set_test",
}
METADATA_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
REQUIRED_FIELDS = (
    "audio_id",
    "audio_path",
    "speaker_id",
    "duration_sec",
    "sample_rate",
    "num_channels",
    "is_valid",
)
METADATA_FIELDS = (
    "audio_id",
    "audio_path",
    "speaker_id",
    "normalized_speaker_id",
    "split",
    "split_name",
    "split_role",
    "protocol",
    "role",
    "duration_sec",
    "sample_rate",
    "num_channels",
    "checksum",
    "original_split",
    "transcript",
    "locale",
    "speaker_sex",
    "speaker_age",
    "intent",
    "scenario_str",
)
REPORT_FIELDS = (
    "audio_id",
    "audio_path",
    "speaker_id",
    "normalized_speaker_id",
    "selected_split",
    "duration_sec",
    "sample_rate",
    "num_channels",
    "is_valid",
    "checksum",
    "status",
    "issues",
)
MAPPING_FIELDS = (
    "speaker_id",
    "normalized_speaker_id",
    "speaker_sex",
    "speaker_age",
    "eligible_audio",
    "selected_audio",
)


def _read_inventory(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Inventory does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Inventory missing required fields: {missing}")
        rows = list(reader)
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        blank = [field for field in REQUIRED_FIELDS if not row.get(field, "").strip()]
        if blank:
            raise ValueError(f"{path}:{line_number} blank required fields: {blank}")
        audio_id = row["audio_id"].strip()
        audio_path = row["audio_path"].strip()
        if audio_id in seen_ids:
            raise ValueError(f"Duplicate audio_id in inventory: {audio_id}")
        if audio_path in seen_paths:
            raise ValueError(f"Duplicate audio_path in inventory: {audio_path}")
        seen_ids.add(audio_id)
        seen_paths.add(audio_path)
    return rows


def _write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fields))
        writer.writeheader()
        writer.writerows(rows)


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _selection_key(row: dict[str, str], seed: int) -> str:
    value = f"{seed}:{row['speaker_id']}:{row['audio_id']}:{row['audio_path']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _initial_issues(
    row: dict[str, str],
    *,
    audio_root: Path,
    minimum_duration: float,
    maximum_duration: float,
    expected_sample_rate: int,
    expected_channels: int,
) -> list[str]:
    issues: list[str] = []
    if not _is_true(row["is_valid"]):
        issues.append("inventory_invalid")
    if not (audio_root / row["audio_path"]).is_file():
        issues.append("missing_file")
    try:
        duration = float(row["duration_sec"])
    except ValueError:
        issues.append("invalid_duration")
    else:
        if not minimum_duration <= duration <= maximum_duration:
            issues.append("duration_out_of_range")
    if row["sample_rate"].strip() != str(expected_sample_rate):
        issues.append("unexpected_sample_rate")
    if row["num_channels"].strip() != str(expected_channels):
        issues.append("unexpected_channels")
    return issues


def _metadata_row(
    row: dict[str, str],
    *,
    normalized_speaker_id: str,
    split: str,
    checksum: str,
) -> dict[str, str]:
    return {
        "audio_id": row["audio_id"],
        "audio_path": row["audio_path"],
        "speaker_id": row["speaker_id"],
        "normalized_speaker_id": normalized_speaker_id,
        "split": split,
        "split_name": SPLIT_NAMES[split],
        "split_role": split.upper(),
        "protocol": "SVM_CLOSED_SET",
        "role": "ECAPA_RAW_EXPERIMENT",
        "duration_sec": row["duration_sec"],
        "sample_rate": row["sample_rate"],
        "num_channels": row["num_channels"],
        "checksum": checksum,
        "original_split": row.get("original_split", ""),
        "transcript": row.get("transcript", ""),
        "locale": row.get("locale", ""),
        "speaker_sex": row.get("speaker_sex", ""),
        "speaker_age": row.get("speaker_age", ""),
        "intent": row.get("intent", ""),
        "scenario_str": row.get("scenario_str", ""),
    }


def _report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "audio_id": row["audio_id"],
        "audio_path": row["audio_path"],
        "speaker_id": row["speaker_id"],
        "normalized_speaker_id": row.get("normalized_speaker_id", ""),
        "selected_split": row.get("selected_split", ""),
        "duration_sec": row["duration_sec"],
        "sample_rate": row["sample_rate"],
        "num_channels": row["num_channels"],
        "is_valid": row["is_valid"],
        "checksum": row.get("selected_checksum", ""),
        "status": row["status"],
        "issues": ";".join(row.get("issues", ())),
    }


def build_ecapa_raw_dataset(
    *,
    inventory_path: Path = Path("data/metadata/data_inventory.csv"),
    audio_root: Path = Path("data/audio"),
    output_root: Path = Path("data/datasets/ecapa_raw_v1"),
    split_counts: dict[str, int] | None = None,
    seed: int = 42,
    minimum_duration: float = 2.0,
    maximum_duration: float = 10.0,
    expected_sample_rate: int = 48000,
    expected_channels: int = 1,
) -> dict[str, Any]:
    """Create deterministic split metadata without copying or processing audio."""

    counts = dict(split_counts or SPLIT_COUNTS)
    if set(counts) != set(SPLIT_COUNTS) or any(value < 1 for value in counts.values()):
        raise ValueError("split_counts must contain positive train/validation/test values")
    if minimum_duration <= 0 or maximum_duration <= minimum_duration:
        raise ValueError("Invalid duration bounds")
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite dataset: {output_root}. Use a new version path."
        )

    rows = _read_inventory(inventory_path)
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        issues = _initial_issues(
            row,
            audio_root=audio_root,
            minimum_duration=minimum_duration,
            maximum_duration=maximum_duration,
            expected_sample_rate=expected_sample_rate,
            expected_channels=expected_channels,
        )
        row["issues"] = issues
        row["status"] = "rejected_metadata" if issues else "eligible"
        if not issues:
            eligible[row["speaker_id"]].append(row)

    required_per_speaker = sum(counts.values())
    selected_speakers = sorted(
        speaker for speaker, candidates in eligible.items() if len(candidates) >= required_per_speaker
    )
    if len(selected_speakers) < 2:
        raise ValueError(
            f"Need at least two speakers with {required_per_speaker} eligible audio each"
        )
    speaker_mapping = {
        speaker: f"ecapa_raw_spk_{index:04d}"
        for index, speaker in enumerate(selected_speakers, start=1)
    }
    for speaker, candidates in eligible.items():
        if speaker not in speaker_mapping:
            for row in candidates:
                row["status"] = "rejected_insufficient_speaker_audio"
                row["issues"] = ["speaker_below_required_count"]

    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in counts}
    for speaker in selected_speakers:
        candidates = sorted(eligible[speaker], key=lambda row: _selection_key(row, seed))
        chosen = candidates[:required_per_speaker]
        cursor = 0
        for split in ("train", "validation", "test"):
            stop = cursor + counts[split]
            for row in chosen[cursor:stop]:
                row["selected_split"] = split
                row["normalized_speaker_id"] = speaker_mapping[speaker]
                row["status"] = "selected"
                selected[split].append(row)
            cursor = stop
        for row in candidates[required_per_speaker:]:
            row["status"] = "not_selected_balance_cap"

    output_root.mkdir(parents=True)
    try:
        metadata_rows: dict[str, list[dict[str, str]]] = {}
        all_selected_checksums: set[str] = set()
        for split in ("train", "validation", "test"):
            output_rows: list[dict[str, str]] = []
            for row in sorted(
                selected[split],
                key=lambda value: (value["normalized_speaker_id"], value["audio_id"]),
            ):
                checksum = sha256_file(audio_root / row["audio_path"])
                if checksum in all_selected_checksums:
                    raise ValueError(f"Duplicate audio content selected: {row['audio_path']}")
                all_selected_checksums.add(checksum)
                row["selected_checksum"] = checksum
                output_rows.append(
                    _metadata_row(
                        row,
                        normalized_speaker_id=row["normalized_speaker_id"],
                        split=split,
                        checksum=checksum,
                    )
                )
            metadata_rows[split] = output_rows
            _write_csv(output_root / METADATA_FILES[split], METADATA_FIELDS, output_rows)

        mapping_rows = []
        for speaker in selected_speakers:
            source = eligible[speaker][0]
            mapping_rows.append(
                {
                    "speaker_id": speaker,
                    "normalized_speaker_id": speaker_mapping[speaker],
                    "speaker_sex": source.get("speaker_sex", ""),
                    "speaker_age": source.get("speaker_age", ""),
                    "eligible_audio": len(eligible[speaker]),
                    "selected_audio": required_per_speaker,
                }
            )
        _write_csv(output_root / "speaker_mapping.csv", MAPPING_FIELDS, mapping_rows)
        _write_csv(
            output_root / "selection_report.csv",
            REPORT_FIELDS,
            [_report_row(row) for row in sorted(rows, key=lambda value: value["audio_id"])],
        )

        split_manifest = {}
        for split, output_rows in metadata_rows.items():
            metadata_path = output_root / METADATA_FILES[split]
            durations = [float(row["duration_sec"]) for row in output_rows]
            split_manifest[split] = {
                "metadata": METADATA_FILES[split],
                "metadata_sha256": sha256_file(metadata_path),
                "num_audio": len(output_rows),
                "num_speakers": len({row["normalized_speaker_id"] for row in output_rows}),
                "audio_per_speaker": counts[split],
                "duration_seconds": round(sum(durations), 6),
                "minimum_duration_seconds": round(min(durations), 6),
                "maximum_duration_seconds": round(max(durations), 6),
            }
        manifest = {
            "dataset": "ecapa_raw_v1",
            "source_metadata": inventory_path.as_posix(),
            "audio_root": audio_root.as_posix(),
            "audio_storage": "references only; audio was not copied or processed",
            "seed": seed,
            "protocol": "closed-set speaker identification",
            "selection_criteria": {
                "inventory_is_valid": True,
                "minimum_duration_seconds": minimum_duration,
                "maximum_duration_seconds": maximum_duration,
                "sample_rate": expected_sample_rate,
                "channels": expected_channels,
                "required_audio_per_speaker": required_per_speaker,
            },
            "split_counts_per_speaker": counts,
            "selection": {
                "inventory_rows": len(rows),
                "metadata_eligible_rows": sum(len(values) for values in eligible.values()),
                "selected_speakers": len(selected_speakers),
                "selected_rows": sum(len(values) for values in selected.values()),
            },
            "splits": split_manifest,
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_root / "README.md").write_text(
            "# ECAPA raw dataset v1\n\n"
            "Metadata-only closed-set SID dataset referencing `data/audio`.\n\n"
            "- No audio was copied, resampled, trimmed, normalized, or augmented.\n"
            f"- {len(selected_speakers)} speakers; each has {counts['train']} train, "
            f"{counts['validation']} validation, and {counts['test']} test files.\n"
            "- Raw audio remains WAV mono 48 kHz and requires later ECAPA preprocessing.\n"
            "- Splits have no audio ID, path, or SHA-256 content overlap.\n"
            "- Recording-session IDs are unavailable; session-disjointness is unverified.\n",
            encoding="utf-8",
        )
    except Exception:
        for path in sorted(output_root.glob("*"), reverse=True):
            path.unlink()
        output_root.rmdir()
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/metadata/data_inventory.csv"),
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/audio"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/ecapa_raw_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_ecapa_raw_dataset(
        inventory_path=args.inventory,
        audio_root=args.audio_root,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(json.dumps(manifest["selection"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
