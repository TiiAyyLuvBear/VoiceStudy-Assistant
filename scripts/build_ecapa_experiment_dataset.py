"""Build ASR-disjoint ECAPA datasets for SID and speaker verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_ecapa_raw_dataset import (
    REPORT_FIELDS,
    _initial_issues,
    _read_inventory,
    _report_row,
    _selection_key,
    _write_csv,
)
from src.utils.files import sha256_file


SID_COUNTS = {"train": 20, "validation": 5, "test": 5}
PROTOCOL_FILES = {
    "sid_train": "sid_train.csv",
    "sid_validation": "sid_validation.csv",
    "sid_test": "sid_test.csv",
    "sv_validation_enrollment": "sv_validation_enrollment.csv",
    "sv_validation_query": "sv_validation_query.csv",
    "sv_validation_unknown": "sv_validation_unknown.csv",
    "sv_test_enrollment": "sv_test_enrollment.csv",
    "sv_test_query": "sv_test_query.csv",
    "sv_test_unknown": "sv_test_unknown.csv",
}
PROTOCOL_DEFINITIONS = {
    "sid_train": {
        "task": "speaker_identification",
        "split": "train",
        "split_name": "svm_closed_set_train",
        "split_role": "TRAIN",
        "protocol": "SVM_CLOSED_SET",
        "role": "SVM_EXPERIMENTAL",
    },
    "sid_validation": {
        "task": "speaker_identification",
        "split": "validation",
        "split_name": "svm_closed_set_validation",
        "split_role": "VALIDATION",
        "protocol": "SVM_CLOSED_SET",
        "role": "SVM_EXPERIMENTAL",
    },
    "sid_test": {
        "task": "speaker_identification",
        "split": "test",
        "split_name": "svm_closed_set_test",
        "split_role": "TEST",
        "protocol": "SVM_CLOSED_SET",
        "role": "SVM_EXPERIMENTAL",
    },
    "sv_validation_enrollment": {
        "task": "speaker_verification",
        "split": "validation",
        "split_name": "cosine_validation_enrollment",
        "split_role": "ENROLLMENT",
        "protocol": "COSINE_VALIDATION",
        "role": "ENROLLED",
    },
    "sv_validation_query": {
        "task": "speaker_verification",
        "split": "validation",
        "split_name": "cosine_validation_query",
        "split_role": "QUERY",
        "protocol": "COSINE_VALIDATION",
        "role": "ENROLLED",
    },
    "sv_validation_unknown": {
        "task": "speaker_verification",
        "split": "validation",
        "split_name": "cosine_validation_unknown",
        "split_role": "UNKNOWN",
        "protocol": "COSINE_VALIDATION",
        "role": "UNKNOWN",
    },
    "sv_test_enrollment": {
        "task": "speaker_verification",
        "split": "test",
        "split_name": "cosine_test_enrollment",
        "split_role": "ENROLLMENT",
        "protocol": "COSINE_TEST",
        "role": "ENROLLED",
    },
    "sv_test_query": {
        "task": "speaker_verification",
        "split": "test",
        "split_name": "cosine_test_query",
        "split_role": "QUERY",
        "protocol": "COSINE_TEST",
        "role": "ENROLLED",
    },
    "sv_test_unknown": {
        "task": "speaker_verification",
        "split": "test",
        "split_name": "cosine_test_unknown",
        "split_role": "UNKNOWN",
        "protocol": "COSINE_TEST",
        "role": "UNKNOWN",
    },
}
METADATA_FIELDS = (
    "audio_id",
    "audio_path",
    "speaker_id",
    "normalized_speaker_id",
    "task",
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
MAPPING_FIELDS = (
    "speaker_id",
    "normalized_speaker_id",
    "task_group",
    "speaker_sex",
    "speaker_age",
    "eligible_audio",
    "selected_audio",
)


def _normalized_path(value: str) -> str:
    path = Path(value)
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "audio":
        path = Path(*parts[2:])
    return path.as_posix().casefold()


def _asr_reserved_paths(paths: tuple[Path, ...]) -> set[str]:
    reserved: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"ASR metadata does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "audio_path" not in (reader.fieldnames or ()):
                raise ValueError(f"ASR metadata missing audio_path: {path}")
            reserved.update(
                _normalized_path(row["audio_path"])
                for row in reader
                if row.get("audio_path", "").strip()
            )
    return reserved


def _speaker_key(speaker: str, count: int, seed: int) -> tuple[int, str]:
    digest = hashlib.sha256(f"{seed}:{speaker}".encode("utf-8")).hexdigest()
    return -count, digest


def _metadata_row(
    row: dict[str, Any],
    *,
    protocol_key: str,
    checksum: str,
) -> dict[str, str]:
    definition = PROTOCOL_DEFINITIONS[protocol_key]
    return {
        "audio_id": row["audio_id"],
        "audio_path": row["audio_path"],
        "speaker_id": row["speaker_id"],
        "normalized_speaker_id": row["normalized_speaker_id"],
        **definition,
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


def build_ecapa_dataset(
    *,
    inventory_path: Path = Path("data/metadata/data_inventory.csv"),
    audio_root: Path = Path("data/audio"),
    asr_metadata_paths: tuple[Path, ...] = (
        Path("data/metadata/asr_validation.csv"),
        Path("data/metadata/asr_test.csv"),
    ),
    output_root: Path = Path("data/processed/ecapa_experiment_v1"),
    seed: int = 42,
    sid_speaker_count: int = 20,
    sid_counts: dict[str, int] | None = None,
    sv_enrolled_speakers_per_eval: int = 3,
    sv_unknown_speakers_per_eval: int = 2,
    sv_enrollment_audio: int = 5,
    sv_query_audio: int = 5,
    sv_unknown_audio: int = 10,
    minimum_duration: float = 2.0,
    maximum_duration: float = 10.0,
) -> dict[str, Any]:
    """Create metadata-only, speaker-disjoint SID/SV experiment protocols."""

    sid_split_counts = dict(sid_counts or SID_COUNTS)
    if set(sid_split_counts) != set(SID_COUNTS):
        raise ValueError("sid_counts must contain train/validation/test")
    numeric = (
        sid_speaker_count,
        *sid_split_counts.values(),
        sv_enrolled_speakers_per_eval,
        sv_unknown_speakers_per_eval,
        sv_enrollment_audio,
        sv_query_audio,
        sv_unknown_audio,
    )
    if any(value < 1 for value in numeric):
        raise ValueError("All speaker and audio counts must be positive")
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite dataset: {output_root}. Build to a temporary path first."
        )

    rows = _read_inventory(inventory_path)
    reserved = _asr_reserved_paths(asr_metadata_paths)
    eligible: dict[str, list[dict[str, Any]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        issues = _initial_issues(
            row,
            audio_root=audio_root,
            minimum_duration=minimum_duration,
            maximum_duration=maximum_duration,
            expected_sample_rate=48000,
            expected_channels=1,
        )
        if _normalized_path(row["audio_path"]) in reserved:
            issues.append("reserved_for_asr")
        row["issues"] = issues
        row["status"] = "rejected_metadata" if issues else "eligible"
        if not issues:
            grouped.setdefault(row["speaker_id"], []).append(row)
    eligible = grouped

    sid_required = sum(sid_split_counts.values())
    sid_candidates = [
        speaker for speaker, values in eligible.items() if len(values) >= sid_required
    ]
    sid_candidates.sort(key=lambda speaker: _speaker_key(speaker, len(eligible[speaker]), seed))
    if len(sid_candidates) < sid_speaker_count:
        raise ValueError(
            f"Need {sid_speaker_count} SID speakers with {sid_required} audio; "
            f"found {len(sid_candidates)}"
        )
    sid_speakers = sid_candidates[:sid_speaker_count]

    sv_required = max(sv_enrollment_audio + sv_query_audio, sv_unknown_audio)
    remaining = [
        speaker
        for speaker, values in eligible.items()
        if speaker not in sid_speakers and len(values) >= sv_required
    ]
    remaining.sort(key=lambda speaker: _speaker_key(speaker, len(eligible[speaker]), seed + 1))
    sv_speakers_needed = 2 * (
        sv_enrolled_speakers_per_eval + sv_unknown_speakers_per_eval
    )
    if len(remaining) < sv_speakers_needed:
        raise ValueError(
            f"Need {sv_speakers_needed} SV-only speakers with {sv_required} audio; "
            f"found {len(remaining)}"
        )
    sv_speakers = remaining[:sv_speakers_needed]
    cursor = 0
    groups: dict[str, list[str]] = {"sid": sid_speakers}
    for evaluation in ("validation", "test"):
        stop = cursor + sv_enrolled_speakers_per_eval
        groups[f"sv_{evaluation}_enrolled"] = sv_speakers[cursor:stop]
        cursor = stop
        stop = cursor + sv_unknown_speakers_per_eval
        groups[f"sv_{evaluation}_unknown"] = sv_speakers[cursor:stop]
        cursor = stop

    chosen_speakers = [speaker for values in groups.values() for speaker in values]
    speaker_mapping = {
        speaker: f"ecapa_spk_{index:04d}"
        for index, speaker in enumerate(chosen_speakers, start=1)
    }
    selected: dict[str, list[dict[str, Any]]] = {
        key: [] for key in PROTOCOL_FILES
    }

    for speaker in sid_speakers:
        candidates = sorted(eligible[speaker], key=lambda row: _selection_key(row, seed))
        chosen = candidates[:sid_required]
        offset = 0
        for split in ("train", "validation", "test"):
            protocol_key = f"sid_{split}"
            stop = offset + sid_split_counts[split]
            for row in chosen[offset:stop]:
                row["normalized_speaker_id"] = speaker_mapping[speaker]
                row["selected_split"] = protocol_key
                row["status"] = "selected"
                selected[protocol_key].append(row)
            offset = stop
        for row in candidates[sid_required:]:
            row["status"] = "not_selected_balance_cap"

    for evaluation in ("validation", "test"):
        for speaker in groups[f"sv_{evaluation}_enrolled"]:
            candidates = sorted(
                eligible[speaker], key=lambda row: _selection_key(row, seed + 2)
            )
            for protocol_key, start, count in (
                (f"sv_{evaluation}_enrollment", 0, sv_enrollment_audio),
                (f"sv_{evaluation}_query", sv_enrollment_audio, sv_query_audio),
            ):
                for row in candidates[start : start + count]:
                    row["normalized_speaker_id"] = speaker_mapping[speaker]
                    row["selected_split"] = protocol_key
                    row["status"] = "selected"
                    selected[protocol_key].append(row)
            for row in candidates[sv_enrollment_audio + sv_query_audio :]:
                row["status"] = "not_selected_balance_cap"
        for speaker in groups[f"sv_{evaluation}_unknown"]:
            candidates = sorted(
                eligible[speaker], key=lambda row: _selection_key(row, seed + 3)
            )
            protocol_key = f"sv_{evaluation}_unknown"
            for row in candidates[:sv_unknown_audio]:
                row["normalized_speaker_id"] = speaker_mapping[speaker]
                row["selected_split"] = protocol_key
                row["status"] = "selected"
                selected[protocol_key].append(row)
            for row in candidates[sv_unknown_audio:]:
                row["status"] = "not_selected_balance_cap"

    selected_set = set(chosen_speakers)
    for speaker, values in eligible.items():
        if speaker not in selected_set:
            for row in values:
                row["status"] = "not_selected_speaker"

    output_root.mkdir(parents=True)
    try:
        metadata_dir = output_root / "metadata"
        metadata_rows: dict[str, list[dict[str, str]]] = {}
        checksums: set[str] = set()
        selected_paths: set[str] = set()
        for protocol_key, filename in PROTOCOL_FILES.items():
            output_rows = []
            for row in sorted(
                selected[protocol_key],
                key=lambda value: (value["normalized_speaker_id"], value["audio_id"]),
            ):
                normalized_path = _normalized_path(row["audio_path"])
                if normalized_path in reserved:
                    raise ValueError(f"ASR leakage selected: {row['audio_path']}")
                if normalized_path in selected_paths:
                    raise ValueError(f"Duplicate selected path: {row['audio_path']}")
                checksum = sha256_file(audio_root / row["audio_path"])
                if checksum in checksums:
                    raise ValueError(f"Duplicate selected content: {row['audio_path']}")
                selected_paths.add(normalized_path)
                checksums.add(checksum)
                row["selected_checksum"] = checksum
                output_rows.append(
                    _metadata_row(row, protocol_key=protocol_key, checksum=checksum)
                )
            metadata_rows[protocol_key] = output_rows
            _write_csv(metadata_dir / filename, METADATA_FIELDS, output_rows)

        mapping_rows = []
        group_by_speaker = {
            speaker: group for group, speakers in groups.items() for speaker in speakers
        }
        for speaker in chosen_speakers:
            source = eligible[speaker][0]
            selected_count = sum(
                row.get("speaker_id") == speaker
                for values in metadata_rows.values()
                for row in values
            )
            mapping_rows.append(
                {
                    "speaker_id": speaker,
                    "normalized_speaker_id": speaker_mapping[speaker],
                    "task_group": group_by_speaker[speaker],
                    "speaker_sex": source.get("speaker_sex", ""),
                    "speaker_age": source.get("speaker_age", ""),
                    "eligible_audio": len(eligible[speaker]),
                    "selected_audio": selected_count,
                }
            )
        _write_csv(output_root / "speaker_mapping.csv", MAPPING_FIELDS, mapping_rows)
        _write_csv(
            output_root / "selection_report.csv",
            REPORT_FIELDS,
            [_report_row(row) for row in sorted(rows, key=lambda value: value["audio_id"])],
        )

        protocol_manifest = {}
        for protocol_key, output_rows in metadata_rows.items():
            metadata_path = metadata_dir / PROTOCOL_FILES[protocol_key]
            protocol_manifest[protocol_key] = {
                "metadata": f"metadata/{PROTOCOL_FILES[protocol_key]}",
                "metadata_sha256": sha256_file(metadata_path),
                "num_audio": len(output_rows),
                "num_speakers": len(
                    {row["normalized_speaker_id"] for row in output_rows}
                ),
                "duration_seconds": round(
                    sum(float(row["duration_sec"]) for row in output_rows), 6
                ),
            }
        manifest = {
            "dataset": "ecapa_experiment_v1",
            "source_metadata": inventory_path.as_posix(),
            "audio_root": audio_root.as_posix(),
            "audio_storage": "references only; audio was not copied or preprocessed",
            "asr_exclusions": [path.as_posix() for path in asr_metadata_paths],
            "asr_reserved_audio": len(reserved),
            "seed": seed,
            "quality_filter": {
                "inventory_is_valid": True,
                "minimum_duration_seconds": minimum_duration,
                "maximum_duration_seconds": maximum_duration,
                "sample_rate": 48000,
                "channels": 1,
            },
            "speaker_groups": {group: len(speakers) for group, speakers in groups.items()},
            "sid_counts_per_speaker": sid_split_counts,
            "sv_counts": {
                "enrollment_audio_per_enrolled_speaker": sv_enrollment_audio,
                "query_audio_per_enrolled_speaker": sv_query_audio,
                "unknown_audio_per_unknown_speaker": sv_unknown_audio,
            },
            "selection": {
                "inventory_rows": len(rows),
                "eligible_rows_after_asr_exclusion": sum(
                    len(values) for values in eligible.values()
                ),
                "selected_speakers": len(chosen_speakers),
                "selected_audio": sum(len(values) for values in selected.values()),
            },
            "protocols": protocol_manifest,
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_root / "README.md").write_text(
            "# ECAPA experiment v1\n\n"
            "Metadata-only ECAPA experiment referencing raw `data/audio`.\n\n"
            "- ASR validation/test audio excluded globally.\n"
            "- SID: 20 speakers, 400 train, 100 validation, 100 test.\n"
            "- SV validation: 3 enrolled + 2 unknown speakers, 50 audio.\n"
            "- SV test: 3 new enrolled + 2 new unknown speakers, 50 audio.\n"
            "- SID and every SV group are speaker-disjoint.\n"
            "- No audio was copied, resampled, trimmed, normalized, or augmented.\n"
            "- Raw audio remains mono 48 kHz; preprocessing comes later.\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output_root)
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
        default=Path("data/processed/ecapa_experiment_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_ecapa_dataset(
        inventory_path=args.inventory,
        audio_root=args.audio_root,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(json.dumps(manifest["selection"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
