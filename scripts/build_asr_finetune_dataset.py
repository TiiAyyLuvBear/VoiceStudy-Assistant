"""Build a leakage-safe ASR fine-tuning dataset from the full audio inventory.

All usable source rows are assigned exactly once. Existing v1/v2 artifacts are
never modified: the default output is the new ``data/processed/v3`` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.utils import canonical_csv_sha256, sha256_file


DATASET_FIELDS = (
    "audio_id",
    "audio_path",
    "transcript",
    "speaker_id",
    "locale",
    "source_original_split",
    "source_project_split",
    "duration_sec",
    "sample_rate",
    "num_channels",
    "source_intent",
    "audio_sha256",
)

REJECTED_FIELDS = (
    "audio_id",
    "audio_path",
    "speaker_id",
    "source_original_split",
    "source_project_split",
    "source_is_valid",
    "rejection_reason",
    "source_transcript",
)

PROJECT_SPLITS = {
    "train": frozenset({"SVM", "UNUSED"}),
    "validation": frozenset({"VALIDATION"}),
    "test": frozenset({"TEST"}),
}

REQUIRED_INVENTORY_FIELDS = {
    "audio_id",
    "audio_path",
    "original_split",
    "project_split",
    "speaker_id",
    "transcript",
    "is_valid",
}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or ())


def _write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_transcript(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "valid", "ok"}


def _resolve_audio_path(audio_root: Path, value: str) -> tuple[Path, str]:
    source = Path(value.strip())
    candidate = source if source.is_absolute() else audio_root / source
    resolved_root = audio_root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Audio path escapes audio root: {value}") from exc
    return resolved_candidate, candidate.as_posix()


def _invalid_reasons(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    rows, fields = _read_csv(path)
    if not {"audio_id", "reason"}.issubset(fields):
        raise ValueError(f"Invalid-audio report has an unsupported schema: {path}")
    return {
        row["audio_id"].strip(): row["reason"].strip()
        for row in rows
        if row["audio_id"].strip()
    }


def _target_split(project_split: str) -> str | None:
    normalized = project_split.strip().upper()
    for split_name, project_values in PROJECT_SPLITS.items():
        if normalized in project_values:
            return split_name
    return None


def _duration_hours(rows: list[dict[str, str]]) -> float:
    seconds = sum(float(row["duration_sec"] or 0) for row in rows)
    return round(seconds / 3600, 6)


def _assert_unique_and_disjoint(splits: dict[str, list[dict[str, str]]]) -> None:
    all_ids: set[str] = set()
    all_paths: set[str] = set()
    speaker_sets: dict[str, set[str]] = {}
    for split_name, rows in splits.items():
        ids = {row["audio_id"] for row in rows}
        paths = {row["audio_path"].casefold() for row in rows}
        if len(ids) != len(rows):
            raise ValueError(f"Duplicate audio_id inside {split_name}")
        if len(paths) != len(rows):
            raise ValueError(f"Duplicate audio_path inside {split_name}")
        if all_ids & ids or all_paths & paths:
            raise ValueError(f"Audio overlap detected at split {split_name}")
        all_ids.update(ids)
        all_paths.update(paths)
        speaker_sets[split_name] = {
            row["speaker_id"] for row in rows if row["speaker_id"]
        }

    split_names = tuple(splits)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = speaker_sets[left] & speaker_sets[right]
            if overlap:
                raise ValueError(
                    f"Speaker leakage between {left} and {right}: {sorted(overlap)[:5]}"
                )


def build_dataset(
    *,
    inventory_path: Path,
    audio_root: Path,
    invalid_audio_path: Path | None,
    output_root: Path,
) -> dict[str, object]:
    """Build all ASR splits and return the written manifest."""

    existing_manifest = output_root / "asr_finetune_manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("freeze_status") == "FROZEN":
            raise RuntimeError(
                f"Refusing to overwrite frozen ASR dataset: {existing_manifest}. "
                "Create a new dataset version instead."
            )

    inventory, fields = _read_csv(inventory_path)
    missing_fields = REQUIRED_INVENTORY_FIELDS - set(fields)
    if missing_fields:
        raise ValueError(
            "Inventory is missing required fields: " + ", ".join(sorted(missing_fields))
        )

    reasons = _invalid_reasons(invalid_audio_path)
    splits: dict[str, list[dict[str, str]]] = {
        split_name: [] for split_name in PROJECT_SPLITS
    }
    rejected: list[dict[str, str]] = []
    seen_source_ids: set[str] = set()
    seen_source_paths: set[str] = set()

    for source_row in inventory:
        audio_id = source_row["audio_id"].strip()
        source_path = source_row["audio_path"].strip()
        if not audio_id or audio_id in seen_source_ids:
            raise ValueError(f"Missing or duplicate source audio_id: {audio_id!r}")
        normalized_source_path = source_path.casefold()
        if not source_path or normalized_source_path in seen_source_paths:
            raise ValueError(f"Missing or duplicate source audio_path: {source_path!r}")
        seen_source_ids.add(audio_id)
        seen_source_paths.add(normalized_source_path)

        target = _target_split(source_row["project_split"])
        transcript = _normalize_transcript(source_row["transcript"])
        resolved_path, output_audio_path = _resolve_audio_path(audio_root, source_path)

        rejection_reasons: list[str] = []
        if not _truthy(source_row["is_valid"]):
            rejection_reasons.append(reasons.get(audio_id, "source_marked_invalid"))
        if not transcript:
            rejection_reasons.append("empty_transcript")
        if target is None:
            rejection_reasons.append("unsupported_project_split")
        if not resolved_path.is_file():
            rejection_reasons.append("missing_audio_file")

        if rejection_reasons:
            rejected.append(
                {
                    "audio_id": audio_id,
                    "audio_path": output_audio_path,
                    "speaker_id": source_row["speaker_id"].strip(),
                    "source_original_split": source_row["original_split"].strip(),
                    "source_project_split": source_row["project_split"].strip(),
                    "source_is_valid": source_row["is_valid"].strip(),
                    "rejection_reason": ";".join(dict.fromkeys(rejection_reasons)),
                    "source_transcript": source_row["transcript"],
                }
            )
            continue

        assert target is not None
        splits[target].append(
            {
                "audio_id": audio_id,
                "audio_path": output_audio_path,
                "transcript": transcript,
                "speaker_id": source_row["speaker_id"].strip(),
                "locale": source_row.get("locale", "").strip(),
                "source_original_split": source_row["original_split"].strip(),
                "source_project_split": source_row["project_split"].strip(),
                "duration_sec": source_row.get("duration_sec", "").strip(),
                "sample_rate": source_row.get("sample_rate", "").strip(),
                "num_channels": source_row.get("num_channels", "").strip(),
                "source_intent": source_row.get("intent", "").strip(),
                "audio_sha256": sha256_file(resolved_path),
            }
        )

    for rows in splits.values():
        rows.sort(key=lambda row: (row["speaker_id"], row["audio_id"]))
    rejected.sort(key=lambda row: row["audio_id"])
    _assert_unique_and_disjoint(splits)

    accepted_count = sum(len(rows) for rows in splits.values())
    if accepted_count + len(rejected) != len(inventory):
        raise AssertionError("Not every source row was accounted for")

    metadata_root = output_root / "metadata"
    dataset_paths: dict[str, Path] = {}
    for split_name, rows in splits.items():
        path = metadata_root / f"asr_finetune_{split_name}.csv"
        _write_csv(path, DATASET_FIELDS, rows)
        dataset_paths[split_name] = path
    rejected_path = metadata_root / "asr_finetune_rejected.csv"
    _write_csv(rejected_path, REJECTED_FIELDS, rejected)

    speaker_sets = {
        split_name: {row["speaker_id"] for row in rows if row["speaker_id"]}
        for split_name, rows in splits.items()
    }
    overlap_counts = {
        "train_validation": len(speaker_sets["train"] & speaker_sets["validation"]),
        "train_test": len(speaker_sets["train"] & speaker_sets["test"]),
        "validation_test": len(
            speaker_sets["validation"] & speaker_sets["test"]
        ),
    }
    rejection_counts = Counter(
        reason
        for row in rejected
        for reason in row["rejection_reason"].split(";")
        if reason
    )

    def file_record(path: Path, rows: list[dict[str, str]]) -> dict[str, object]:
        return {
            "path": path.as_posix(),
            "row_count": len(rows),
            "speaker_count": len(
                {row["speaker_id"] for row in rows if row.get("speaker_id")}
            ),
            "duration_hours": _duration_hours(rows),
            "sha256": sha256_file(path),
            "canonical_csv_sha256": canonical_csv_sha256(path),
        }

    manifest: dict[str, object] = {
        "manifest_schema_version": 1,
        "dataset_version": "v3",
        "component": "asr_finetune",
        "freeze_status": "DEVELOPMENT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "inventory_path": inventory_path.as_posix(),
            "inventory_sha256": sha256_file(inventory_path),
            "audio_root": audio_root.as_posix(),
            "source_row_count": len(inventory),
            "accepted_row_count": accepted_count,
            "rejected_row_count": len(rejected),
            "all_source_rows_accounted_for": True,
        },
        "selection_policy": {
            "train_project_splits": sorted(PROJECT_SPLITS["train"]),
            "validation_project_splits": sorted(PROJECT_SPLITS["validation"]),
            "test_project_splits": sorted(PROJECT_SPLITS["test"]),
            "use_every_usable_source_row": True,
            "require_source_valid_flag": True,
            "require_existing_audio": True,
            "require_nonempty_transcript": True,
            "transcript_normalization": "Unicode NFC plus whitespace collapse",
            "speaker_disjoint": True,
            "random_sampling": False,
        },
        "invariants": {
            "audio_overlap_across_splits": 0,
            "speaker_overlap": overlap_counts,
            "accepted_plus_rejected_equals_source": accepted_count
            + len(rejected)
            == len(inventory),
        },
        "datasets": {
            split_name: file_record(dataset_paths[split_name], rows)
            for split_name, rows in splits.items()
        },
        "rejected": {
            "path": rejected_path.as_posix(),
            "row_count": len(rejected),
            "reason_counts": dict(sorted(rejection_counts.items())),
            "sha256": sha256_file(rejected_path),
            "canonical_csv_sha256": canonical_csv_sha256(rejected_path),
        },
        "scope_notes": [
            "Only data/audio rows represented by data_inventory.csv are source data.",
            "data/commands/audio remains an external real-command evaluation set.",
            "data/processed and data/samples are excluded because they contain derived copies.",
            "Do not inspect or tune against the v3 test split before model/configuration lock.",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "asr_finetune_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
        "--invalid-audio",
        type=Path,
        default=Path("data/metadata/invalid_audio.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/v3")
    )
    args = parser.parse_args()

    manifest = build_dataset(
        inventory_path=args.inventory,
        audio_root=args.audio_root,
        invalid_audio_path=args.invalid_audio,
        output_root=args.output_root,
    )
    datasets = manifest["datasets"]
    source = manifest["source"]
    assert isinstance(datasets, dict) and isinstance(source, dict)
    print(
        "Created ASR fine-tune v3: "
        f"train={datasets['train']['row_count']}, "
        f"validation={datasets['validation']['row_count']}, "
        f"test={datasets['test']['row_count']}, "
        f"rejected={source['rejected_row_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
