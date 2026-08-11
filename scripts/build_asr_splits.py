"""Tạo ASR validation/test theo project_split trong data inventory."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

from src.utils import sha256_file


OUTPUT_FIELDS = (
    "audio_id",
    "audio_path",
    "original_split",
    "project_split",
    "speaker_id",
    "reference_transcript",
    "source_intent",
)


def _column(fieldnames: list[str], *aliases: str, required: bool = True) -> str | None:
    lookup = {name.lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise ValueError(f"Missing column; expected one of: {', '.join(aliases)}")
    return None


def _is_valid(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "invalid", "error"}


def _load_inventory(path: Path) -> tuple[list[dict[str, str]], dict[str, str | None]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Inventory not found: {path}. Wait for Member 1 to export data_inventory.csv."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        columns = {
            "audio_id": _column(fields, "audio_id", "id", "file_id", required=False),
            "audio_path": _column(fields, "audio_path", "file_path", "path"),
            "original_split": _column(
                fields, "original_split", "split", "dataset_split"
            ),
            "project_split": _column(fields, "project_split"),
            "speaker_id": _column(
                fields, "speaker_id", "original_speaker_id", "speaker", required=False
            ),
            "transcript": _column(fields, "transcript", "text", "sentence"),
            "intent": _column(fields, "intent", "scenario", required=False),
            "valid": _column(fields, "is_valid", "valid", "audio_valid", required=False),
        }
        return list(reader), columns


def _resolve_audio_path(path_value: str, audio_root: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute() or audio_root is None or path.is_file():
        return path
    return audio_root / path


def _canonical_row(
    row: dict[str, str],
    columns: dict[str, str | None],
    audio_root: Path | None = None,
) -> dict[str, str]:
    path_value = row[str(columns["audio_path"])].strip()
    resolved_path = _resolve_audio_path(path_value, audio_root)
    audio_id_column = columns["audio_id"]
    return {
        "audio_id": row[str(audio_id_column)] if audio_id_column else Path(path_value).stem,
        "audio_path": resolved_path.as_posix(),
        "original_split": row[str(columns["original_split"])],
        "project_split": row[str(columns["project_split"])],
        "speaker_id": row[str(columns["speaker_id"])] if columns["speaker_id"] else "",
        "reference_transcript": row[str(columns["transcript"])],
        "source_intent": row[str(columns["intent"])] if columns["intent"] else "",
    }


def _select(
    rows: list[dict[str, str]],
    columns: dict[str, str | None],
    split_names: set[str],
    size: int | None,
    seed: int,
    audio_root: Path | None = None,
) -> list[dict[str, str]]:
    split_column = str(columns["project_split"])
    transcript_column = str(columns["transcript"])
    path_column = str(columns["audio_path"])
    valid_column = columns["valid"]
    candidates = []
    seen_paths: set[str] = set()
    for row in rows:
        split = row[split_column].strip().lower()
        transcript = row[transcript_column].strip()
        path_value = row[path_column].strip()
        resolved_path = _resolve_audio_path(path_value, audio_root)
        valid = not valid_column or _is_valid(row[str(valid_column)])
        path_exists = audio_root is None or resolved_path.is_file()
        if (
            split not in split_names
            or not transcript
            or not path_value
            or not valid
            or not path_exists
        ):
            continue
        normalized_path = str(resolved_path).casefold()
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        candidates.append(row)

    if size is not None and size < 1:
        raise ValueError("Split size must be positive or None for all usable rows")
    if size is not None and len(candidates) < size:
        raise ValueError(
            f"Project split {sorted(split_names)} has only {len(candidates)} usable rows; "
            f"requested {size}"
        )
    random.Random(seed).shuffle(candidates)
    selected = candidates if size is None else candidates[:size]
    return [
        _canonical_row(row, columns, audio_root) for row in selected
    ]


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    path: Path,
    inventory: Path,
    validation_path: Path,
    test_path: Path,
    validation: list[dict[str, str]],
    test: list[dict[str, str]],
    seed: int,
    validation_size: int | None,
    test_size: int | None,
) -> None:
    asr_component = {
        "source_inventory": {
            "path": inventory.as_posix(),
            "sha256": sha256_file(inventory),
        },
        "selection": {
            "validation_project_split": "VALIDATION",
            "test_project_split": "TEST",
            "use_all_usable_rows": (
                validation_size is None and test_size is None
            ),
            "requested_validation_size": validation_size,
            "requested_test_size": test_size,
            "unused_rows_included": False,
            "require_nonempty_transcript": True,
            "require_valid_flag": True,
            "require_existing_audio": True,
            "deduplicate_audio_path": True,
            "random_seed": seed,
            "validation_test_overlap": 0,
        },
        "datasets": {
            "asr_validation": {
                "path": validation_path.as_posix(),
                "row_count": len(validation),
                "sha256": sha256_file(validation_path),
            },
            "asr_test": {
                "path": test_path.as_posix(),
                "row_count": len(test),
                "sha256": sha256_file(test_path),
            },
        },
    }
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    if "components" not in manifest:
        previous_asr = {
            key: manifest[key]
            for key in ("source_inventory", "selection", "datasets")
            if key in manifest
        }
        manifest = {
            "manifest_schema_version": 1,
            "dataset_version": "v2",
            "created_at": manifest.get(
                "created_at",
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
            "random_seed": seed,
            "components": {"asr": previous_asr},
        }
    manifest["dataset_version"] = "v2"
    manifest["random_seed"] = seed
    manifest["freeze_status"] = "FROZEN"
    manifest.setdefault("components", {})["asr"] = asr_component
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("data_inventory.csv"))
    parser.add_argument(
        "--validation-size",
        type=int,
        default=None,
        help="Number of usable VALIDATION rows; default uses all usable rows",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Number of usable TEST rows; default uses all usable rows",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/audio"),
        help="Base directory for relative audio_path values",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/v2/metadata")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/v2/split_manifest.json"),
    )
    args = parser.parse_args()

    rows, columns = _load_inventory(args.inventory)
    validation = _select(
        rows,
        columns,
        {"validation"},
        args.validation_size,
        args.seed,
        args.audio_root,
    )
    test = _select(
        rows, columns, {"test"}, args.test_size, args.seed + 1, args.audio_root
    )

    validation_paths = {row["audio_path"].casefold() for row in validation}
    test_paths = {row["audio_path"].casefold() for row in test}
    overlap = validation_paths & test_paths
    if overlap:
        raise ValueError(f"ASR validation/test overlap detected: {sorted(overlap)[:5]}")

    validation_path = args.output_dir / "asr_validation.csv"
    test_path = args.output_dir / "asr_test.csv"
    _write(validation_path, validation)
    _write(test_path, test)
    _write_manifest(
        args.manifest,
        args.inventory,
        validation_path,
        test_path,
        validation,
        test,
        args.seed,
        args.validation_size,
        args.test_size,
    )
    print(
        f"Created {len(validation)} validation and {len(test)} test rows "
        f"with seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
