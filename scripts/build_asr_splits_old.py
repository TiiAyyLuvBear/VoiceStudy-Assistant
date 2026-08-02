"""Tạo ASR validation/test từ inventory official split của Speech-MASSIVE."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


OUTPUT_FIELDS = (
    "audio_id",
    "audio_path",
    "original_split",
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
            "split": _column(fields, "original_split", "split", "dataset_split"),
            "speaker_id": _column(
                fields, "speaker_id", "original_speaker_id", "speaker", required=False
            ),
            "transcript": _column(fields, "transcript", "text", "sentence"),
            "intent": _column(fields, "intent", "scenario", required=False),
            "valid": _column(fields, "is_valid", "valid", "audio_valid", required=False),
        }
        return list(reader), columns


def _canonical_row(row: dict[str, str], columns: dict[str, str | None]) -> dict[str, str]:
    path_value = row[str(columns["audio_path"])]
    audio_id_column = columns["audio_id"]
    return {
        "audio_id": row[str(audio_id_column)] if audio_id_column else Path(path_value).stem,
        "audio_path": path_value,
        "original_split": row[str(columns["split"])],
        "speaker_id": row[str(columns["speaker_id"])] if columns["speaker_id"] else "",
        "reference_transcript": row[str(columns["transcript"])],
        "source_intent": row[str(columns["intent"])] if columns["intent"] else "",
    }


def _select(
    rows: list[dict[str, str]],
    columns: dict[str, str | None],
    split_names: set[str],
    size: int,
    seed: int,
) -> list[dict[str, str]]:
    split_column = str(columns["split"])
    transcript_column = str(columns["transcript"])
    path_column = str(columns["audio_path"])
    valid_column = columns["valid"]
    candidates = []
    seen_paths: set[str] = set()
    for row in rows:
        split = row[split_column].strip().lower()
        transcript = row[transcript_column].strip()
        path_value = row[path_column].strip()
        valid = not valid_column or _is_valid(row[str(valid_column)])
        if split not in split_names or not transcript or not path_value or not valid:
            continue
        normalized_path = str(Path(path_value)).casefold()
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        candidates.append(row)

    if len(candidates) < size:
        raise ValueError(
            f"Split {sorted(split_names)} has only {len(candidates)} usable rows; "
            f"requested {size}"
        )
    random.Random(seed).shuffle(candidates)
    return [_canonical_row(row, columns) for row in candidates[:size]]


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("data_inventory.csv"))
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/metadata")
    )
    args = parser.parse_args()

    rows, columns = _load_inventory(args.inventory)
    validation = _select(
        rows, columns, {"validation", "valid", "dev"}, args.validation_size, args.seed
    )
    test = _select(rows, columns, {"test"}, args.test_size, args.seed + 1)

    validation_paths = {row["audio_path"].casefold() for row in validation}
    test_paths = {row["audio_path"].casefold() for row in test}
    overlap = validation_paths & test_paths
    if overlap:
        raise ValueError(f"ASR validation/test overlap detected: {sorted(overlap)[:5]}")

    _write(args.output_dir / "asr_validation.csv", validation)
    _write(args.output_dir / "asr_test.csv", test)
    print(
        f"Created {len(validation)} validation and {len(test)} test rows "
        f"with seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
