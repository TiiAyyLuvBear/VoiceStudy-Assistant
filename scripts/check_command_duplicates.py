"""Kiểm tra câu trùng trong và giữa ba command split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.nlu.text_normalizer import normalize_text


DEFAULT_FILES = (
    Path("data/metadata/command_development.csv"),
    Path("data/metadata/command_validation.csv"),
    Path("data/metadata/command_test.csv"),
)


def find_duplicates(paths: list[Path]) -> list[dict[str, object]]:
    occurrences: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Command dataset does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"command_id", "transcript", "intent", "split"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for row in reader:
                normalized = normalize_text(row["transcript"])
                if not normalized:
                    raise ValueError(f"{path}: empty transcript at {row['command_id']}")
                occurrences[normalized].append(
                    {
                        "command_id": row["command_id"],
                        "split": row["split"],
                        "transcript": row["transcript"],
                        "source_file": str(path),
                    }
                )

    return [
        {"normalized_transcript": text, "occurrences": rows}
        for text, rows in sorted(occurrences.items())
        if len(rows) > 1
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, default=list(DEFAULT_FILES))
    args = parser.parse_args()

    duplicates = find_duplicates(args.files)
    result = {
        "files_checked": [str(path) for path in args.files],
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())
