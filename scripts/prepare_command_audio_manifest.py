"""Tạo hàng đợi thu âm từ command validation và command test."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
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


def build_rows(command_files: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in command_files:
        if not path.is_file():
            raise FileNotFoundError(f"Command dataset does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for command in csv.DictReader(stream):
                split = command["split"]
                if split not in {"validation", "test"}:
                    continue
                rows.append(
                    {
                        "recording_id": f"REC_{command['command_id']}",
                        "command_id": command["command_id"],
                        "split": split,
                        "expected_transcript": command["transcript"],
                        "intent": command["intent"],
                        "speaker_id": "",
                        "audio_path": "",
                        "sample_rate": "",
                        "channels": "",
                        "duration_sec": "",
                        "recorded_at": "",
                        "recording_device": "",
                        "status": "pending",
                        "notes": "Do not use for Speaker model training",
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commands",
        nargs="+",
        type=Path,
        default=[
            Path("data/metadata/command_validation.csv"),
            Path("data/metadata/command_test.csv"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/commands/command_audio_manifest.csv"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Manifest already exists: {args.output}. Use --force only before recording."
        )
    rows = build_rows(args.commands)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {args.output} with {len(rows)} recording prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
