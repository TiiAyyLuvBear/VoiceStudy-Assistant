"""Tạo checksum manifest để đóng băng dữ liệu Thành viên 2 phiên bản v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from scripts.check_command_duplicates import find_duplicates


COMMAND_FILES = (
    Path("data/metadata/command_development.csv"),
    Path("data/metadata/command_validation.csv"),
    Path("data/metadata/command_test.csv"),
)
ASR_FILES = (
    Path("data/metadata/asr_validation.csv"),
    Path("data/metadata/asr_test.csv"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_info(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    intents = Counter(row.get("intent", "") for row in rows if row.get("intent"))
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "row_count": len(rows),
        "intent_distribution": dict(sorted(intents.items())),
    }


def build_manifest() -> dict[str, object]:
    missing = [path for path in COMMAND_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing command datasets: {missing}")
    duplicates = find_duplicates(list(COMMAND_FILES))
    if duplicates:
        raise ValueError(f"Cannot freeze datasets with duplicates: {len(duplicates)}")

    asr_status = {
        path.stem: (_csv_info(path) if path.is_file() else {"status": "pending_member1_data"})
        for path in ASR_FILES
    }
    audio_manifest = Path("data/commands/command_audio_manifest.csv")
    audio_status: dict[str, object] = {"status": "not_prepared"}
    if audio_manifest.is_file():
        with audio_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        audio_status = {
            "path": audio_manifest.as_posix(),
            "row_count": len(rows),
            "recorded_count": sum(row.get("status") == "recorded" for row in rows),
            "status": "complete" if rows and all(row.get("status") == "recorded" for row in rows) else "pending_recording",
        }

    return {
        "version": "member2-v1",
        "frozen_on": date.today().isoformat(),
        "random_seed": 42,
        "reference_date": "2026-07-28",
        "rules": {
            "command_split_overlap": False,
            "normalized_duplicate_count": 0,
            "test_used_for_rule_tuning": False,
            "command_audio_allowed_for_speaker_training": False,
        },
        "command_datasets": {
            path.stem: _csv_info(path) for path in COMMAND_FILES
        },
        "asr_datasets": asr_status,
        "command_audio": audio_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/member2_split_manifest.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest()
    if args.output.is_file() and not args.force:
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("command_datasets") != manifest["command_datasets"]:
            raise RuntimeError(
                "Frozen command datasets changed. Review changes before using --force."
            )
        print(f"Frozen command checksums unchanged: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created freeze manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
