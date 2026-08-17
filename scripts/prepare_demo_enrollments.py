"""Enroll three real application users from command audio (not speaker-v2 train data)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.speaker.application import enroll_user


MANIFEST = PROJECT_ROOT / "data/commands/command_audio_manifest.csv"
OUTPUT = PROJECT_ROOT / "experiments/system/demo_enrollment_data.csv"
SPEAKER_TO_USER = {
    "cmdspk01": ("user_001", "Application User 001"),
    "cmdspk02": ("user_002", "Application User 002"),
    "cmdspk03": ("user_003", "Application User 003"),
}


def _load_validation_audio() -> dict[str, list[dict[str, str]]]:
    selected = {speaker_id: [] for speaker_id in SPEAKER_TO_USER}
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            speaker_id = row["speaker_id"]
            if (
                row["split"] == "validation"
                and row["status"] == "recorded"
                and speaker_id in selected
            ):
                audio_path = PROJECT_ROOT / row["audio_path"]
                if audio_path.is_file():
                    selected[speaker_id].append(row)
    for speaker_id, rows in selected.items():
        if len(rows) < 10:
            raise RuntimeError(
                f"{speaker_id} needs 10 validation recordings; found {len(rows)}"
            )
    return selected


def prepare_demo_enrollments() -> list[dict]:
    audio_by_speaker = _load_validation_audio()
    output_rows: list[dict] = []
    for speaker_id, (user_id, name) in SPEAKER_TO_USER.items():
        rows = audio_by_speaker[speaker_id]
        enrollment_rows = rows[:5]
        heldout_rows = rows[5:10]
        result = enroll_user(
            user_id,
            name,
            [str(PROJECT_ROOT / row["audio_path"]) for row in enrollment_rows],
        )
        if not result["success"]:
            raise RuntimeError(f"Enrollment failed for {user_id}: {result['error']}")
        for role, role_rows in (
            ("ENROLLMENT", enrollment_rows),
            ("HELDOUT_QUERY", heldout_rows),
        ):
            for row in role_rows:
                output_rows.append(
                    {
                        "user_id": user_id,
                        "source_speaker_id": speaker_id,
                        "role": role,
                        "recording_id": row["recording_id"],
                        "audio_path": row["audio_path"],
                        "command_split": row["split"],
                        "centroid_path": result["centroid_path"],
                        "speaker_model_training": "false",
                        "dataset_version": "application-demo-v1",
                    }
                )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def main() -> int:
    rows = prepare_demo_enrollments()
    summary = {
        "users": sorted({row["user_id"] for row in rows}),
        "enrollment_audio_count": sum(row["role"] == "ENROLLMENT" for row in rows),
        "heldout_query_count": sum(row["role"] == "HELDOUT_QUERY" for row in rows),
        "output": str(OUTPUT.relative_to(PROJECT_ROOT)),
        "speaker_model_training": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

