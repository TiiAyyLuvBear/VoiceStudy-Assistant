"""Tạo hàng đợi thu âm từ command validation và command test."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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

INTENT_ORDER = (
    'GET_TIME',
    'VIEW_SCHEDULE',
    'ADD_SCHEDULE',
    'VIEW_PRIVATE_NOTE',
    'OUT_OF_SCOPE',
)
DEFAULT_SPEAKER_IDS = ('cmdspk01', 'cmdspk02', 'cmdspk03')


def assign_balanced_speakers(
    rows: list[dict[str, str]],
    speaker_ids: tuple[str, ...] = DEFAULT_SPEAKER_IDS,
) -> None:
    '''Assign every speaker every intent and equal prompts per split.'''

    if not speaker_ids or len(set(speaker_ids)) != len(speaker_ids):
        raise ValueError('speaker_ids must be non-empty and unique')

    split_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        split_rows[row['split']].append(row)

    for split, current_rows in split_rows.items():
        speaker_count = len(speaker_ids)
        if len(current_rows) % speaker_count:
            raise ValueError(
                f'{split} has {len(current_rows)} prompts; cannot divide '
                f'equally among {speaker_count} speakers'
            )
        target_count = len(current_rows) // speaker_count
        by_intent: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in current_rows:
            by_intent[row['intent']].append(row)

        missing = [intent for intent in INTENT_ORDER if not by_intent[intent]]
        if missing:
            raise ValueError(f'{split} is missing intents: {missing}')
        too_small = [
            intent
            for intent in INTENT_ORDER
            if len(by_intent[intent]) < speaker_count
        ]
        if too_small:
            raise ValueError(
                f'{split} cannot give every speaker these intents: {too_small}'
            )

        totals = {speaker_id: 0 for speaker_id in speaker_ids}
        for intent_index, intent in enumerate(INTENT_ORDER):
            intent_rows = by_intent[intent]
            base, remainder = divmod(len(intent_rows), speaker_count)
            quotas = {speaker_id: base for speaker_id in speaker_ids}
            for speaker_id in speaker_ids:
                totals[speaker_id] += base

            rotation = intent_index % speaker_count
            rotated = speaker_ids[rotation:] + speaker_ids[:rotation]
            for _ in range(remainder):
                eligible = [
                    speaker_id
                    for speaker_id in rotated
                    if totals[speaker_id] < target_count
                ]
                if not eligible:
                    raise ValueError(f'Cannot balance speaker assignment for {split}')
                speaker_id = min(
                    eligible,
                    key=lambda value: (totals[value], rotated.index(value)),
                )
                quotas[speaker_id] += 1
                totals[speaker_id] += 1

            cursor = 0
            while cursor < len(intent_rows):
                for speaker_id in rotated:
                    if quotas[speaker_id] <= 0:
                        continue
                    intent_rows[cursor]['speaker_id'] = speaker_id
                    quotas[speaker_id] -= 1
                    cursor += 1

        if any(count != target_count for count in totals.values()):
            raise ValueError(
                f'Unbalanced assignment for {split}: {totals}; '
                f'expected {target_count} each'
            )


def build_rows(
    command_files: list[Path],
    speaker_ids: tuple[str, ...] = DEFAULT_SPEAKER_IDS,
) -> list[dict[str, str]]:
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
    assign_balanced_speakers(rows, speaker_ids)
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
    parser.add_argument(
        '--speakers',
        nargs='+',
        default=list(DEFAULT_SPEAKER_IDS),
        help='Speaker IDs assigned evenly within every split and intent',
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Manifest already exists: {args.output}. Use --force only before recording."
        )
    rows = build_rows(args.commands, tuple(args.speakers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {args.output} with {len(rows)} recording prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
