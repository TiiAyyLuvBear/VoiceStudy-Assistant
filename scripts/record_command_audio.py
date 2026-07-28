"""Thu command audio 16 kHz mono trực tiếp từ microphone."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_FIELDS = (
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


def _sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed; run: pip install -r requirements.txt"
        ) from exc
    return sd


def _load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run scripts.prepare_command_audio_manifest first."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _save_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(samples.reshape(-1), -1.0, 1.0)
    pcm = (pcm * np.iinfo(np.int16).max).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def _device_value(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def _safe_speaker_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("speaker_id may contain only letters, digits, '_' and '-'")
    return value


def _record_one(
    row: dict[str, str],
    *,
    speaker_id: str,
    duration: float,
    sample_rate: int,
    device: int | str | None,
    assume_yes: bool,
) -> None:
    sd = _sounddevice()
    print(f"\n[{row['command_id']}] {row['expected_transcript']}")
    if not assume_yes:
        input("Nhấn Enter khi sẵn sàng thu...")
    print("Bắt đầu sau: 3", end=" ", flush=True)
    time.sleep(1)
    print("2", end=" ", flush=True)
    time.sleep(1)
    print("1", flush=True)
    time.sleep(1)

    samples = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("Đã thu xong.")

    output = Path("data/commands/audio") / row["split"]
    output = output / f"{row['recording_id']}_{speaker_id}.wav"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing recording: {output}")
    _write_wav(output, samples, sample_rate)

    device_info: dict[str, Any] = sd.query_devices(device, "input")
    row.update(
        {
            "speaker_id": speaker_id,
            "audio_path": output.as_posix(),
            "sample_rate": str(sample_rate),
            "channels": "1",
            "duration_sec": f"{duration:.3f}",
            "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "recording_device": str(device_info.get("name", "unknown")),
            "status": "recorded",
        }
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/commands/command_audio_manifest.csv"),
    )
    parser.add_argument("--speaker-id")
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--command-id")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", help="Input device index or name")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip Enter prompt")
    args = parser.parse_args()

    if args.list_devices:
        print(_sounddevice().query_devices())
        return 0
    if not args.speaker_id:
        parser.error("--speaker-id is required unless --list-devices is used")
    if args.count < 1 or args.duration <= 0:
        parser.error("--count and --duration must be positive")

    speaker_id = _safe_speaker_id(args.speaker_id)
    rows = _load_manifest(args.manifest)
    pending = [row for row in rows if row.get("status") != "recorded"]
    if args.split:
        pending = [row for row in pending if row["split"] == args.split]
    if args.command_id:
        pending = [row for row in pending if row["command_id"] == args.command_id]
    if not pending:
        print("Không còn câu phù hợp đang chờ thu.")
        return 0

    for row in pending[: args.count]:
        _record_one(
            row,
            speaker_id=speaker_id,
            duration=args.duration,
            sample_rate=args.sample_rate,
            device=_device_value(args.device),
            assume_yes=args.yes,
        )
        _save_manifest(args.manifest, rows)
    print(f"Đã cập nhật manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
