"""Lock the converted ASR v3 model and decoding configuration before test."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.utils import canonical_csv_sha256, sha256_file


def _directory_checksums(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/experimental/asr/v3/ctranslate2"),
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("models/experimental/asr/v3/training_summary.json"),
    )
    parser.add_argument(
        "--export-summary",
        type=Path,
        default=Path("models/experimental/asr/v3/export_summary.json"),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/v3/metadata/asr_finetune_train.csv"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/processed/v3/metadata/asr_finetune_validation.csv"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/v3/metadata/asr_finetune_test.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/experimental/asr/v3/locked_model.json"),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Lock already exists: {args.output}")
    training = json.loads(args.training_summary.read_text(encoding="utf-8"))
    exported = json.loads(args.export_summary.read_text(encoding="utf-8"))
    if training.get("status") != "TRAINED_VALIDATION_SELECTED_NOT_TESTED":
        raise ValueError("Unexpected training status")
    if training.get("test_split_accessed") is not False:
        raise ValueError("Test was accessed during training")
    if exported.get("status") != "MERGED_NOT_LOCKED_NOT_TESTED":
        raise ValueError("Unexpected export status")
    required = {"model.bin", "config.json", "tokenizer.json"}
    missing = required - {file.name for file in args.model_dir.iterdir() if file.is_file()}
    if missing:
        raise ValueError(f"Converted model is incomplete: {sorted(missing)}")

    lock = {
        "schema_version": 1,
        "dataset_version": "asr-v3",
        "model_version": "whisper-small-lora-v3",
        "status": "LOCKED_BEFORE_FINAL_TEST",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection": training["selection"],
        "training_summary": {
            "path": args.training_summary.as_posix(),
            "sha256": sha256_file(args.training_summary),
        },
        "export_summary": {
            "path": args.export_summary.as_posix(),
            "sha256": sha256_file(args.export_summary),
        },
        "datasets": {
            name: {
                "path": path.as_posix(),
                "canonical_csv_sha256": canonical_csv_sha256(path),
            }
            for name, path in (
                ("train", args.train),
                ("validation", args.validation),
                ("test", args.test),
            )
        },
        "inference": {
            "backend": "faster-whisper",
            "device": "cpu",
            "compute_type": "int8",
            "language": "vi",
            "task": "transcribe",
            "beam_size": 10,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        },
        "model_directory": args.model_dir.as_posix(),
        "model_files": _directory_checksums(args.model_dir),
        "test_used_for_training_or_selection": False,
    }
    args.output.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
