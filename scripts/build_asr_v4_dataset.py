"""Create ASR v4 by versioning the immutable ASR v3 train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from src.utils import canonical_csv_sha256, sha256_file


SPLITS = ("train", "validation", "test")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty ASR split: {path}")
    return rows


def build_dataset(source_root: Path, output_root: Path) -> dict:
    """Copy v3 splits to v4 and prove that their canonical content is identical."""

    source_manifest_path = source_root / "asr_finetune_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset_version") != "v3":
        raise ValueError("Source dataset is not v3")
    if source_manifest.get("freeze_status") != "FROZEN":
        raise ValueError("ASR v3 must be FROZEN before it can seed v4")

    manifest_path = output_root / "asr_finetune_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("freeze_status") == "FROZEN":
            raise ValueError("ASR v4 is FROZEN and cannot be rebuilt")
        raise FileExistsError(f"Refusing to overwrite existing v4 manifest: {manifest_path}")

    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, dict] = {}
    split_rows: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        filename = f"asr_finetune_{split}.csv"
        source = source_root / "metadata" / filename
        target = metadata_root / filename
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite v4 split: {target}")
        rows = _read_rows(source)
        for row in rows:
            audio = Path(row["audio_path"])
            if not audio.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio}")
            if sha256_file(audio) != row["audio_sha256"]:
                raise ValueError(f"Audio checksum mismatch: {audio}")
        shutil.copyfile(source, target)
        source_canonical = canonical_csv_sha256(source)
        target_canonical = canonical_csv_sha256(target)
        if target_canonical != source_canonical:
            raise RuntimeError(f"Canonical CSV changed while copying {split}")
        expected = source_manifest["datasets"][split]["canonical_csv_sha256"]
        if target_canonical != expected:
            raise ValueError(f"v3 manifest checksum mismatch for {split}")
        split_rows[split] = rows
        datasets[split] = {
            "path": target.as_posix(),
            "row_count": len(rows),
            "speaker_count": len({row["speaker_id"] for row in rows}),
            "sha256": sha256_file(target),
            "canonical_csv_sha256": target_canonical,
            "identical_to_v3": True,
        }

    audio_sets = {
        split: {row["audio_sha256"] for row in rows}
        for split, rows in split_rows.items()
    }
    speaker_sets = {
        split: {row["speaker_id"] for row in rows}
        for split, rows in split_rows.items()
    }
    overlaps = {
        "train_validation": len(audio_sets["train"] & audio_sets["validation"]),
        "train_test": len(audio_sets["train"] & audio_sets["test"]),
        "validation_test": len(audio_sets["validation"] & audio_sets["test"]),
    }
    speaker_overlaps = {
        "train_validation": len(speaker_sets["train"] & speaker_sets["validation"]),
        "train_test": len(speaker_sets["train"] & speaker_sets["test"]),
        "validation_test": len(speaker_sets["validation"] & speaker_sets["test"]),
    }
    if any(overlaps.values()) or any(speaker_overlaps.values()):
        raise ValueError("Source v3 splits are not audio/speaker disjoint")

    manifest = {
        "manifest_schema_version": 1,
        "dataset_version": "v4",
        "component": "asr_finetune",
        "freeze_status": "DEVELOPMENT",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "dataset_version": "v3",
            "manifest_path": source_manifest_path.as_posix(),
            "manifest_sha256": sha256_file(source_manifest_path),
            "copy_policy": "exact train/validation/test reuse",
        },
        "datasets": datasets,
        "invariants": {
            "canonical_rows_identical_to_v3": True,
            "audio_overlap": overlaps,
            "speaker_overlap": speaker_overlaps,
        },
        "experiment_protocol": {
            "train_only_on": "train",
            "checkpoint_selection_only_on": "validation",
            "test_used_for_training_or_selection": False,
            "test_is_fresh_holdout": False,
            "test_history_note": (
                "This exact test split was already evaluated in the ASR v3 experiment. "
                "It supports a controlled same-set comparison, not a new unbiased holdout estimate."
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/processed/v3"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/v4"))
    args = parser.parse_args()
    manifest = build_dataset(args.source_root, args.output_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
