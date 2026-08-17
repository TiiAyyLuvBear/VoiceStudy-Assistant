"""Build a portable ECAPA Kaggle dataset from frozen v2 metadata."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.files import sha256_file


SPEAKER_SPLITS = (
    "svm_closed_set_enrollment.csv",
    "svm_closed_set_train.csv",
    "svm_closed_set_validation.csv",
    "svm_closed_set_test.csv",
    "cosine_test_enrollment.csv",
    "cosine_test_query.csv",
    "cosine_test_unknown.csv",
)
SELECTION_FILES = (
    "selected_svm_experimental_speakers.csv",
    "selected_test_enrolled_speakers.csv",
    "selected_test_unknown_speakers.csv",
)
ASR_EXCLUSION_FILES = ("asr_validation.csv", "asr_test.csv")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    missing = [
        field
        for field in ("audio_id", "audio_path", "normalized_speaker_id", "checksum")
        if field not in fields
    ]
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}")
    if not rows:
        raise ValueError(f"Empty speaker split: {path}")
    return fields, rows


def _relative_audio_path(value: str) -> Path:
    path = Path(value.strip())
    parts = path.parts
    if len(parts) >= 2 and parts[0].casefold() == "data" and parts[1].casefold() == "audio":
        path = Path(*parts[2:])
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe audio path: {value}")
    return path


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_groups(rows_by_split: dict[str, list[dict[str, str]]]) -> None:
    svm_names = SPEAKER_SPLITS[:4]
    svm_sets = [
        {row["normalized_speaker_id"] for row in rows_by_split[name]}
        for name in svm_names
    ]
    if any(group != svm_sets[0] for group in svm_sets[1:]):
        raise ValueError("SVM speaker set differs across enrollment/train/validation/test")
    enrolled = {
        row["normalized_speaker_id"]
        for name in ("cosine_test_enrollment.csv", "cosine_test_query.csv")
        for row in rows_by_split[name]
    }
    enrollment_only = {
        row["normalized_speaker_id"]
        for row in rows_by_split["cosine_test_enrollment.csv"]
    }
    query_only = {
        row["normalized_speaker_id"]
        for row in rows_by_split["cosine_test_query.csv"]
    }
    unknown = {
        row["normalized_speaker_id"]
        for row in rows_by_split["cosine_test_unknown.csv"]
    }
    if enrollment_only != query_only:
        raise ValueError("Cosine enrollment and query speaker sets differ")
    if svm_sets[0] & enrolled or svm_sets[0] & unknown or enrolled & unknown:
        raise ValueError("Speaker leakage across SVM/enrolled/unknown groups")


def build_kaggle_dataset(
    *,
    source_root: Path = Path("data/raw/v2/v2"),
    audio_root: Path = Path("data/audio"),
    output_root: Path = Path("data/datasets/ecapa_kaggle_v2"),
) -> dict[str, Any]:
    """Copy frozen speaker protocol audio and write portable metadata paths."""

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_root}")
    metadata_source = source_root / "metadata"
    source_manifest_path = source_root / "split_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest_path}")

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    fields_by_split: dict[str, list[str]] = {}
    for name in SPEAKER_SPLITS:
        fields, rows = _read_csv(metadata_source / name)
        fields_by_split[name] = fields
        rows_by_split[name] = rows
    _validate_groups(rows_by_split)

    all_rows = [row for name in SPEAKER_SPLITS for row in rows_by_split[name]]
    for field in ("audio_id", "audio_path", "checksum"):
        values = [row[field].strip().casefold() for row in all_rows]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field} across speaker splits")

    asr_paths: set[str] = set()
    for name in ASR_EXCLUSION_FILES:
        path = metadata_source / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing ASR exclusion metadata: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            asr_paths.update(
                _relative_audio_path(row["audio_path"]).as_posix().casefold()
                for row in csv.DictReader(stream)
            )
    selected_paths = {
        _relative_audio_path(row["audio_path"]).as_posix().casefold()
        for row in all_rows
    }
    overlap = selected_paths & asr_paths
    if overlap:
        raise ValueError(f"Speaker audio overlaps ASR exclusions: {len(overlap)}")

    output_root.mkdir(parents=True)
    try:
        split_summary: dict[str, Any] = {}
        total_bytes = 0
        for name in SPEAKER_SPLITS:
            portable_rows = []
            speakers = Counter()
            split_bytes = 0
            for row in rows_by_split[name]:
                relative = _relative_audio_path(row["audio_path"])
                source = audio_root / relative
                if not source.is_file():
                    raise FileNotFoundError(f"Missing source audio: {source}")
                checksum = sha256_file(source)
                if checksum.casefold() != row["checksum"].strip().casefold():
                    raise ValueError(f"Audio checksum mismatch: {source}")
                destination = output_root / "audio" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied_checksum = sha256_file(destination)
                if copied_checksum != checksum:
                    raise ValueError(f"Copied audio checksum mismatch: {destination}")
                copied = dict(row)
                copied["audio_path"] = (Path("audio") / relative).as_posix()
                portable_rows.append(copied)
                speakers[row["normalized_speaker_id"]] += 1
                split_bytes += destination.stat().st_size
            metadata_path = output_root / "metadata" / name
            _write_csv(metadata_path, fields_by_split[name], portable_rows)
            split_summary[name] = {
                "rows": len(portable_rows),
                "speakers": len(speakers),
                "bytes": split_bytes,
                "metadata_sha256": sha256_file(metadata_path),
            }
            total_bytes += split_bytes

        for name in SELECTION_FILES:
            source = metadata_source / name
            if not source.is_file():
                raise FileNotFoundError(f"Missing selection metadata: {source}")
            shutil.copy2(source, output_root / "metadata" / name)
        exclusion_dir = output_root / "metadata" / "asr_exclusions"
        exclusion_dir.mkdir()
        for name in ASR_EXCLUSION_FILES:
            shutil.copy2(metadata_source / name, exclusion_dir / name)
        shutil.copy2(source_manifest_path, output_root / "source_split_manifest.json")

        manifest = {
            "dataset": "ecapa_kaggle_v2",
            "purpose": "portable ECAPA training and evaluation package for Kaggle",
            "source": source_root.as_posix(),
            "audio_storage": "copied raw audio; no preprocessing",
            "audio_path_base": "dataset root",
            "asr_audio_included": False,
            "asr_exclusion_rows": len(asr_paths),
            "speaker_audio": len(all_rows),
            "speaker_groups": {
                "svm": len({row["normalized_speaker_id"] for name in SPEAKER_SPLITS[:4] for row in rows_by_split[name]}),
                "cosine_test_enrolled": len({row["normalized_speaker_id"] for name in SPEAKER_SPLITS[4:6] for row in rows_by_split[name]}),
                "cosine_test_unknown": len({row["normalized_speaker_id"] for row in rows_by_split[SPEAKER_SPLITS[6]]}),
            },
            "total_audio_bytes": total_bytes,
            "splits": split_summary,
            "invariants": {
                "duplicate_audio_id_path_checksum": 0,
                "speaker_overlap_across_task_groups": 0,
                "audio_overlap_with_asr": 0,
                "all_audio_checksums_verified": True,
            },
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_root / "README.md").write_text(
            "# ECAPA Kaggle dataset v2\n\n"
            "Portable copy of frozen speaker-v2 protocols. Audio is raw, mono 48 kHz, "
            "and not preprocessed. Resolve `audio_path` from dataset root.\n\n"
            "Train ECAPA with `metadata/svm_closed_set_train.csv`; use validation for "
            "model selection and test only for final evaluation. Cosine files are a "
            "speaker-disjoint verification test. Files under `metadata/asr_exclusions` "
            "document excluded ASR samples; their audio is not included.\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(output_root)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/raw/v2/v2"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/audio"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/datasets/ecapa_kaggle_v2")
    )
    args = parser.parse_args()
    manifest = build_kaggle_dataset(
        source_root=args.source_root,
        audio_root=args.audio_root,
        output_root=args.output_root,
    )
    print(json.dumps({
        "output": args.output_root.as_posix(),
        "speaker_audio": manifest["speaker_audio"],
        "total_audio_bytes": manifest["total_audio_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
