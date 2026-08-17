"""Repackage ECAPA audio into task-aware folders for Kaggle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.files import sha256_file


PROTOCOL_LAYOUT = {
    "svm_closed_set_train.csv": Path("train"),
    "svm_closed_set_validation.csv": Path("validation"),
    "svm_closed_set_test.csv": Path("test"),
    "svm_closed_set_enrollment.csv": Path("identification_enrollment"),
    "cosine_test_enrollment.csv": Path("verification_test/enrollment"),
    "cosine_test_query.csv": Path("verification_test/known"),
    "cosine_test_unknown.csv": Path("verification_test/unknown"),
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    required = {"audio_id", "audio_path", "normalized_speaker_id", "checksum"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}")
    if not rows:
        raise ValueError(f"Empty protocol: {path}")
    return fields, rows


def _write_csv(
    path: Path, fields: list[str], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_component(value: str, *, field: str) -> str:
    component = value.strip()
    if (
        not component
        or component in {".", ".."}
        or Path(component).name != component
        or "/" in component
        or "\\" in component
    ):
        raise ValueError(f"Unsafe {field}: {value}")
    return component


def organize_kaggle_dataset(
    *,
    source_root: Path = Path("data/datasets/ecapa_kaggle_v2"),
    output_root: Path = Path("data/datasets/ecapa_kaggle_split_v1"),
) -> dict[str, Any]:
    """Copy one portable ECAPA package into task-aware split folders."""

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_root}")
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset") != "ecapa_kaggle_v2":
        raise ValueError("Source is not an ecapa_kaggle_v2 package")

    output_root.mkdir(parents=True)
    try:
        protocols: dict[str, Any] = {}
        seen_audio_ids: set[str] = set()
        seen_source_paths: set[str] = set()
        seen_checksums: set[str] = set()
        total_bytes = 0
        total_rows = 0

        for metadata_name, leaf in PROTOCOL_LAYOUT.items():
            fields, rows = _read_csv(source_root / "metadata" / metadata_name)
            portable_rows: list[dict[str, str]] = []
            speakers: Counter[str] = Counter()
            protocol_bytes = 0

            for row in rows:
                audio_id = _safe_component(row["audio_id"], field="audio_id")
                speaker = _safe_component(
                    row["normalized_speaker_id"], field="normalized_speaker_id"
                )
                source_relative = Path(row["audio_path"])
                if source_relative.is_absolute() or ".." in source_relative.parts:
                    raise ValueError(f"Unsafe audio_path: {row['audio_path']}")
                source = source_root / source_relative
                if not source.is_file():
                    raise FileNotFoundError(f"Missing source audio: {source}")

                checksum = sha256_file(source)
                expected_checksum = row["checksum"].strip().casefold()
                if checksum != expected_checksum:
                    raise ValueError(f"Audio checksum mismatch: {source}")
                if audio_id.casefold() in seen_audio_ids:
                    raise ValueError(f"Duplicate audio_id: {audio_id}")
                if source_relative.as_posix().casefold() in seen_source_paths:
                    raise ValueError(f"Duplicate audio_path: {source_relative}")
                if checksum in seen_checksums:
                    raise ValueError(f"Duplicate checksum: {checksum}")
                seen_audio_ids.add(audio_id.casefold())
                seen_source_paths.add(source_relative.as_posix().casefold())
                seen_checksums.add(checksum)

                suffix = source.suffix.casefold() or ".wav"
                destination_relative = leaf / "audio" / speaker / f"{audio_id}{suffix}"
                destination = output_root / destination_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if sha256_file(destination) != checksum:
                    raise ValueError(f"Copied audio checksum mismatch: {destination}")

                copied = dict(row)
                copied["audio_path"] = destination_relative.as_posix()
                portable_rows.append(copied)
                speakers[speaker] += 1
                size = destination.stat().st_size
                protocol_bytes += size
                total_bytes += size
                total_rows += 1

            metadata_path = output_root / leaf / "metadata.csv"
            _write_csv(metadata_path, fields, portable_rows)
            protocols[metadata_name] = {
                "directory": leaf.as_posix(),
                "metadata_path": (leaf / "metadata.csv").as_posix(),
                "rows": len(portable_rows),
                "speakers": len(speakers),
                "bytes": protocol_bytes,
                "metadata_sha256": sha256_file(metadata_path),
            }

        expected_rows = source_manifest.get("speaker_audio")
        if expected_rows is not None and total_rows != expected_rows:
            raise ValueError(
                f"Audio count differs from source manifest: {total_rows} != {expected_rows}"
            )

        reference_root = output_root / "reference"
        for name in (
            "selected_svm_experimental_speakers.csv",
            "selected_test_enrolled_speakers.csv",
            "selected_test_unknown_speakers.csv",
        ):
            source = source_root / "metadata" / name
            if source.is_file():
                reference_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, reference_root / name)
        exclusion_source = source_root / "metadata" / "asr_exclusions"
        if exclusion_source.is_dir():
            shutil.copytree(exclusion_source, reference_root / "asr_exclusions")
        shutil.copy2(source_manifest_path, output_root / "source_manifest.json")

        manifest = {
            "dataset": "ecapa_kaggle_split_v1",
            "purpose": "task-aware ECAPA training and evaluation package for Kaggle",
            "source_dataset": source_manifest.get("dataset"),
            "audio_storage": "copied raw audio; no preprocessing",
            "audio_path_base": "dataset root",
            "speaker_audio": total_rows,
            "total_audio_bytes": total_bytes,
            "protocols": protocols,
            "invariants": {
                "duplicate_audio_id_path_checksum": 0,
                "all_audio_checksums_verified": True,
                "test_not_used_for_training_or_model_selection": True,
            },
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_root / "README.md").write_text(
            "# ECAPA Kaggle split dataset v1\n\n"
            "Each task-aware leaf contains `audio/` and `metadata.csv`. Audio paths "
            "inside metadata resolve from dataset root.\n\n"
            "- `train`, `validation`, `test`: closed-set speaker identification.\n"
            "- `identification_enrollment`: enrollment samples for known speakers.\n"
            "- `verification_test/enrollment`: enrolled speaker references.\n"
            "- `verification_test/known`: genuine queries.\n"
            "- `verification_test/unknown`: impostor queries.\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(output_root)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/datasets/ecapa_kaggle_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/ecapa_kaggle_split_v1"),
    )
    args = parser.parse_args()
    manifest = organize_kaggle_dataset(
        source_root=args.source_root, output_root=args.output_root
    )
    print(
        json.dumps(
            {
                "output": args.output_root.as_posix(),
                "speaker_audio": manifest["speaker_audio"],
                "total_audio_bytes": manifest["total_audio_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
