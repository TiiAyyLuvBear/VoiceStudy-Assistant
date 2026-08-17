"""Build larger Speaker v2 splits without overlapping ASR v2 audio."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils import canonical_csv_sha256, sha256_file


RANDOM_SEED = 42
SVM_FILES = (
    "svm_closed_set_enrollment.csv",
    "svm_closed_set_train.csv",
    "svm_closed_set_validation.csv",
    "svm_closed_set_test.csv",
)
SVM_GLOBAL_TARGETS = {
    "svm_closed_set_enrollment.csv": 45,
    "svm_closed_set_train.csv": 600,
    "svm_closed_set_validation.csv": 129,
    "svm_closed_set_test.csv": 128,
}
SVM_ENROLLMENT_PER_SPEAKER = 5
SVM_MINIMUM_TOTAL_PER_SPEAKER = 75
SVM_BASE_NON_ENROLLMENT_PER_SPEAKER = 95
SVM_NON_ENROLLMENT_TOTAL = 857
SVM_SPLIT_ROLES = {
    "svm_closed_set_enrollment.csv": "ENROLLMENT",
    "svm_closed_set_train.csv": "TRAIN",
    "svm_closed_set_validation.csv": "VALIDATION",
    "svm_closed_set_test.csv": "TEST",
}
COSINE_ENROLLED_SPEAKERS = 8
COSINE_UNKNOWN_SPEAKERS = 8
COSINE_ENROLLMENT_PER_SPEAKER = 5
COSINE_QUERY_PER_SPEAKER = 25
ASR_FILES = ("asr_validation.csv", "asr_test.csv")
SELECTION_FIELDS = (
    "speaker_id",
    "normalized_speaker_id",
    "protocol",
    "role",
    "project_split",
    "source_project_split",
    "usable_audio_count",
    "enrollment_count",
    "train_count",
    "validation_count",
    "test_query_count",
    "unknown_query_count",
    "speaker_sex",
    "speaker_age",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        return fields, list(reader)


def _is_valid(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "invalid", "error"}


def _audio_key(value: str) -> str:
    normalized = value.strip().replace("\\", "/").casefold()
    marker = "/data/audio/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    elif normalized.startswith("data/audio/"):
        normalized = normalized[len("data/audio/") :]
    return normalized.lstrip("./")


def _audio_file(audio_root: Path, path_value: str) -> Path:
    source = Path(path_value)
    if source.is_absolute():
        return source
    key = _audio_key(path_value)
    return audio_root / Path(key)


def _usable_inventory(
    inventory: Path,
    audio_root: Path,
    asr_paths: set[str],
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, str]]]:
    fields, source_rows = _read_csv(inventory)
    required = {
        "audio_path",
        "project_split",
        "speaker_id",
        "speaker_sex",
        "speaker_age",
        "transcript",
        "is_valid",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Inventory missing columns: {sorted(missing)}")

    usable: list[dict[str, str]] = []
    by_path: dict[str, dict[str, str]] = {}
    for row in source_rows:
        path_value = row["audio_path"].strip()
        key = _audio_key(path_value)
        if (
            not path_value
            or not row["transcript"].strip()
            or not _is_valid(row["is_valid"])
            or key in asr_paths
            or not _audio_file(audio_root, path_value).is_file()
        ):
            continue
        if key in by_path:
            continue
        copied = dict(row)
        usable.append(copied)
        by_path[key] = copied
    return fields, usable, by_path


def _load_asr_paths(metadata_dir: Path) -> set[str]:
    paths: set[str] = set()
    for name in ASR_FILES:
        _, rows = _read_csv(metadata_dir / name)
        paths.update(_audio_key(row["audio_path"]) for row in rows)
    return paths


def _group_by_speaker(
    rows: list[dict[str, str]],
    project_split: str,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["project_split"].strip().upper() == project_split:
            grouped[row["speaker_id"]].append(row)
    return dict(grouped)


def _speaker_metadata(rows: list[dict[str, str]]) -> tuple[str, str]:
    return rows[0]["speaker_sex"], rows[0]["speaker_age"]


def _shuffled(rows: list[dict[str, str]], salt: str) -> list[dict[str, str]]:
    selected = list(rows)
    random.Random(f"{RANDOM_SEED}:{salt}").shuffle(selected)
    return selected


def _write_csv(
    path: Path,
    fields: list[str] | tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _materialize(
    row: dict[str, str],
    *,
    audio_root: Path,
    normalized_speaker_id: str,
    protocol: str,
    role: str,
    dataset_type: str,
    split_role: str,
    split_name: str,
    checksum_cache: dict[str, str],
) -> dict[str, str]:
    result = dict(row)
    key = _audio_key(row["audio_path"])
    if key not in checksum_cache:
        checksum_cache[key] = sha256_file(_audio_file(audio_root, row["audio_path"]))
    result.update(
        {
            "normalized_speaker_id": normalized_speaker_id,
            "protocol": protocol,
            "role": role,
            "checksum": checksum_cache[key],
            "dataset_type": dataset_type,
            "split_role": split_role,
            "split_name": split_name,
        }
    )
    return result


def _load_svm_mapping(path: Path) -> dict[str, str]:
    _, rows = _read_csv(path)
    return {
        row["speaker_id"]: row["normalized_speaker_id"]
        for row in rows
    }


def _load_v1_anchors(
    metadata_dir: Path,
    source_by_path: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, list[dict[str, str]]]], set[str]]:
    anchors: dict[str, dict[str, list[dict[str, str]]]] = {}
    all_paths: set[str] = set()
    for name in SVM_FILES:
        _, rows = _read_csv(metadata_dir / name)
        by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            key = _audio_key(row["audio_path"])
            all_paths.add(key)
            source = source_by_path.get(key)
            if source is None:
                raise ValueError(f"Speaker v1 anchor is not usable in v2: {key}")
            by_speaker[row["speaker_id"]].append(source)
        anchors[name] = dict(by_speaker)
    return anchors, all_paths


def _build_svm(
    inventory_fields: list[str],
    grouped: dict[str, list[dict[str, str]]],
    mapping: dict[str, str],
    anchors: dict[str, dict[str, list[dict[str, str]]]],
    anchor_paths: set[str],
    audio_root: Path,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]], list[str]]:
    selected_speakers = [
        speaker_id
        for speaker_id in sorted(mapping, key=lambda value: mapping[value])
        if len(grouped.get(speaker_id, ())) >= SVM_MINIMUM_TOTAL_PER_SPEAKER
    ]
    if not selected_speakers:
        raise ValueError("No SVM speakers can satisfy the v2 targets")

    available_non_enrollment = {
        speaker_id: len(grouped[speaker_id]) - SVM_ENROLLMENT_PER_SPEAKER
        for speaker_id in selected_speakers
    }
    assigned_total = {
        speaker_id: min(
            SVM_BASE_NON_ENROLLMENT_PER_SPEAKER,
            available_non_enrollment[speaker_id],
        )
        for speaker_id in selected_speakers
    }
    extra_needed = SVM_NON_ENROLLMENT_TOTAL - sum(assigned_total.values())
    while extra_needed:
        progressed = False
        for speaker_id in selected_speakers:
            if assigned_total[speaker_id] >= available_non_enrollment[speaker_id]:
                continue
            assigned_total[speaker_id] += 1
            extra_needed -= 1
            progressed = True
            if not extra_needed:
                break
        if not progressed:
            raise ValueError("Insufficient SVM audio for the requested 70/15/15 split")

    train_targets = {
        speaker_id: assigned_total[speaker_id] * 70 // 100
        for speaker_id in selected_speakers
    }
    train_extra = SVM_GLOBAL_TARGETS["svm_closed_set_train.csv"] - sum(
        train_targets.values()
    )
    train_remainders = sorted(
        selected_speakers,
        key=lambda speaker_id: (
            -(assigned_total[speaker_id] * 70 % 100),
            mapping[speaker_id],
        ),
    )
    for speaker_id in train_remainders[:train_extra]:
        train_targets[speaker_id] += 1

    remaining_after_train = {
        speaker_id: assigned_total[speaker_id] - train_targets[speaker_id]
        for speaker_id in selected_speakers
    }
    validation_targets = {
        speaker_id: remaining_after_train[speaker_id] // 2
        for speaker_id in selected_speakers
    }
    validation_extra = SVM_GLOBAL_TARGETS[
        "svm_closed_set_validation.csv"
    ] - sum(validation_targets.values())
    validation_remainders = sorted(
        selected_speakers,
        key=lambda speaker_id: (
            -(remaining_after_train[speaker_id] % 2),
            mapping[speaker_id],
        ),
    )
    for speaker_id in validation_remainders[:validation_extra]:
        validation_targets[speaker_id] += 1

    quotas = {
        speaker_id: {
            "svm_closed_set_enrollment.csv": SVM_ENROLLMENT_PER_SPEAKER,
            "svm_closed_set_train.csv": train_targets[speaker_id],
            "svm_closed_set_validation.csv": validation_targets[speaker_id],
            "svm_closed_set_test.csv": (
                remaining_after_train[speaker_id]
                - validation_targets[speaker_id]
            ),
        }
        for speaker_id in selected_speakers
    }

    outputs = {name: [] for name in SVM_FILES}
    selection_rows: list[dict[str, str]] = []
    checksum_cache: dict[str, str] = {}
    for speaker_id in selected_speakers:
        normalized = mapping[speaker_id]
        speaker_rows = grouped[speaker_id]
        pool = [
            row for row in speaker_rows
            if _audio_key(row["audio_path"]) not in anchor_paths
        ]
        pool = _shuffled(pool, f"svm:{speaker_id}")
        cursor = 0
        for name in SVM_FILES:
            target = quotas[speaker_id][name]
            fixed = list(anchors[name].get(speaker_id, ()))
            needed = target - len(fixed)
            if needed < 0:
                raise ValueError(f"v1 has more rows than v2 target for {speaker_id}:{name}")
            chosen = fixed + pool[cursor : cursor + needed]
            cursor += needed
            if len(chosen) != target:
                raise ValueError(f"Insufficient rows for {speaker_id}:{name}")
            outputs[name].extend(
                _materialize(
                    row,
                    audio_root=audio_root,
                    normalized_speaker_id=normalized,
                    protocol="SVM_CLOSED_SET",
                    role="SVM_EXPERIMENTAL",
                    dataset_type="SVM_CLOSED_SET",
                    split_role=SVM_SPLIT_ROLES[name],
                    split_name=Path(name).stem,
                    checksum_cache=checksum_cache,
                )
                for row in chosen
            )
        sex, age = _speaker_metadata(speaker_rows)
        selection_rows.append(
            {
                "speaker_id": speaker_id,
                "normalized_speaker_id": normalized,
                "protocol": "SVM_CLOSED_SET",
                "role": "SVM_EXPERIMENTAL",
                "project_split": "SVM",
                "source_project_split": "SVM",
                "usable_audio_count": str(len(speaker_rows)),
                "enrollment_count": str(
                    quotas[speaker_id]["svm_closed_set_enrollment.csv"]
                ),
                "train_count": str(
                    quotas[speaker_id]["svm_closed_set_train.csv"]
                ),
                "validation_count": str(
                    quotas[speaker_id]["svm_closed_set_validation.csv"]
                ),
                "test_query_count": str(
                    quotas[speaker_id]["svm_closed_set_test.csv"]
                ),
                "unknown_query_count": "0",
                "speaker_sex": sex,
                "speaker_age": age,
            }
        )
    dropped = sorted(set(mapping) - set(selected_speakers))
    return outputs, selection_rows, dropped


def _build_cosine_test(
    grouped: dict[str, list[dict[str, str]]],
    audio_root: Path,
) -> tuple[
    dict[str, list[dict[str, str]]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    ranked = sorted(grouped, key=lambda speaker: (-len(grouped[speaker]), speaker))
    enrolled = [
        speaker for speaker in ranked
        if len(grouped[speaker])
        >= COSINE_ENROLLMENT_PER_SPEAKER + COSINE_QUERY_PER_SPEAKER
    ][:COSINE_ENROLLED_SPEAKERS]
    remaining = [speaker for speaker in ranked if speaker not in enrolled]
    if len(enrolled) != COSINE_ENROLLED_SPEAKERS:
        raise ValueError("Not enough UNUSED speakers for cosine enrolled targets")
    if len(remaining) < COSINE_UNKNOWN_SPEAKERS:
        raise ValueError("Not enough UNUSED speakers for cosine unknown targets")
    unknown = remaining[:COSINE_UNKNOWN_SPEAKERS]

    outputs = {
        "cosine_test_enrollment.csv": [],
        "cosine_test_query.csv": [],
        "cosine_test_unknown.csv": [],
    }
    enrolled_selection: list[dict[str, str]] = []
    unknown_selection: list[dict[str, str]] = []
    checksum_cache: dict[str, str] = {}
    for index, speaker_id in enumerate(enrolled, start=1):
        normalized = f"v2_test_enrolled_spk_{index:04d}"
        rows = _shuffled(grouped[speaker_id], f"cosine-enrolled:{speaker_id}")
        enrollment = rows[:COSINE_ENROLLMENT_PER_SPEAKER]
        query = rows[
            COSINE_ENROLLMENT_PER_SPEAKER:
            COSINE_ENROLLMENT_PER_SPEAKER + COSINE_QUERY_PER_SPEAKER
        ]
        for name, split_rows, split_role in (
            ("cosine_test_enrollment.csv", enrollment, "ENROLLMENT"),
            ("cosine_test_query.csv", query, "QUERY"),
        ):
            outputs[name].extend(
                _materialize(
                    row,
                    audio_root=audio_root,
                    normalized_speaker_id=normalized,
                    protocol="COSINE_TEST",
                    role="ENROLLED",
                    dataset_type="COSINE_TEST",
                    split_role=split_role,
                    split_name=Path(name).stem,
                    checksum_cache=checksum_cache,
                )
                for row in split_rows
            )
        sex, age = _speaker_metadata(rows)
        enrolled_selection.append(
            {
                "speaker_id": speaker_id,
                "normalized_speaker_id": normalized,
                "protocol": "COSINE_TEST",
                "role": "ENROLLED",
                "project_split": "TEST",
                "source_project_split": "UNUSED",
                "usable_audio_count": str(len(rows)),
                "enrollment_count": str(len(enrollment)),
                "train_count": "0",
                "validation_count": "0",
                "test_query_count": str(len(query)),
                "unknown_query_count": "0",
                "speaker_sex": sex,
                "speaker_age": age,
            }
        )

    for index, speaker_id in enumerate(unknown, start=1):
        normalized = f"v2_test_unknown_spk_{index:04d}"
        rows = _shuffled(grouped[speaker_id], f"cosine-unknown:{speaker_id}")
        outputs["cosine_test_unknown.csv"].extend(
            _materialize(
                row,
                audio_root=audio_root,
                normalized_speaker_id=normalized,
                protocol="COSINE_TEST",
                role="UNKNOWN",
                dataset_type="COSINE_TEST",
                split_role="UNKNOWN_QUERY",
                split_name="cosine_test_unknown",
                checksum_cache=checksum_cache,
            )
            for row in rows
        )
        sex, age = _speaker_metadata(rows)
        unknown_selection.append(
            {
                "speaker_id": speaker_id,
                "normalized_speaker_id": normalized,
                "protocol": "COSINE_TEST",
                "role": "UNKNOWN",
                "project_split": "TEST",
                "source_project_split": "UNUSED",
                "usable_audio_count": str(len(rows)),
                "enrollment_count": "0",
                "train_count": "0",
                "validation_count": "0",
                "test_query_count": "0",
                "unknown_query_count": str(len(rows)),
                "speaker_sex": sex,
                "speaker_age": age,
            }
        )
    return outputs, enrolled_selection, unknown_selection


def _speaker_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["normalized_speaker_id"] for row in rows).items()))


def _validate(
    split_rows: dict[str, list[dict[str, str]]],
    asr_paths: set[str],
) -> None:
    all_paths: list[str] = []
    for name, rows in split_rows.items():
        paths = [_audio_key(row["audio_path"]) for row in rows]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate audio inside {name}")
        if set(paths) & asr_paths:
            raise ValueError(f"ASR v2 overlap found in {name}")
        all_paths.extend(paths)
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("Audio overlap found across Speaker v2 splits")

    for name, target in SVM_GLOBAL_TARGETS.items():
        counts = Counter(
            row["normalized_speaker_id"] for row in split_rows[name]
        )
        if len(counts) != 9 or sum(counts.values()) != target:
            raise ValueError(f"Unexpected count in {name}: {counts}")
    enrollment_counts = Counter(
        row["normalized_speaker_id"]
        for row in split_rows["svm_closed_set_enrollment.csv"]
    )
    if set(enrollment_counts.values()) != {SVM_ENROLLMENT_PER_SPEAKER}:
        raise ValueError("SVM enrollment must contain 5 audio per speaker")
    enrolled = Counter(
        row["normalized_speaker_id"]
        for row in split_rows["cosine_test_enrollment.csv"]
    )
    query = Counter(
        row["normalized_speaker_id"]
        for row in split_rows["cosine_test_query.csv"]
    )
    unknown = Counter(
        row["normalized_speaker_id"]
        for row in split_rows["cosine_test_unknown.csv"]
    )
    if len(enrolled) != 8 or set(enrolled.values()) != {5}:
        raise ValueError("Cosine enrollment must be 8 speakers x 5")
    if len(query) != 8 or set(query.values()) != {25}:
        raise ValueError("Cosine query must be 8 speakers x 25")
    if len(unknown) != 8:
        raise ValueError("Cosine unknown must contain 8 speakers")
    if set(enrolled) != set(query) or set(enrolled) & set(unknown):
        raise ValueError("Cosine enrolled/query/unknown speaker leakage")


def _merge_manifest(
    path: Path,
    inventory: Path,
    asr_metadata_dir: Path,
    output_dir: Path,
    split_rows: dict[str, list[dict[str, str]]],
    selection_paths: dict[str, Path],
    svm_selection: list[dict[str, str]],
    enrolled_selection: list[dict[str, str]],
    unknown_selection: list[dict[str, str]],
    dropped_svm: list[str],
) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
    else:
        previous = {}
    if "components" in previous:
        manifest = previous
    else:
        asr_component = {
            key: previous[key]
            for key in ("source_inventory", "selection", "datasets")
            if key in previous
        }
        manifest = {
            "manifest_schema_version": 1,
            "dataset_version": "v2",
            "created_at": previous.get(
                "created_at",
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
            "random_seed": RANDOM_SEED,
            "components": {"asr": asr_component},
        }
    manifest["dataset_version"] = "v2"
    manifest["random_seed"] = RANDOM_SEED
    manifest["freeze_status"] = "FROZEN"

    asr_exclusions = {}
    for name in ASR_FILES:
        source = asr_metadata_dir / name
        _, rows = _read_csv(source)
        asr_exclusions[name] = {
            "path": source.as_posix(),
            "num_audio": len(rows),
            "checksum": canonical_csv_sha256(source),
        }
    split_summary = {
        name: {
            "path": (output_dir / name).as_posix(),
            "num_audio": len(rows),
            "num_speaker": len(_speaker_counts(rows)),
            "per_speaker": _speaker_counts(rows),
            "checksum": canonical_csv_sha256(output_dir / name),
        }
        for name, rows in split_rows.items()
    }
    manifest.setdefault("components", {})["speaker"] = {
        "dataset_version": "speaker-v2",
        "freeze_status": "FROZEN",
        "checksum_algorithm": "sha256-canonical-csv-v1",
        "source_inventory": {
            "path": inventory.as_posix(),
            "checksum": canonical_csv_sha256(inventory),
        },
        "selection_policy": {
            "asr_v2_audio_excluded": True,
            "svm_v1_roles_preserved": True,
            "svm_global_target": {
                "enrollment": 45,
                "train": 600,
                "validation": 129,
                "test": 128,
                "train_validation_test_ratio": {
                    "train": 0.70011669,
                    "validation": 0.15052509,
                    "test": 0.14935823,
                },
            },
            "speaker_disjoint_test": {
                "enrolled_speakers": 8,
                "enrollment_per_speaker": 5,
                "query_per_speaker": 25,
                "unknown_speakers": 8,
                "unknown_policy": "all usable audio from each selected unknown speaker",
            },
            "dropped_svm_speakers_below_75_usable_audio": dropped_svm,
        },
        "asr_v2_exclusions": asr_exclusions,
        "speaker_groups": {
            "svm": len(svm_selection),
            "test_enrolled": len(enrolled_selection),
            "test_unknown": len(unknown_selection),
        },
        "invariants": {
            "audio_overlap_across_speaker_splits": 0,
            "audio_overlap_with_asr_v2": 0,
            "svm_and_cosine_speaker_overlap": 0,
            "cosine_enrolled_unknown_speaker_overlap": 0,
        },
        "selection_files": {
            name: {
                "path": source.as_posix(),
                "checksum": canonical_csv_sha256(source),
            }
            for name, source in selection_paths.items()
        },
        "splits": split_summary,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build(
    *,
    inventory: Path,
    audio_root: Path,
    v1_metadata_dir: Path,
    asr_v2_metadata_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    svm_mapping_path: Path,
) -> dict[str, int]:
    asr_paths = _load_asr_paths(asr_v2_metadata_dir)
    inventory_fields, usable, source_by_path = _usable_inventory(
        inventory, audio_root, asr_paths
    )
    svm_grouped = _group_by_speaker(usable, "SVM")
    unused_grouped = _group_by_speaker(usable, "UNUSED")
    mapping = _load_svm_mapping(svm_mapping_path)
    anchors, anchor_paths = _load_v1_anchors(v1_metadata_dir, source_by_path)
    svm_outputs, svm_selection, dropped_svm = _build_svm(
        inventory_fields,
        svm_grouped,
        mapping,
        anchors,
        anchor_paths,
        audio_root,
    )
    cosine_outputs, enrolled_selection, unknown_selection = _build_cosine_test(
        unused_grouped, audio_root
    )
    split_rows = {**svm_outputs, **cosine_outputs}
    _validate(split_rows, asr_paths)

    output_fields = list(inventory_fields)
    for field in ("dataset_type", "split_role", "split_name"):
        if field not in output_fields:
            output_fields.append(field)
    for name, rows in split_rows.items():
        _write_csv(output_dir / name, output_fields, rows)

    selection_paths = {
        "svm": output_dir / "selected_svm_experimental_speakers.csv",
        "test_enrolled": output_dir / "selected_test_enrolled_speakers.csv",
        "test_unknown": output_dir / "selected_test_unknown_speakers.csv",
    }
    _write_csv(selection_paths["svm"], SELECTION_FIELDS, svm_selection)
    _write_csv(
        selection_paths["test_enrolled"], SELECTION_FIELDS, enrolled_selection
    )
    _write_csv(
        selection_paths["test_unknown"], SELECTION_FIELDS, unknown_selection
    )
    _merge_manifest(
        manifest_path,
        inventory,
        asr_v2_metadata_dir,
        output_dir,
        split_rows,
        selection_paths,
        svm_selection,
        enrolled_selection,
        unknown_selection,
        dropped_svm,
    )
    return {name: len(rows) for name, rows in split_rows.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/metadata/data_inventory.csv"),
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/audio"))
    parser.add_argument(
        "--v1-metadata-dir",
        type=Path,
        default=Path("data/processed/v1/metadata"),
    )
    parser.add_argument(
        "--asr-v2-metadata-dir",
        type=Path,
        default=Path("data/processed/v2/metadata"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v2/metadata"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/v2/split_manifest.json"),
    )
    parser.add_argument(
        "--svm-mapping",
        type=Path,
        default=Path("data/metadata/selected_svm_experimental_speakers.csv"),
    )
    args = parser.parse_args()

    counts = build(
        inventory=args.inventory,
        audio_root=args.audio_root,
        v1_metadata_dir=args.v1_metadata_dir,
        asr_v2_metadata_dir=args.asr_v2_metadata_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        svm_mapping_path=args.svm_mapping,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
