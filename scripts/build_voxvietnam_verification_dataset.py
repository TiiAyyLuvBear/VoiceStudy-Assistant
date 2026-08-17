"""Build compact VoxVietnam protocols for three ECAPA speaker tasks.

The builder streams gated Hugging Face data, writes 16 kHz PCM16 WAV files,
keeps protocol roles disjoint where required, and emits closed-set
identification, verification, and open-set identification ground truth. It
intentionally refuses to overwrite output or exceed the byte budget.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import soundfile as sf

from src.utils.files import sha256_file


HF_DATASET = "hustep-lab/VoxVietnam-Dataset"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_MAX_BYTES = 12 * 1024**3

METADATA_FIELDS = (
    "audio_id",
    "audio_path",
    "speaker_id",
    "normalized_speaker_id",
    "split",
    "split_name",
    "split_role",
    "protocol",
    "role",
    "duration_sec",
    "sample_rate",
    "num_channels",
    "checksum",
    "source_dataset",
    "source_partition",
    "source_index",
)

TRIAL_FIELDS = (
    "trial_id",
    "enrollment_speaker_id",
    "query_audio_path",
    "query_speaker_id",
    "label",
)

PROTOCOL_AUDIO_FIELDS = ("audio_path", "speaker_id", "checksum")
OPEN_QUERY_FIELDS = (
    "query_audio_path",
    "query_speaker_id",
    "is_known",
    "checksum",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def select_speaker_splits(
    training_counts: Mapping[str, int],
    test_counts: Mapping[str, int],
    *,
    train_speakers: int = 600,
    validation_speakers: int = 75,
    test_speakers: int = 75,
    train_audio_per_speaker: int = 30,
    evaluation_audio_per_speaker: int = 15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Select deterministic speakers from official train and test partitions."""

    numeric = (
        train_speakers,
        validation_speakers,
        test_speakers,
        train_audio_per_speaker,
        evaluation_audio_per_speaker,
    )
    if any(value < 1 for value in numeric):
        raise ValueError("Speaker and audio counts must be positive")

    source_overlap = set(training_counts) & set(test_counts)
    if source_overlap:
        raise ValueError(
            f"Official source partitions overlap by {len(source_overlap)} speakers"
        )

    training_candidates = [
        speaker
        for speaker, count in training_counts.items()
        if count >= train_audio_per_speaker
    ]
    training_candidates.sort(key=lambda speaker: _stable_key(seed, speaker))
    if len(training_candidates) < train_speakers:
        raise ValueError(
            f"Need {train_speakers} train speakers with at least "
            f"{train_audio_per_speaker} audio; found {len(training_candidates)}"
        )
    selected_train = training_candidates[:train_speakers]
    selected_train_set = set(selected_train)

    validation_candidates = [
        speaker
        for speaker, count in training_counts.items()
        if speaker not in selected_train_set
        and count >= evaluation_audio_per_speaker
    ]
    validation_candidates.sort(key=lambda speaker: _stable_key(seed, speaker))
    if len(validation_candidates) < validation_speakers:
        raise ValueError(
            f"Need {validation_speakers} remaining validation speakers with at "
            f"least {evaluation_audio_per_speaker} audio; found "
            f"{len(validation_candidates)} after reserving train speakers"
        )

    test_candidates = [
        speaker
        for speaker, count in test_counts.items()
        if count >= evaluation_audio_per_speaker
    ]
    test_candidates.sort(key=lambda speaker: _stable_key(seed + 1, speaker))
    if len(test_candidates) < test_speakers:
        raise ValueError(
            f"Need {test_speakers} official-test speakers with at least "
            f"{evaluation_audio_per_speaker} audio; found {len(test_candidates)}"
        )

    return {
        "train": selected_train,
        "validation": validation_candidates[:validation_speakers],
        "test": test_candidates[:test_speakers],
    }


def _waveform(audio: Any) -> tuple[np.ndarray, int]:
    if not isinstance(audio, Mapping) or "array" not in audio:
        raise ValueError("VoxVietnam audio must contain array and sampling_rate")
    sample_rate = int(audio.get("sampling_rate", 0))
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE} Hz source audio; received {sample_rate} Hz"
        )
    values = np.asarray(audio["array"], dtype=np.float32)
    if values.ndim == 2:
        axis = 0 if values.shape[0] <= 8 else 1
        values = values.mean(axis=axis)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Source audio is empty, non-finite, or not mono-compatible")
    peak = float(np.max(np.abs(values)))
    if peak <= np.finfo(np.float32).eps:
        raise ValueError("Source audio is silent")
    return np.clip(values, -1.0, 1.0), sample_rate


def _verification_trials(
    rows: list[dict[str, Any]],
    *,
    negative_trials_per_query: int,
    seed: int,
) -> list[dict[str, Any]]:
    enrollment_speakers = sorted(
        {row["normalized_speaker_id"] for row in rows if row["role"] == "ENROLLMENT"}
    )
    queries = [row for row in rows if row["role"] == "QUERY"]
    trials: list[dict[str, Any]] = []
    for query in queries:
        query_speaker = query["normalized_speaker_id"]
        targets = [speaker for speaker in enrollment_speakers if speaker != query_speaker]
        targets.sort(
            key=lambda speaker: _stable_key(
                seed, f"{query['audio_id']}:{speaker}"
            )
        )
        selected_negatives = targets[: min(negative_trials_per_query, len(targets))]
        pairs = [(query_speaker, 1), *((speaker, 0) for speaker in selected_negatives)]
        for enrollment_speaker, label in pairs:
            trial_index = len(trials)
            trials.append(
                {
                    "trial_id": f"trial_{trial_index:08d}",
                    "enrollment_speaker_id": enrollment_speaker,
                    "query_audio_path": query["audio_path"],
                    "query_speaker_id": query_speaker,
                    "label": label,
                }
            )
    return trials


def _audio_protocol_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "audio_path": row["audio_path"],
            "speaker_id": row["normalized_speaker_id"],
            "checksum": row["checksum"],
        }
        for row in rows
    ]


def _protocol_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    relative_path: Path,
) -> dict[str, Any]:
    return {
        "path": relative_path.as_posix(),
        "rows": len(rows),
        "sha256": sha256_file(path),
    }


def materialize_voxvietnam_subset(
    records: Iterable[Mapping[str, Any]],
    speaker_splits: Mapping[str, list[str]],
    *,
    output_root: Path,
    train_audio_per_speaker: int = 30,
    evaluation_audio_per_speaker: int = 15,
    enrollment_audio_per_speaker: int = 5,
    negative_trials_per_query: int = 5,
    closed_set_train_audio_per_speaker: int = 20,
    closed_set_validation_audio_per_speaker: int = 5,
    open_set_known_speakers: int = 25,
    max_bytes: int = DEFAULT_MAX_BYTES,
    seed: int = 42,
) -> dict[str, Any]:
    """Write selected records into current project dataset layout."""

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_root}")
    if not 1 <= enrollment_audio_per_speaker < evaluation_audio_per_speaker:
        raise ValueError(
            "enrollment_audio_per_speaker must be positive and smaller than "
            "evaluation_audio_per_speaker"
        )
    closed_test_audio_per_speaker = (
        train_audio_per_speaker
        - closed_set_train_audio_per_speaker
        - closed_set_validation_audio_per_speaker
    )
    if (
        closed_set_train_audio_per_speaker < 1
        or closed_set_validation_audio_per_speaker < 1
        or closed_test_audio_per_speaker < 1
    ):
        raise ValueError(
            "closed-set partitions must each contain at least one audio per speaker"
        )
    if negative_trials_per_query < 1 or max_bytes < 1:
        raise ValueError("Trial count and byte budget must be positive")

    split_sets = {name: set(values) for name, values in speaker_splits.items()}
    if set(split_sets) != {"train", "validation", "test"}:
        raise ValueError("speaker_splits must contain train/validation/test")
    if any(
        left & right
        for index, left in enumerate(split_sets.values())
        for right in list(split_sets.values())[index + 1 :]
    ):
        raise ValueError("Speaker splits must be pairwise disjoint")
    if not 1 <= open_set_known_speakers < len(split_sets["validation"]):
        raise ValueError(
            "open_set_known_speakers must leave known and unknown validation speakers"
        )
    if open_set_known_speakers >= len(split_sets["test"]):
        raise ValueError(
            "open_set_known_speakers must leave known and unknown test speakers"
        )

    speaker_to_split = {
        speaker: split for split, speakers in split_sets.items() for speaker in speakers
    }
    ordered_speakers = [
        speaker for split in ("train", "validation", "test")
        for speaker in sorted(split_sets[split])
    ]
    normalized = {
        speaker: f"voxvi_spk_{index:04d}"
        for index, speaker in enumerate(ordered_speakers, start=1)
    }
    caps = {
        "train": train_audio_per_speaker,
        "validation": evaluation_audio_per_speaker,
        "test": evaluation_audio_per_speaker,
    }
    selected_counts: Counter[str] = Counter()
    metadata: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    total_bytes = 0
    seen_checksums: set[str] = set()
    skipped_duplicate_content = 0

    output_root.mkdir(parents=True)
    try:
        for source_index, record in enumerate(records):
            source_speaker = str(record.get("speaker", "")).strip()
            split = speaker_to_split.get(source_speaker)
            if split is None or selected_counts[source_speaker] >= caps[split]:
                continue

            waveform, sample_rate = _waveform(record.get("audio"))
            normalized_speaker = normalized[source_speaker]
            audio_key = hashlib.sha256(
                f"{source_speaker}:{record.get('_source_partition', '')}:"
                f"{source_index}".encode("utf-8")
            ).hexdigest()[:20]
            audio_id = f"voxvi_{audio_key}"
            relative = (
                Path(split) / "audio" / normalized_speaker / f"{audio_id}.wav"
            )
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, waveform, sample_rate, subtype="PCM_16")
            checksum = sha256_file(destination)
            if checksum in seen_checksums:
                destination.unlink()
                skipped_duplicate_content += 1
                continue
            seen_checksums.add(checksum)
            file_bytes = destination.stat().st_size
            total_bytes += file_bytes
            if total_bytes > max_bytes:
                raise ValueError(
                    f"Dataset byte budget exceeded: {total_bytes} > {max_bytes}"
                )

            speaker_position = selected_counts[source_speaker]
            if split == "train":
                role = "TRAIN"
                protocol = "AAM_TRAIN"
            else:
                role = (
                    "ENROLLMENT"
                    if speaker_position < enrollment_audio_per_speaker
                    else "QUERY"
                )
                protocol = "COSINE_VERIFICATION"
            duration = waveform.size / sample_rate
            metadata[split].append(
                {
                    "audio_id": audio_id,
                    "audio_path": relative.as_posix(),
                    "speaker_id": source_speaker,
                    "normalized_speaker_id": normalized_speaker,
                    "split": split,
                    "split_name": f"voxvietnam_{split}",
                    "split_role": role,
                    "protocol": protocol,
                    "role": role,
                    "duration_sec": f"{duration:.6f}",
                    "sample_rate": sample_rate,
                    "num_channels": 1,
                    "checksum": checksum,
                    "source_dataset": HF_DATASET,
                    "source_partition": record.get("_source_partition", ""),
                    "source_index": source_index,
                }
            )
            selected_counts[source_speaker] += 1

            if all(
                selected_counts[speaker] >= caps[speaker_to_split[speaker]]
                for speaker in speaker_to_split
            ):
                break

        missing = {
            speaker: caps[split] - selected_counts[speaker]
            for speaker, split in speaker_to_split.items()
            if selected_counts[speaker] < caps[split]
        }
        if missing:
            preview = dict(list(sorted(missing.items()))[:5])
            raise ValueError(
                f"Source ended before selected speakers reached audio caps: {preview}"
            )

        split_manifest: dict[str, Any] = {}
        for split, rows in metadata.items():
            rows.sort(key=lambda row: (row["normalized_speaker_id"], row["audio_id"]))
            metadata_path = output_root / split / "metadata.csv"
            _write_csv(metadata_path, METADATA_FIELDS, rows)
            details: dict[str, Any] = {
                "audio": len(rows),
                "speakers": len({row["normalized_speaker_id"] for row in rows}),
                "metadata": f"{split}/metadata.csv",
                "metadata_sha256": sha256_file(metadata_path),
            }
            if split in {"validation", "test"}:
                trials = _verification_trials(
                    rows,
                    negative_trials_per_query=negative_trials_per_query,
                    seed=seed + (1 if split == "validation" else 2),
                )
                trials_path = output_root / split / "verification_trials.csv"
                _write_csv(trials_path, TRIAL_FIELDS, trials)
                details.update(
                    {
                        "trials": len(trials),
                        "positive_trials": sum(int(row["label"]) for row in trials),
                        "negative_trials": sum(
                            int(row["label"]) == 0 for row in trials
                        ),
                        "trials_path": f"{split}/verification_trials.csv",
                        "trials_sha256": sha256_file(trials_path),
                    }
                )
            split_manifest[split] = details

        protocols_root = output_root / "protocols"
        protocol_manifest: dict[str, Any] = {
            "closed_set": {}, "verification": {}, "open_set": {}
        }

        train_by_speaker: dict[str, list[dict[str, Any]]] = {}
        for row in metadata["train"]:
            train_by_speaker.setdefault(row["normalized_speaker_id"], []).append(row)
        closed_partitions: dict[str, list[dict[str, Any]]] = {
            "classifier_train.csv": [],
            "validation_queries.csv": [],
            "test_queries.csv": [],
        }
        for speaker in sorted(train_by_speaker):
            rows = sorted(train_by_speaker[speaker], key=lambda row: row["audio_id"])
            train_end = closed_set_train_audio_per_speaker
            valid_end = train_end + closed_set_validation_audio_per_speaker
            closed_partitions["classifier_train.csv"].extend(rows[:train_end])
            closed_partitions["validation_queries.csv"].extend(rows[train_end:valid_end])
            closed_partitions["test_queries.csv"].extend(rows[valid_end:])
        closed_path_sets: list[set[str]] = []
        closed_checksum_sets: list[set[str]] = []
        for filename, selected in closed_partitions.items():
            rows = _audio_protocol_rows(selected)
            relative = Path("protocols") / "closed_set" / filename
            _write_csv(output_root / relative, PROTOCOL_AUDIO_FIELDS, rows)
            protocol_manifest["closed_set"][filename.removesuffix(".csv")] = (
                _protocol_manifest(output_root / relative, rows, relative)
            )
            closed_path_sets.append({row["audio_path"] for row in rows})
            closed_checksum_sets.append({row["checksum"] for row in rows})
        closed_disjoint = all(
            not left & right
            for index, left in enumerate(closed_path_sets)
            for right in closed_path_sets[index + 1 :]
        ) and all(
            not left & right
            for index, left in enumerate(closed_checksum_sets)
            for right in closed_checksum_sets[index + 1 :]
        )
        if not closed_disjoint:
            raise ValueError("Closed-set audio path/checksum leakage detected")

        open_speaker_sets: dict[str, set[str]] = {}
        for split in ("validation", "test"):
            split_rows = metadata[split]
            enrollment = [row for row in split_rows if row["role"] == "ENROLLMENT"]
            verification_trials = _verification_trials(
                split_rows,
                negative_trials_per_query=negative_trials_per_query,
                seed=seed + (1 if split == "validation" else 2),
            )
            enrollment_relative = Path("protocols") / "verification" / f"{split}_enrollment.csv"
            trials_relative = Path("protocols") / "verification" / f"{split}_trials.csv"
            enrollment_rows = _audio_protocol_rows(enrollment)
            _write_csv(output_root / enrollment_relative, PROTOCOL_AUDIO_FIELDS, enrollment_rows)
            _write_csv(output_root / trials_relative, TRIAL_FIELDS, verification_trials)
            protocol_manifest["verification"][f"{split}_enrollment"] = _protocol_manifest(
                output_root / enrollment_relative, enrollment_rows, enrollment_relative
            )
            protocol_manifest["verification"][f"{split}_trials"] = _protocol_manifest(
                output_root / trials_relative, verification_trials, trials_relative
            )

            speakers = sorted(
                {row["normalized_speaker_id"] for row in split_rows},
                key=lambda value: _stable_key(seed + (10 if split == "validation" else 20), value),
            )
            known_speakers = set(speakers[:open_set_known_speakers])
            open_speaker_sets[split] = set(speakers)
            gallery_selected = [
                row for row in enrollment
                if row["normalized_speaker_id"] in known_speakers
            ]
            query_selected = [
                row for row in split_rows if row["role"] == "QUERY"
            ]
            gallery_rows = _audio_protocol_rows(gallery_selected)
            query_rows = [
                {
                    "query_audio_path": row["audio_path"],
                    "query_speaker_id": row["normalized_speaker_id"],
                    "is_known": int(row["normalized_speaker_id"] in known_speakers),
                    "checksum": row["checksum"],
                }
                for row in query_selected
            ]
            gallery_relative = Path("protocols") / "open_set" / f"{split}_gallery.csv"
            queries_relative = Path("protocols") / "open_set" / f"{split}_queries.csv"
            _write_csv(output_root / gallery_relative, PROTOCOL_AUDIO_FIELDS, gallery_rows)
            _write_csv(output_root / queries_relative, OPEN_QUERY_FIELDS, query_rows)
            protocol_manifest["open_set"][f"{split}_gallery"] = _protocol_manifest(
                output_root / gallery_relative, gallery_rows, gallery_relative
            )
            protocol_manifest["open_set"][f"{split}_queries"] = _protocol_manifest(
                output_root / queries_relative, query_rows, queries_relative
            )
        if open_speaker_sets["validation"] & open_speaker_sets["test"]:
            raise ValueError("Open-set validation/test speaker leakage detected")

        mapping_rows = [
            {
                "speaker_id": speaker,
                "normalized_speaker_id": normalized[speaker],
                "split": speaker_to_split[speaker],
            }
            for speaker in ordered_speakers
        ]
        _write_csv(
            output_root / "speaker_mapping.csv",
            ("speaker_id", "normalized_speaker_id", "split"),
            mapping_rows,
        )

        manifest = {
            "dataset": "voxvietnam_ecapa_three_task_v1",
            "source_dataset": HF_DATASET,
            "license": "CC-BY-NC-4.0; gated source terms also apply",
            "seed": seed,
            "audio_format": "WAV PCM16 mono 16 kHz",
            "total_audio": sum(len(rows) for rows in metadata.values()),
            "total_audio_bytes": total_bytes,
            "maximum_audio_bytes": max_bytes,
            "skipped_duplicate_content": skipped_duplicate_content,
            "train_ground_truth": "normalized_speaker_id class",
            "verification_ground_truth": "binary trial label: 1 same, 0 different",
            "closed_set_ground_truth": "speaker_id class shared across classifier train/validation/test",
            "open_set_ground_truth": "query_speaker_id plus is_known membership",
            "enrollment_audio_per_speaker": enrollment_audio_per_speaker,
            "negative_trials_per_query": negative_trials_per_query,
            "splits": split_manifest,
            "protocols": protocol_manifest,
            "protocol_configuration": {
                "closed_set_train_audio_per_speaker": closed_set_train_audio_per_speaker,
                "closed_set_validation_audio_per_speaker": closed_set_validation_audio_per_speaker,
                "closed_set_test_audio_per_speaker": closed_test_audio_per_speaker,
                "open_set_known_speakers_per_split": open_set_known_speakers,
            },
            "invariants": {
                "speaker_disjoint": True,
                "duplicate_checksum": 0,
                "validation_selects_checkpoint_and_threshold": True,
                "test_used_once_for_final_evaluation": True,
                "byte_budget_respected": total_bytes <= max_bytes,
                "closed_set_audio_and_checksum_disjoint": closed_disjoint,
                "open_set_validation_test_speakers_disjoint": True,
                "open_set_unknown_absent_from_gallery": True,
            },
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_root / "README.md").write_text(
            "# VoxVietnam ECAPA three-task subset v1\n\n"
            "Compact subset for one ECAPA checkpoint serving closed-set "
            "identification, claimed-centroid verification, and rejected "
            "open-set identification. Ground truth lives under `protocols/`.\n\n"
            "Source: `hustep-lab/VoxVietnam-Dataset`. License: CC BY-NC 4.0. "
            "The source is gated; do not publish this derived package unless the "
            "source access terms permit redistribution.\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(output_root)
        raise


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        get_token = None
    cached = get_token() if get_token is not None else None
    if not cached:
        raise RuntimeError(
            "HF_TOKEN is required. Accept VoxVietnam access terms, then run "
            "`hf auth login` or configure a Kaggle Secret named HF_TOKEN."
        )
    return cached


def _hf_speaker_counts(split: str, token: str) -> Counter[str]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face datasets is required; install project requirements"
        ) from error
    dataset = load_dataset(
        HF_DATASET,
        split=split,
        streaming=True,
        token=token,
    ).select_columns(["speaker"])
    return Counter(str(row["speaker"]) for row in dataset)


def _hf_records(split: str, token: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        HF_DATASET,
        split=split,
        streaming=True,
        token=token,
    ).select_columns(["audio", "speaker"])
    for row in dataset:
        yield {**row, "_source_partition": split}


def build_from_huggingface(
    *,
    output_root: Path,
    training_source_split: str = "train_small",
    test_source_split: str = "test",
    train_speakers: int = 230,
    validation_speakers: int = 50,
    test_speakers: int = 50,
    train_audio_per_speaker: int = 30,
    evaluation_audio_per_speaker: int = 15,
    enrollment_audio_per_speaker: int = 5,
    negative_trials_per_query: int = 5,
    closed_set_train_audio_per_speaker: int = 20,
    closed_set_validation_audio_per_speaker: int = 5,
    open_set_known_speakers: int = 25,
    max_bytes: int = DEFAULT_MAX_BYTES,
    seed: int = 42,
) -> dict[str, Any]:
    """Scan gated source metadata, select speakers, then stream selected audio."""

    token = _hf_token()
    training_counts = _hf_speaker_counts(training_source_split, token)
    test_counts = _hf_speaker_counts(test_source_split, token)
    splits = select_speaker_splits(
        training_counts,
        test_counts,
        train_speakers=train_speakers,
        validation_speakers=validation_speakers,
        test_speakers=test_speakers,
        train_audio_per_speaker=train_audio_per_speaker,
        evaluation_audio_per_speaker=evaluation_audio_per_speaker,
        seed=seed,
    )
    records = chain(
        _hf_records(training_source_split, token),
        _hf_records(test_source_split, token),
    )
    return materialize_voxvietnam_subset(
        records,
        splits,
        output_root=output_root,
        train_audio_per_speaker=train_audio_per_speaker,
        evaluation_audio_per_speaker=evaluation_audio_per_speaker,
        enrollment_audio_per_speaker=enrollment_audio_per_speaker,
        negative_trials_per_query=negative_trials_per_query,
        closed_set_train_audio_per_speaker=closed_set_train_audio_per_speaker,
        closed_set_validation_audio_per_speaker=closed_set_validation_audio_per_speaker,
        open_set_known_speakers=open_set_known_speakers,
        max_bytes=max_bytes,
        seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/voxvietnam_ecapa_three_task_v1"),
    )
    parser.add_argument("--training-source-split", default="train_small")
    parser.add_argument("--test-source-split", default="test")
    parser.add_argument("--train-speakers", type=int, default=230)
    parser.add_argument("--validation-speakers", type=int, default=50)
    parser.add_argument("--test-speakers", type=int, default=50)
    parser.add_argument("--train-audio-per-speaker", type=int, default=30)
    parser.add_argument("--evaluation-audio-per-speaker", type=int, default=15)
    parser.add_argument("--enrollment-audio-per-speaker", type=int, default=5)
    parser.add_argument("--negative-trials-per-query", type=int, default=5)
    parser.add_argument("--closed-set-train-audio-per-speaker", type=int, default=20)
    parser.add_argument("--closed-set-validation-audio-per-speaker", type=int, default=5)
    parser.add_argument("--open-set-known-speakers", type=int, default=25)
    parser.add_argument("--max-gib", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_from_huggingface(
        output_root=args.output_root,
        training_source_split=args.training_source_split,
        test_source_split=args.test_source_split,
        train_speakers=args.train_speakers,
        validation_speakers=args.validation_speakers,
        test_speakers=args.test_speakers,
        train_audio_per_speaker=args.train_audio_per_speaker,
        evaluation_audio_per_speaker=args.evaluation_audio_per_speaker,
        enrollment_audio_per_speaker=args.enrollment_audio_per_speaker,
        negative_trials_per_query=args.negative_trials_per_query,
        closed_set_train_audio_per_speaker=args.closed_set_train_audio_per_speaker,
        closed_set_validation_audio_per_speaker=args.closed_set_validation_audio_per_speaker,
        open_set_known_speakers=args.open_set_known_speakers,
        max_bytes=int(args.max_gib * 1024**3),
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
