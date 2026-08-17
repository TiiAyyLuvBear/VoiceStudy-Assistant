"""Build a self-contained Kaggle notebook for preparing compact VoxVietnam."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_voxvietnam_verification_dataset.py"
OUTPUT = ROOT / "notebooks" / "prepare-voxvietnam-on-kaggle.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def embedded_builder() -> str:
    source = BUILDER.read_text(encoding="utf-8")
    source = source.replace(
        "from src.utils.files import sha256_file",
        dedent(
            """
            def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(chunk_size), b""):
                        digest.update(chunk)
                return digest.hexdigest()
            """
        ).strip(),
    )
    main_guard = '\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
    if main_guard not in source:
        raise RuntimeError("Builder main guard changed; update notebook generator")
    return source.replace(main_guard, "\n")


def build_cells() -> list[dict]:
    return [
        markdown(
            """
            # Prepare three-task VoxVietnam protocols from Kaggle Input

            This notebook reads the existing sharded Parquet dataset from
            `/kaggle/input/voxvietnam-dataset`. It scans metadata first, performs
            streaming audio quality control, selects leakage-safe protocols, and
            writes only the compact derived dataset to `/kaggle/working`.

            Output protocol:

            - 230 closed-set speakers, 30 audio each: 20 train / 5 validation / 5 test.
            - 50 validation speakers, 15 audio each.
            - 50 official-test speakers, 15 audio each.
            - Validation/test: 5 enrollment audio per speaker; remaining audio are queries.
            - One positive and five negative trials per query.
            - 25 known-gallery and 25 unknown speakers per open-set split.
            - Speaker-disjoint train/validation/test and hard 12 GiB audio limit.

            The full redundant `train` split is inventoried but never decoded.
            `train_small` supplies train/validation speakers; official `test`
            supplies held-out test speakers. Keep output private unless gated
            redistribution terms explicitly permit publication.
            """
        ),
        markdown("## 1. Install dependencies"),
        code('!pip install -q "datasets[audio]" soundfile scipy pyarrow tqdm'),
        markdown("## 2. Input discovery and configuration"),
        code(
            """
            import json
            import shutil
            from pathlib import Path

            INPUT_ROOT = Path("/kaggle/input/voxvietnam-dataset")
            OUTPUT_ROOT = Path(
                "/kaggle/working/voxvietnam_ecapa_three_task_v1"
            )
            QC_SAMPLE_PATH = Path("/kaggle/working/voxvietnam_qc_sample.csv")
            QC_INVENTORY_PATH = Path("/kaggle/working/voxvietnam_qc_inventory.csv")
            SELECTION_PLAN_PATH = Path(
                "/kaggle/working/voxvietnam_selection_plan.csv"
            )

            # Reserve 230 speakers with 30 valid audio for training. Validation
            # speakers are different identities and need only 15 valid audio.
            TRAIN_SPEAKERS = 230
            VALIDATION_SPEAKERS = 50
            TEST_SPEAKERS = 50
            TRAIN_AUDIO_PER_SPEAKER = 30
            EVALUATION_AUDIO_PER_SPEAKER = 15
            ENROLLMENT_AUDIO_PER_SPEAKER = 5
            NEGATIVE_TRIALS_PER_QUERY = 5
            MAX_BYTES = 12 * 1024**3
            CLOSED_TRAIN_AUDIO = 20
            CLOSED_VALIDATION_AUDIO = 5
            OPEN_SET_KNOWN_SPEAKERS = 25
            MIN_DURATION_SEC = 2.0
            MAX_DURATION_SEC = 10.0
            SILENCE_TOP_DB = 35.0
            TRIM_PAD_SEC = 0.10
            QC_SAMPLE_PER_PARTITION = 250
            SEED = 42

            assert INPUT_ROOT.is_dir(), f"Missing Kaggle input: {INPUT_ROOT}"
            assert not OUTPUT_ROOT.exists(), (
                f"Refusing to overwrite existing output: {OUTPUT_ROOT}"
            )
            disk = shutil.disk_usage("/kaggle/working")
            print(f"Working disk free: {disk.free / 1024**3:.2f} GiB")
            assert disk.free > MAX_BYTES + 4 * 1024**3, (
                "Need at least 4 GiB working headroom beyond audio budget"
            )
            """
        ),
        markdown("## 3. Self-contained streaming builder"),
        code(embedded_builder()),
        markdown("## 4. Discover shards and audit speaker metadata"),
        code(
            """
            from collections import Counter

            import pandas as pd
            import pyarrow.parquet as pq

            SOURCE_FILES = {
                "train_small": sorted(INPUT_ROOT.glob("train_small-*.parquet")),
                "train": sorted(INPUT_ROOT.glob("train-*.parquet")),
                "test": sorted(INPUT_ROOT.glob("test-*.parquet")),
            }
            assert SOURCE_FILES["train_small"], "Missing train_small Parquet shards"
            assert SOURCE_FILES["test"], "Missing test Parquet shards"

            source_summary = []
            for split, files in SOURCE_FILES.items():
                rows = 0
                size = 0
                for path in files:
                    parquet = pq.ParquetFile(path)
                    names = set(parquet.schema_arrow.names)
                    assert {"audio", "speaker"} <= names, (
                        f"{path.name} missing audio/speaker: {sorted(names)}"
                    )
                    rows += parquet.metadata.num_rows
                    size += path.stat().st_size
                source_summary.append({
                    "partition": split,
                    "shards": len(files),
                    "rows": rows,
                    "gib": size / 1024**3,
                })
            source_summary = pd.DataFrame(source_summary)
            display(source_summary)


            def parquet_speaker_counts(files):
                counts = Counter()
                missing = 0
                for path in files:
                    parquet = pq.ParquetFile(path)
                    for batch in parquet.iter_batches(
                        columns=["speaker"], batch_size=262_144, use_threads=True
                    ):
                        index = batch.schema.get_field_index("speaker")
                        for value in batch.column(index).to_pylist():
                            speaker = "" if value is None else str(value).strip()
                            if speaker:
                                counts[speaker] += 1
                            else:
                                missing += 1
                return counts, missing


            raw_counts = {}
            missing_speakers = {}
            for split in ("train_small", "test"):
                raw_counts[split], missing_speakers[split] = parquet_speaker_counts(
                    SOURCE_FILES[split]
                )

            audit_rows = []
            for split, counts in raw_counts.items():
                for speaker, audio_count in counts.items():
                    audit_rows.append({
                        "partition": split,
                        "speaker": speaker,
                        "audio_count": audio_count,
                        "eligible_15": audio_count >= EVALUATION_AUDIO_PER_SPEAKER,
                        "eligible_30": audio_count >= TRAIN_AUDIO_PER_SPEAKER,
                    })
            speaker_audit = pd.DataFrame(audit_rows)
            speaker_audit.to_csv(
                "/kaggle/working/voxvietnam_speaker_audit.csv", index=False
            )
            display(
                speaker_audit.groupby("partition").agg(
                    speakers=("speaker", "nunique"),
                    eligible_15=("eligible_15", "sum"),
                    eligible_30=("eligible_30", "sum"),
                    min_audio=("audio_count", "min"),
                    median_audio=("audio_count", "median"),
                    max_audio=("audio_count", "max"),
                )
            )

            source_overlap = set(raw_counts["train_small"]) & set(raw_counts["test"])
            assert not source_overlap, (
                f"Speaker overlap between train_small/test: {len(source_overlap)}"
            )
            assert not any(missing_speakers.values()), missing_speakers
            """
        ),
        markdown("## 5. Define streaming preprocessing and QC"),
        code(
            """
            import csv
            import io
            from math import gcd

            import soundfile as sf
            from datasets import load_dataset
            from scipy.signal import resample_poly
            from tqdm.auto import tqdm

            QC_FIELDS = (
                "row_id", "source_partition", "source_file", "source_row",
                "speaker", "status", "reason", "original_sample_rate",
                "original_channels", "duration_sec", "content_hash",
            )


            def source_records(split):
                for path in SOURCE_FILES[split]:
                    relative = path.relative_to(INPUT_ROOT).as_posix()
                    dataset = load_dataset(
                        "parquet",
                        data_files={"source": str(path)},
                        split="source",
                        streaming=True,
                    ).select_columns(["audio", "speaker"])
                    for row_index, row in enumerate(dataset):
                        yield {
                            "speaker": "" if row["speaker"] is None
                            else str(row["speaker"]).strip(),
                            "audio": row["audio"],
                            "_source_partition": split,
                            "_source_file": relative,
                            "_source_row": row_index,
                        }


            def decode_audio(audio):
                if hasattr(audio, "get_all_samples"):
                    samples = audio.get_all_samples()
                    values = samples.data
                    if hasattr(values, "detach"):
                        values = values.detach().cpu().numpy()
                    return np.asarray(values, dtype=np.float32), int(
                        samples.sample_rate
                    )

                if isinstance(audio, Mapping) and "array" in audio:
                    return (
                        np.asarray(audio["array"], dtype=np.float32),
                        int(audio.get("sampling_rate", 0)),
                    )

                if isinstance(audio, Mapping) and audio.get("bytes") is not None:
                    values, sample_rate = sf.read(
                        io.BytesIO(audio["bytes"]),
                        dtype="float32",
                        always_2d=False,
                    )
                    return np.asarray(values), int(sample_rate)

                if isinstance(audio, Mapping) and audio.get("path"):
                    values, sample_rate = sf.read(
                        audio["path"], dtype="float32", always_2d=False
                    )
                    return np.asarray(values), int(sample_rate)

                raise ValueError("Unsupported audio representation")


            def preprocess_audio(audio):
                values, original_sample_rate = decode_audio(audio)
                if original_sample_rate <= 0:
                    raise ValueError("invalid_sample_rate")
                if values.ndim == 1:
                    original_channels = 1
                elif values.ndim == 2:
                    original_channels = min(values.shape)
                    axis = 0 if values.shape[0] <= 8 else 1
                    values = values.mean(axis=axis)
                else:
                    raise ValueError("invalid_shape")
                values = np.asarray(values, dtype=np.float32)
                if values.ndim != 1 or values.size == 0:
                    raise ValueError("empty_audio")
                if not np.isfinite(values).all():
                    raise ValueError("non_finite_audio")

                peak = float(np.max(np.abs(values)))
                if peak <= np.finfo(np.float32).eps:
                    raise ValueError("silent_audio")
                threshold = peak * 10 ** (-SILENCE_TOP_DB / 20.0)
                active = np.flatnonzero(np.abs(values) >= threshold)
                if active.size == 0:
                    raise ValueError("no_active_audio")
                pad = int(round(TRIM_PAD_SEC * original_sample_rate))
                start = max(0, int(active[0]) - pad)
                stop = min(values.size, int(active[-1]) + pad + 1)
                values = values[start:stop]

                if original_sample_rate != TARGET_SAMPLE_RATE:
                    divisor = gcd(original_sample_rate, TARGET_SAMPLE_RATE)
                    values = resample_poly(
                        values,
                        TARGET_SAMPLE_RATE // divisor,
                        original_sample_rate // divisor,
                    ).astype(np.float32, copy=False)

                minimum = int(round(MIN_DURATION_SEC * TARGET_SAMPLE_RATE))
                maximum = int(round(MAX_DURATION_SEC * TARGET_SAMPLE_RATE))
                if values.size < minimum:
                    raise ValueError("duration_below_minimum")
                if values.size > maximum:
                    offset = (values.size - maximum) // 2
                    values = values[offset : offset + maximum]

                values = np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)
                pcm16 = np.rint(values * 32767.0).astype("<i2", copy=False)
                return {
                    "waveform": values,
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "original_sample_rate": original_sample_rate,
                    "original_channels": original_channels,
                    "duration_sec": values.size / TARGET_SAMPLE_RATE,
                    "content_hash": hashlib.sha256(pcm16.tobytes()).hexdigest(),
                }


            # Override embedded builder preprocessing. The materializer resolves this
            # global function at runtime and therefore writes the exact QC waveform.
            def _waveform(audio):
                processed = preprocess_audio(audio)
                return processed["waveform"], processed["sample_rate"]


            def scan_audio_quality(output_path, limit_per_partition=None):
                seen_content = set()
                status_counts = Counter()
                partition_totals = {
                    split: sum(
                        pq.ParquetFile(path).metadata.num_rows
                        for path in SOURCE_FILES[split]
                    )
                    for split in ("train_small", "test")
                }
                with output_path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=QC_FIELDS)
                    writer.writeheader()
                    for split in ("train_small", "test"):
                        total = partition_totals[split]
                        if limit_per_partition is not None:
                            total = min(total, limit_per_partition)
                        progress = tqdm(total=total, desc=f"QC {split}")
                        for partition_index, record in enumerate(source_records(split)):
                            if (
                                limit_per_partition is not None
                                and partition_index >= limit_per_partition
                            ):
                                break
                            row_id = hashlib.sha256(
                                f"{split}:{record['_source_file']}:"
                                f"{record['_source_row']}".encode("utf-8")
                            ).hexdigest()[:24]
                            result = {
                                "row_id": row_id,
                                "source_partition": split,
                                "source_file": record["_source_file"],
                                "source_row": record["_source_row"],
                                "speaker": record["speaker"],
                                "status": "invalid",
                                "reason": "",
                                "original_sample_rate": "",
                                "original_channels": "",
                                "duration_sec": "",
                                "content_hash": "",
                            }
                            try:
                                if not record["speaker"]:
                                    raise ValueError("missing_speaker")
                                processed = preprocess_audio(record["audio"])
                                result.update({
                                    "original_sample_rate": processed[
                                        "original_sample_rate"
                                    ],
                                    "original_channels": processed[
                                        "original_channels"
                                    ],
                                    "duration_sec": f"{processed['duration_sec']:.6f}",
                                    "content_hash": processed["content_hash"],
                                })
                                if processed["content_hash"] in seen_content:
                                    result["status"] = "duplicate"
                                    result["reason"] = "duplicate_content"
                                else:
                                    seen_content.add(processed["content_hash"])
                                    result["status"] = "valid"
                                    result["reason"] = "ok"
                            except Exception as error:
                                result["reason"] = str(error)
                            writer.writerow(result)
                            status_counts[(split, result["status"], result["reason"])] += 1
                            progress.update(1)
                        progress.close()
                return status_counts
            """
        ),
        markdown("## 6. Sample QC before full scan"),
        code(
            """
            sample_status = scan_audio_quality(
                QC_SAMPLE_PATH,
                limit_per_partition=QC_SAMPLE_PER_PARTITION,
            )
            sample_qc = pd.read_csv(QC_SAMPLE_PATH)
            display(
                sample_qc.groupby(
                    ["source_partition", "status", "reason"], dropna=False
                ).size().rename("rows").reset_index()
            )
            display(
                sample_qc.loc[sample_qc["status"] == "valid"].groupby(
                    "source_partition"
                ).agg(
                    audio=("row_id", "count"),
                    median_duration=("duration_sec", "median"),
                    min_duration=("duration_sec", "min"),
                    max_duration=("duration_sec", "max"),
                )
            )
            assert (sample_qc["status"] == "valid").any(), "No valid sample audio"
            """
        ),
        markdown(
            """
            ## 7. Full streaming QC inventory

            This is the long CPU step. It decodes `train_small` and `test` once,
            never loads an entire Parquet shard into RAM, and does not touch full
            `train`. Restart from this cell only if the sample QC looks reasonable.
            """
        ),
        code(
            """
            full_status = scan_audio_quality(QC_INVENTORY_PATH)
            quality = pd.read_csv(QC_INVENTORY_PATH)
            display(
                quality.groupby(
                    ["source_partition", "status", "reason"], dropna=False
                ).size().rename("rows").reset_index()
            )

            valid_quality = quality.loc[quality["status"] == "valid"].copy()
            valid_counts_frame = (
                valid_quality.groupby(["source_partition", "speaker"])
                .size()
                .rename("valid_audio")
                .reset_index()
            )
            display(
                valid_counts_frame.groupby("source_partition").agg(
                    speakers=("speaker", "nunique"),
                    eligible_15=(
                        "valid_audio",
                        lambda values: int((values >= EVALUATION_AUDIO_PER_SPEAKER).sum()),
                    ),
                    eligible_30=(
                        "valid_audio",
                        lambda values: int((values >= TRAIN_AUDIO_PER_SPEAKER).sum()),
                    ),
                    median_valid_audio=("valid_audio", "median"),
                )
            )
            """
        ),
        markdown("## 8. Freeze deterministic speaker and audio selection"),
        code(
            """
            quality_counts = {
                split: Counter(dict(
                    valid_counts_frame.loc[
                        valid_counts_frame["source_partition"] == split,
                        ["speaker", "valid_audio"],
                    ].itertuples(index=False, name=None)
                ))
                for split in ("train_small", "test")
            }
            speaker_splits = select_speaker_splits(
                quality_counts["train_small"],
                quality_counts["test"],
                train_speakers=TRAIN_SPEAKERS,
                validation_speakers=VALIDATION_SPEAKERS,
                test_speakers=TEST_SPEAKERS,
                train_audio_per_speaker=TRAIN_AUDIO_PER_SPEAKER,
                evaluation_audio_per_speaker=EVALUATION_AUDIO_PER_SPEAKER,
                seed=SEED,
            )

            speaker_to_split = {
                speaker: split
                for split, speakers in speaker_splits.items()
                for speaker in speakers
            }
            caps = {
                "train": TRAIN_AUDIO_PER_SPEAKER,
                "validation": EVALUATION_AUDIO_PER_SPEAKER,
                "test": EVALUATION_AUDIO_PER_SPEAKER,
            }
            plan_parts = []
            for speaker, dataset_split in speaker_to_split.items():
                source_partition = (
                    "test" if dataset_split == "test" else "train_small"
                )
                candidates = valid_quality.loc[
                    (valid_quality["source_partition"] == source_partition)
                    & (valid_quality["speaker"] == speaker)
                ].copy()
                candidates["selection_key"] = candidates["row_id"].map(
                    lambda value: _stable_key(SEED, str(value))
                )
                selected = candidates.sort_values("selection_key").head(
                    caps[dataset_split]
                )
                assert len(selected) == caps[dataset_split]
                selected["dataset_split"] = dataset_split
                plan_parts.append(selected)

            selection_plan = pd.concat(plan_parts, ignore_index=True)
            assert selection_plan["row_id"].is_unique
            assert selection_plan["content_hash"].is_unique
            selection_plan.to_csv(SELECTION_PLAN_PATH, index=False)
            display(
                selection_plan.groupby("dataset_split").agg(
                    audio=("row_id", "count"),
                    speakers=("speaker", "nunique"),
                    duration_hours=("duration_sec", lambda x: x.sum() / 3600),
                )
            )
            """
        ),
        markdown("## 9. Materialize selected WAV files and create three protocols"),
        code(
            """
            def planned_records(plan):
                for source_file, group in plan.groupby("source_file", sort=True):
                    wanted = {
                        int(row.source_row): row
                        for row in group.itertuples(index=False)
                    }
                    path = INPUT_ROOT / source_file
                    dataset = load_dataset(
                        "parquet",
                        data_files={"source": str(path)},
                        split="source",
                        streaming=True,
                    ).select_columns(["audio", "speaker"])
                    found = 0
                    for row_index, row in enumerate(dataset):
                        planned = wanted.get(row_index)
                        if planned is None:
                            continue
                        speaker = "" if row["speaker"] is None else str(
                            row["speaker"]
                        ).strip()
                        assert speaker == planned.speaker
                        found += 1
                        yield {
                            "speaker": speaker,
                            "audio": row["audio"],
                            "_source_partition": planned.source_partition,
                            "_source_file": source_file,
                            "_source_row": row_index,
                        }
                    assert found == len(wanted), (
                        f"Missing planned rows in {source_file}: "
                        f"{found}/{len(wanted)}"
                    )


            manifest = materialize_voxvietnam_subset(
                planned_records(selection_plan),
                speaker_splits,
                output_root=OUTPUT_ROOT,
                train_audio_per_speaker=TRAIN_AUDIO_PER_SPEAKER,
                evaluation_audio_per_speaker=EVALUATION_AUDIO_PER_SPEAKER,
                enrollment_audio_per_speaker=ENROLLMENT_AUDIO_PER_SPEAKER,
                negative_trials_per_query=NEGATIVE_TRIALS_PER_QUERY,
                max_bytes=MAX_BYTES,
                closed_set_train_audio_per_speaker=CLOSED_TRAIN_AUDIO,
                closed_set_validation_audio_per_speaker=CLOSED_VALIDATION_AUDIO,
                open_set_known_speakers=OPEN_SET_KNOWN_SPEAKERS,
                seed=SEED,
            )

            shutil.copy2(
                SELECTION_PLAN_PATH,
                OUTPUT_ROOT / "source_selection_plan.csv",
            )
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## 10. Inspect current files and create package metadata/manifest"),
        code(
            """
            import hashlib
            import pandas as pd

            GENERATED_PACKAGE_FILES = {
                "dataset_metadata.json",
                "file_manifest.csv",
            }
            required_files = [
                "manifest.json",
                "speaker_mapping.csv",
                "train/metadata.csv",
                "validation/metadata.csv",
                "test/metadata.csv",
                "protocols/closed_set/classifier_train.csv",
                "protocols/closed_set/validation_queries.csv",
                "protocols/closed_set/test_queries.csv",
                "protocols/verification/validation_enrollment.csv",
                "protocols/verification/validation_trials.csv",
                "protocols/verification/test_enrollment.csv",
                "protocols/verification/test_trials.csv",
                "protocols/open_set/validation_gallery.csv",
                "protocols/open_set/validation_queries.csv",
                "protocols/open_set/test_gallery.csv",
                "protocols/open_set/test_queries.csv",
            ]


            def sha256_file_status(path, chunk_size=1024 * 1024):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(chunk_size), b""):
                        digest.update(chunk)
                return digest.hexdigest()


            assert OUTPUT_ROOT.is_dir(), f"Output directory not found: {OUTPUT_ROOT}"

            status = pd.DataFrame([
                {
                    "path": relative,
                    "exists": (OUTPUT_ROOT / relative).is_file(),
                    "bytes": (
                        (OUTPUT_ROOT / relative).stat().st_size
                        if (OUTPUT_ROOT / relative).is_file()
                        else 0
                    ),
                }
                for relative in required_files
            ])
            display(status)

            missing_required = status.loc[~status["exists"], "path"].tolist()
            source_manifest_path = OUTPUT_ROOT / "manifest.json"
            source_manifest = (
                json.loads(source_manifest_path.read_text(encoding="utf-8"))
                if source_manifest_path.is_file()
                else None
            )

            audio_checksums = {}
            split_summary = []
            missing_audio = []
            for split in ("train", "validation", "test"):
                metadata_path = OUTPUT_ROOT / split / "metadata.csv"
                if not metadata_path.is_file():
                    split_summary.append({
                        "split": split,
                        "metadata": "missing",
                        "audio": 0,
                        "speakers": 0,
                        "missing_audio": None,
                    })
                    continue

                frame = pd.read_csv(metadata_path)
                required_columns = {
                    "audio_path", "normalized_speaker_id", "checksum"
                }
                absent_columns = required_columns - set(frame.columns)
                if absent_columns:
                    raise ValueError(
                        f"{metadata_path} missing columns: {sorted(absent_columns)}"
                    )

                split_missing = [
                    str(relative)
                    for relative in frame["audio_path"]
                    if not (OUTPUT_ROOT / str(relative)).is_file()
                ]
                missing_audio.extend(split_missing)
                audio_checksums.update({
                    str(row.audio_path): str(row.checksum)
                    for row in frame.itertuples(index=False)
                })
                split_summary.append({
                    "split": split,
                    "metadata": "present",
                    "audio": len(frame),
                    "speakers": frame["normalized_speaker_id"].nunique(),
                    "missing_audio": len(split_missing),
                })

            display(pd.DataFrame(split_summary))

            inventory = []
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(OUTPUT_ROOT).as_posix()
                if relative in GENERATED_PACKAGE_FILES:
                    continue
                checksum = audio_checksums.get(relative)
                if checksum is None:
                    checksum = sha256_file_status(path)
                inventory.append({
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": checksum,
                })

            file_manifest = pd.DataFrame(
                inventory, columns=["path", "bytes", "sha256"]
            )
            file_manifest.to_csv(OUTPUT_ROOT / "file_manifest.csv", index=False)

            total_bytes = int(file_manifest["bytes"].sum()) if len(file_manifest) else 0
            package_complete = not missing_required and not missing_audio
            dataset_metadata = {
                "dataset": (
                    source_manifest.get("dataset")
                    if source_manifest is not None
                    else "voxvietnam_ecapa_three_task_v1"
                ),
                "status": "complete" if package_complete else "incomplete",
                "root": str(OUTPUT_ROOT),
                "files": int(len(file_manifest)),
                "audio_files": int(sum(
                    str(path).lower().endswith(".wav")
                    for path in file_manifest["path"]
                )),
                "total_bytes": total_bytes,
                "total_gib": total_bytes / 1024**3,
                "required_files": len(required_files),
                "missing_required_files": missing_required,
                "missing_referenced_audio": missing_audio,
                "splits": split_summary,
                "source_manifest": "manifest.json" if source_manifest else None,
                "file_manifest": "file_manifest.csv",
            }
            (OUTPUT_ROOT / "dataset_metadata.json").write_text(
                json.dumps(dataset_metadata, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )

            print(json.dumps(dataset_metadata, ensure_ascii=False, indent=2))
            print("Created:", OUTPUT_ROOT / "dataset_metadata.json")
            print("Created:", OUTPUT_ROOT / "file_manifest.csv")

            if package_complete:
                assert source_manifest is not None
                assert source_manifest["dataset"] == "voxvietnam_ecapa_three_task_v1"
                assert source_manifest["total_audio_bytes"] <= MAX_BYTES
                assert source_manifest["invariants"]["speaker_disjoint"] is True
                assert source_manifest["invariants"]["duplicate_checksum"] == 0
                assert source_manifest["invariants"]["byte_budget_respected"] is True
                print("Package complete and ready:", OUTPUT_ROOT)
            else:
                print("Package incomplete. Missing required files:", missing_required)
                print("Missing referenced audio:", missing_audio[:10])
            """
        ),
        markdown(
            """
            ## 11. Save to Kaggle

            Do not ZIP output; ZIP temporarily duplicates disk usage.

            1. Click **Save Version** and select **Save & Run All**.
            2. After run completes, open notebook **Output**.
            3. Create a new **private** Kaggle Dataset from folder
               `voxvietnam_ecapa_three_task_v1`.
            4. Attach that dataset to
               `finetune-ecapa-for-vietnamese-speaker-indentificat.ipynb`.

            Fine-tuning notebook discovers dataset through `manifest.json`; Kaggle
            slug and nesting depth do not matter.
            """
        ),
    ]


def main() -> None:
    notebook = {
        "cells": build_cells(),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
