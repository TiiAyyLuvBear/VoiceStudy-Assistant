"""Train leakage-safe Speaker v2 SVM artifacts without touching v1 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_speaker_experiments import (
    EXPECTED_EMBEDDING_DIM,
    SVM_ENROLLMENT_SPLIT,
    SVM_PROTOCOL,
    _read_csv,
    _write_json,
    build_centroids,
    build_embedding_quality_report,
    build_svm_train_features,
    train_and_select_svm,
)
from src.utils import sha256_file


DEFAULT_EMBEDDING_METADATA = Path("data/processed/v2/embedding_metadata.csv")
DEFAULT_MODEL_DIR = Path("models/experimental/v2")
DEFAULT_EXPERIMENT_DIR = Path("experiments/v2")


def _lock_v1_threshold(
    source: Path,
    destination: Path,
    *,
    evaluation_dataset: str,
) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    locked = {
        **payload,
        "evaluation_dataset": evaluation_dataset,
        "selection_dataset_version": "v1",
        "selection_source": source.as_posix(),
        "selection_source_sha256": sha256_file(source),
        "threshold_tuned_on_v2_test": False,
    }
    _write_json(destination, locked)
    return locked


def build_v2_artifacts(
    embedding_metadata: Path = DEFAULT_EMBEDDING_METADATA,
    model_dir: Path = DEFAULT_MODEL_DIR,
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
) -> dict[str, object]:
    rows = _read_csv(embedding_metadata)
    quality_path = experiment_dir / "embedding_quality_report.csv"
    quality = build_embedding_quality_report(rows, output_path=quality_path)
    invalid = [row for row in quality if row["valid"] != "true"]
    if invalid:
        raise ValueError(f"Embedding quality failed for {len(invalid)} v2 rows")

    svm_centroids = build_centroids(
        rows,
        protocol=SVM_PROTOCOL,
        split=SVM_ENROLLMENT_SPLIT,
        output_dir=model_dir / "svm_closed_set_centroids",
        expected_per_speaker=5,
    )
    train_features, train_labels, training_audio_ids = build_svm_train_features(
        rows,
        output_path=experiment_dir
        / "svm"
        / "svm_closed_set_train_features.npz",
    )
    best_config = train_and_select_svm(
        rows,
        train_features,
        train_labels,
        training_audio_ids,
        results_path=experiment_dir / "svm" / "svm_training_results.csv",
        config_path=model_dir / "svm_best_config.json",
        model_path=model_dir / "speaker_svm_linear.pkl",
    )
    test_centroids = build_centroids(
        rows,
        protocol="COSINE_TEST",
        split="cosine_test_enrollment",
        output_dir=model_dir / "cosine_test_centroids",
        expected_per_speaker=5,
    )
    sid_threshold = _lock_v1_threshold(
        Path("models/experimental/cosine_unknown_threshold.json"),
        model_dir / "cosine_unknown_threshold.json",
        evaluation_dataset="speaker-v2",
    )
    verification_threshold = _lock_v1_threshold(
        Path("models/experimental/verification_threshold.json"),
        model_dir / "verification_threshold.json",
        evaluation_dataset="speaker-v2",
    )
    summary: dict[str, object] = {
        "dataset_version": "speaker-v2",
        "embedding_metadata": embedding_metadata.as_posix(),
        "embedding_metadata_sha256": sha256_file(embedding_metadata),
        "embedding_count": len(rows),
        "embedding_dim": EXPECTED_EMBEDDING_DIM,
        "quality_valid_count": len(quality),
        "svm_centroid_count": len(svm_centroids),
        "svm_train_count": len(train_labels),
        "svm_validation_split": "svm_closed_set_validation",
        "svm_best_config": best_config,
        "cosine_test_centroid_count": len(test_centroids),
        "cosine_query_split": "cosine_test_query",
        "sid_threshold": sid_threshold,
        "verification_threshold": verification_threshold,
        "v1_artifacts_overwritten": False,
    }
    _write_json(experiment_dir / "speaker_v2_training_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-metadata",
        type=Path,
        default=DEFAULT_EMBEDDING_METADATA,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
    )
    args = parser.parse_args()
    summary = build_v2_artifacts(
        args.embedding_metadata,
        args.model_dir,
        args.experiment_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
