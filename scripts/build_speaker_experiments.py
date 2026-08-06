"""Build leakage-safe SVM and cosine-validation speaker artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC


EMBEDDING_METADATA = Path("data/metadata/embedding_metadata.csv")
QUALITY_REPORT = Path("data/metadata/embedding_quality_report.csv")
SVM_CENTROID_DIR = Path("models/experimental/svm_closed_set_centroids")
SVM_FEATURES = Path("experiments/svm/svm_closed_set_train_features.npz")
SVM_RESULTS = Path("experiments/svm/svm_training_results.csv")
SVM_CONFIG = Path("models/experimental/svm_best_config.json")
SVM_MODEL = Path("models/experimental/speaker_svm_linear.pkl")
COSINE_CENTROID_DIR = Path("models/experimental/cosine_validation_centroids")
COSINE_KNOWN_SCORES = Path(
    "experiments/validation/cosine_validation_known_scores.csv"
)
COSINE_UNKNOWN_SCORES = Path(
    "experiments/validation/cosine_validation_unknown_scores.csv"
)

EXPECTED_EMBEDDING_DIM = 192
SVM_PROTOCOL = "SVM_CLOSED_SET"
COSINE_VALIDATION_PROTOCOL = "COSINE_VALIDATION"
SVM_ENROLLMENT_SPLIT = "svm_closed_set_enrollment"
SVM_TRAIN_SPLIT = "svm_closed_set_train"
SVM_VALIDATION_SPLIT = "svm_closed_set_validation"
COSINE_ENROLLMENT_SPLIT = "cosine_validation_enrollment"
COSINE_QUERY_SPLIT = "cosine_validation_query"
COSINE_UNKNOWN_SPLIT = "cosine_validation_unknown"
CANDIDATE_CS = (0.1, 1.0, 10.0)

QUALITY_FIELDS = (
    "audio_id",
    "speaker_id",
    "protocol",
    "split",
    "role",
    "embedding_path",
    "file_exists",
    "embedding_dim",
    "dimension_ok",
    "has_nan",
    "is_zero_vector",
    "l2_norm",
    "l2_norm_ok",
    "latency_ms",
    "latency_ok",
    "valid",
    "issues",
)

SVM_RESULT_FIELDS = (
    "C",
    "train_sample_count",
    "validation_sample_count",
    "validation_accuracy",
    "validation_macro_f1",
    "per_speaker_accuracy",
)

COSINE_SCORE_FIELDS = (
    "audio_id",
    "true_speaker_id",
    "candidate_speaker_id",
    "max_similarity",
    "correct",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npy(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(vector, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def _load_vector(row: dict[str, str]) -> np.ndarray:
    path = Path(row["embedding_path"])
    vector = np.load(path, allow_pickle=False)
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"Embedding must be 1-D: {path}")
    return vector


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(np.float32).eps:
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return np.asarray(vector / norm, dtype=np.float32)


def build_embedding_quality_report(
    rows: list[dict[str, str]],
    output_path: Path = QUALITY_REPORT,
) -> list[dict[str, Any]]:
    """Validate every embedding file, dimension, values, norm, and latency."""

    report: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["embedding_path"])
        issues: list[str] = []
        exists = path.is_file()
        dimension = 0
        has_nan = False
        is_zero = False
        norm = float("nan")
        if not exists:
            issues.append("missing_file")
        else:
            try:
                vector = _load_vector(row)
                dimension = int(vector.size)
                has_nan = not bool(np.isfinite(vector).all())
                is_zero = bool(np.allclose(vector, 0.0))
                norm = float(np.linalg.norm(vector))
            except (OSError, ValueError) as error:
                issues.append(f"unreadable:{error}")

        dimension_ok = dimension == EXPECTED_EMBEDDING_DIM
        norm_ok = np.isfinite(norm) and abs(norm - 1.0) <= 1e-5
        try:
            latency = float(row["latency_ms"])
            latency_ok = np.isfinite(latency) and latency >= 0.0
        except (KeyError, ValueError):
            latency = float("nan")
            latency_ok = False
        if exists and not dimension_ok:
            issues.append("invalid_dimension")
        if has_nan:
            issues.append("nan_or_inf")
        if is_zero:
            issues.append("zero_vector")
        if exists and not norm_ok:
            issues.append("invalid_l2_norm")
        if not latency_ok:
            issues.append("invalid_latency")

        report.append(
            {
                "audio_id": row["audio_id"],
                "speaker_id": row["speaker_id"],
                "protocol": row["protocol"],
                "split": row["split"],
                "role": row["role"],
                "embedding_path": row["embedding_path"],
                "file_exists": str(exists).lower(),
                "embedding_dim": dimension,
                "dimension_ok": str(dimension_ok).lower(),
                "has_nan": str(has_nan).lower(),
                "is_zero_vector": str(is_zero).lower(),
                "l2_norm": f"{norm:.8f}" if np.isfinite(norm) else "",
                "l2_norm_ok": str(bool(norm_ok)).lower(),
                "latency_ms": f"{latency:.3f}" if np.isfinite(latency) else "",
                "latency_ok": str(bool(latency_ok)).lower(),
                "valid": str(not issues).lower(),
                "issues": ";".join(issues),
            }
        )
    _write_csv(output_path, QUALITY_FIELDS, report)
    return report


def _select_rows(
    rows: list[dict[str, str]],
    *,
    protocol: str,
    split: str,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["protocol"] == protocol and row["split"] == split
    ]
    if not selected:
        raise ValueError(f"No rows for protocol={protocol}, split={split}")
    return selected


def build_centroids(
    rows: list[dict[str, str]],
    *,
    protocol: str,
    split: str,
    output_dir: Path,
    expected_per_speaker: int,
) -> dict[str, np.ndarray]:
    """Mean and L2-normalize enrollment embeddings for one isolated gallery."""

    selected = _select_rows(rows, protocol=protocol, split=split)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in selected:
        grouped[row["speaker_id"]].append(_load_vector(row))

    centroids: dict[str, np.ndarray] = {}
    for speaker_id in sorted(grouped):
        vectors = grouped[speaker_id]
        if len(vectors) != expected_per_speaker:
            raise ValueError(
                f"{speaker_id} has {len(vectors)} enrollment embeddings; "
                f"expected {expected_per_speaker}"
            )
        centroid = _normalize(np.mean(np.stack(vectors), axis=0))
        _write_npy(output_dir / f"{speaker_id}.npy", centroid)
        centroids[speaker_id] = centroid
    return centroids


def build_svm_train_features(
    rows: list[dict[str, str]],
    output_path: Path = SVM_FEATURES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build features from the SVM train split only."""

    selected = _select_rows(
        rows,
        protocol=SVM_PROTOCOL,
        split=SVM_TRAIN_SPLIT,
    )
    features = np.stack([_load_vector(row) for row in selected])
    labels = np.asarray([row["speaker_id"] for row in selected])
    audio_ids = [row["audio_id"] for row in selected]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            X=features,
            y=labels,
            audio_ids=np.asarray(audio_ids),
            embedding_paths=np.asarray([row["embedding_path"] for row in selected]),
            protocol=np.asarray([SVM_PROTOCOL]),
            split=np.asarray([SVM_TRAIN_SPLIT]),
        )
    temporary.replace(output_path)
    return features, labels, audio_ids


def _per_speaker_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    return {
        speaker: float(np.mean(prediction[truth == speaker] == speaker))
        for speaker in sorted(set(truth.tolist()))
    }


def train_and_select_svm(
    rows: list[dict[str, str]],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    training_audio_ids: list[str],
    *,
    results_path: Path = SVM_RESULTS,
    config_path: Path = SVM_CONFIG,
    model_path: Path = SVM_MODEL,
    candidate_cs: tuple[float, ...] = CANDIDATE_CS,
) -> dict[str, Any]:
    """Select C using known SVM validation, then refit on SVM train only."""

    validation = _select_rows(
        rows,
        protocol=SVM_PROTOCOL,
        split=SVM_VALIDATION_SPLIT,
    )
    validation_features = np.stack([_load_vector(row) for row in validation])
    validation_labels = np.asarray([row["speaker_id"] for row in validation])

    result_rows: list[dict[str, Any]] = []
    scored: list[tuple[float, float, float]] = []
    for c_value in candidate_cs:
        model = LinearSVC(C=c_value, dual="auto", max_iter=10000, random_state=0)
        model.fit(train_features, train_labels)
        prediction = model.predict(validation_features)
        accuracy = float(accuracy_score(validation_labels, prediction))
        macro_f1 = float(f1_score(validation_labels, prediction, average="macro"))
        result_rows.append(
            {
                "C": f"{c_value:g}",
                "train_sample_count": len(train_labels),
                "validation_sample_count": len(validation_labels),
                "validation_accuracy": f"{accuracy:.8f}",
                "validation_macro_f1": f"{macro_f1:.8f}",
                "per_speaker_accuracy": json.dumps(
                    _per_speaker_accuracy(validation_labels, prediction),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        scored.append((accuracy, macro_f1, c_value))
    _write_csv(results_path, SVM_RESULT_FIELDS, result_rows)

    best_accuracy, best_macro_f1, best_c = max(
        scored,
        key=lambda value: (value[0], value[1], -value[2]),
    )
    config = {
        "protocol": SVM_PROTOCOL,
        "model_type": "LinearSVC",
        "candidate_C": list(candidate_cs),
        "selected_C": best_c,
        "selection_metrics": {
            "validation_accuracy": best_accuracy,
            "validation_macro_f1": best_macro_f1,
        },
        "selection_split": SVM_VALIDATION_SPLIT,
        "selection_sample_count": len(validation_labels),
        "training_split": SVM_TRAIN_SPLIT,
        "training_sample_count": len(train_labels),
        "embedding_dim": int(train_features.shape[1]),
        "tie_breaker": "accuracy, macro_f1, then smaller C",
        "forbidden_selection_data": [
            "svm_closed_set_enrollment",
            "svm_closed_set_test",
            "cosine_validation_*",
            "cosine_test_*",
        ],
    }
    _write_json(config_path, config)

    final_model = LinearSVC(C=best_c, dual="auto", max_iter=10000, random_state=0)
    final_model.fit(train_features, train_labels)
    payload = {
        "protocol": SVM_PROTOCOL,
        "model_type": "LinearSVC",
        "model": final_model,
        "classes": final_model.classes_.tolist(),
        "embedding_dim": int(train_features.shape[1]),
        "selected_C": best_c,
        "training_split": SVM_TRAIN_SPLIT,
        "training_audio_ids": training_audio_ids,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(payload, temporary_model)
    temporary_model.replace(model_path)
    return config


def score_cosine_queries(
    rows: list[dict[str, str]],
    centroids: dict[str, np.ndarray],
    *,
    split: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Score one cosine-validation query split against validation gallery."""

    selected = _select_rows(
        rows,
        protocol=COSINE_VALIDATION_PROTOCOL,
        split=split,
    )
    speaker_ids = sorted(centroids)
    gallery = np.stack([centroids[speaker] for speaker in speaker_ids])
    output: list[dict[str, Any]] = []
    for row in selected:
        query = _normalize(_load_vector(row))
        similarities = gallery @ query
        winner = int(np.argmax(similarities))
        candidate = speaker_ids[winner]
        output.append(
            {
                "audio_id": row["audio_id"],
                "true_speaker_id": row["speaker_id"],
                "candidate_speaker_id": candidate,
                "max_similarity": f"{float(similarities[winner]):.8f}",
                "correct": (
                    str(candidate == row["speaker_id"]).lower()
                    if split == COSINE_QUERY_SPLIT
                    else ""
                ),
            }
        )
    _write_csv(output_path, COSINE_SCORE_FIELDS, output)
    return output


def build_all_artifacts(
    embedding_metadata: Path = EMBEDDING_METADATA,
) -> dict[str, Any]:
    rows = _read_csv(embedding_metadata)
    quality = build_embedding_quality_report(rows)
    invalid = [row for row in quality if row["valid"] != "true"]
    if invalid:
        raise ValueError(f"Embedding quality failed for {len(invalid)} rows")

    svm_centroids = build_centroids(
        rows,
        protocol=SVM_PROTOCOL,
        split=SVM_ENROLLMENT_SPLIT,
        output_dir=SVM_CENTROID_DIR,
        expected_per_speaker=5,
    )
    train_features, train_labels, training_audio_ids = build_svm_train_features(rows)
    best_config = train_and_select_svm(
        rows,
        train_features,
        train_labels,
        training_audio_ids,
    )

    cosine_centroids = build_centroids(
        rows,
        protocol=COSINE_VALIDATION_PROTOCOL,
        split=COSINE_ENROLLMENT_SPLIT,
        output_dir=COSINE_CENTROID_DIR,
        expected_per_speaker=5,
    )
    known_scores = score_cosine_queries(
        rows,
        cosine_centroids,
        split=COSINE_QUERY_SPLIT,
        output_path=COSINE_KNOWN_SCORES,
    )
    unknown_scores = score_cosine_queries(
        rows,
        cosine_centroids,
        split=COSINE_UNKNOWN_SPLIT,
        output_path=COSINE_UNKNOWN_SCORES,
    )
    return {
        "embedding_count": len(rows),
        "quality_valid_count": len(quality),
        "svm_centroid_count": len(svm_centroids),
        "svm_train_count": len(train_labels),
        "svm_best_config": best_config,
        "cosine_centroid_count": len(cosine_centroids),
        "cosine_known_query_count": len(known_scores),
        "cosine_unknown_query_count": len(unknown_scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-metadata",
        type=Path,
        default=EMBEDDING_METADATA,
    )
    args = parser.parse_args()
    summary = build_all_artifacts(args.embedding_metadata)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
