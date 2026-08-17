"""Evaluate all frozen Speaker v2 test protocols from cached embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _vector(row: dict[str, str]) -> np.ndarray:
    vector = np.asarray(
        np.load(row["embedding_path"], allow_pickle=False),
        dtype=np.float32,
    ).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm <= np.finfo(np.float32).eps:
        raise ValueError(f"Invalid embedding: {row['embedding_path']}")
    return vector / norm


def _rows_for(
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


def _load_centroids(path: Path) -> dict[str, np.ndarray]:
    centroids = {
        source.stem: np.asarray(
            np.load(source, allow_pickle=False),
            dtype=np.float32,
        ).reshape(-1)
        for source in sorted(path.glob("*.npy"))
    }
    if not centroids:
        raise ValueError(f"No centroids found: {path}")
    return {
        speaker: vector / np.linalg.norm(vector)
        for speaker, vector in centroids.items()
    }


def _score_gallery(
    vector: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> tuple[str, float, float]:
    speakers = sorted(centroids)
    gallery = np.stack([centroids[speaker] for speaker in speakers])
    started = time.perf_counter()
    similarities = gallery @ vector
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    winner = int(np.argmax(similarities))
    return speakers[winner], float(similarities[winner]), elapsed_ms


def _per_speaker(
    truth: list[str],
    prediction: list[str],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for speaker in sorted(set(truth)):
        indices = [index for index, value in enumerate(truth) if value == speaker]
        correct = sum(prediction[index] == speaker for index in indices)
        output[speaker] = {
            "count": len(indices),
            "correct": correct,
            "accuracy": correct / len(indices),
        }
    return output


def _save_confusion(
    truth: list[str],
    prediction: list[str],
    labels: list[str],
    path: Path,
) -> list[list[int]]:
    matrix = confusion_matrix(truth, prediction, labels=labels)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Speaker v2 SVM closed-set confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return matrix.tolist()


def _closed_set(
    rows: list[dict[str, str]],
    model_path: Path,
    centroid_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selected = _rows_for(
        rows,
        protocol="SVM_CLOSED_SET",
        split="svm_closed_set_test",
    )
    vectors = np.stack([_vector(row) for row in selected])
    truth = [row["speaker_id"] for row in selected]
    payload = joblib.load(model_path)
    model = payload["model"] if isinstance(payload, dict) else payload
    started = time.perf_counter()
    svm_prediction = model.predict(vectors).tolist()
    svm_total_ms = (time.perf_counter() - started) * 1000.0
    labels = sorted(set(truth) | set(svm_prediction))
    svm_rows = [
        {
            "audio_id": row["audio_id"],
            "true_speaker_id": expected,
            "predicted_speaker_id": predicted,
            "correct": str(expected == predicted).lower(),
        }
        for row, expected, predicted in zip(selected, truth, svm_prediction)
    ]
    _write_csv(
        output_dir / "svm_closed_set_predictions.csv",
        svm_rows[0].keys(),
        svm_rows,
    )
    matrix = _save_confusion(
        truth,
        svm_prediction,
        labels,
        output_dir / "svm_closed_set_confusion_matrix.png",
    )
    svm_metrics = {
        "protocol": "SPEAKER_V2_SVM_CLOSED_SET_TEST",
        "test_sample_count": len(selected),
        "num_classes": len(set(truth)),
        "unknown_threshold_applied": False,
        "metrics": {
            "accuracy": float(accuracy_score(truth, svm_prediction)),
            "macro_precision": float(
                precision_score(truth, svm_prediction, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(truth, svm_prediction, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(truth, svm_prediction, average="macro", zero_division=0)
            ),
        },
        "per_speaker_accuracy": _per_speaker(truth, svm_prediction),
        "confusion_matrix": matrix,
        "mean_inference_latency_ms": svm_total_ms / len(selected),
    }
    _write_json(output_dir / "svm_closed_set_metrics.json", svm_metrics)

    centroids = _load_centroids(centroid_dir)
    cosine_prediction: list[str] = []
    cosine_rows: list[dict[str, Any]] = []
    for row, vector, expected in zip(selected, vectors, truth):
        predicted, similarity, latency_ms = _score_gallery(vector, centroids)
        cosine_prediction.append(predicted)
        cosine_rows.append(
            {
                "audio_id": row["audio_id"],
                "true_speaker_id": expected,
                "predicted_speaker_id": predicted,
                "max_similarity": f"{similarity:.8f}",
                "correct": str(expected == predicted).lower(),
                "latency_ms": f"{latency_ms:.6f}",
            }
        )
    _write_csv(
        output_dir / "cosine_closed_set_predictions.csv",
        cosine_rows[0].keys(),
        cosine_rows,
    )
    cosine_metrics = {
        "protocol": "SPEAKER_V2_COSINE_CLOSED_SET_TEST",
        "test_sample_count": len(selected),
        "num_classes": len(set(truth)),
        "unknown_threshold_applied": False,
        "metrics": {
            "accuracy": float(accuracy_score(truth, cosine_prediction)),
            "macro_precision": float(
                precision_score(truth, cosine_prediction, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(truth, cosine_prediction, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(truth, cosine_prediction, average="macro", zero_division=0)
            ),
        },
        "per_speaker_accuracy": _per_speaker(truth, cosine_prediction),
        "mean_inference_latency_ms": float(
            np.mean([float(row["latency_ms"]) for row in cosine_rows])
        ),
    }
    _write_json(output_dir / "cosine_closed_set_metrics.json", cosine_metrics)
    comparison = [
        {
            "audio_id": row["audio_id"],
            "true_speaker_id": expected,
            "svm_predicted_speaker_id": svm_predicted,
            "cosine_predicted_speaker_id": cosine_predicted,
            "svm_correct": str(expected == svm_predicted).lower(),
            "cosine_correct": str(expected == cosine_predicted).lower(),
        }
        for row, expected, svm_predicted, cosine_predicted in zip(
            selected,
            truth,
            svm_prediction,
            cosine_prediction,
        )
    ]
    _write_csv(
        output_dir / "closed_set_svm_vs_cosine.csv",
        comparison[0].keys(),
        comparison,
    )
    return {"svm": svm_metrics, "cosine": cosine_metrics}


def _open_set_and_verification(
    rows: list[dict[str, str]],
    centroid_dir: Path,
    sid_threshold_path: Path,
    verification_threshold_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    known = _rows_for(
        rows,
        protocol="COSINE_TEST",
        split="cosine_test_query",
    )
    unknown = _rows_for(
        rows,
        protocol="COSINE_TEST",
        split="cosine_test_unknown",
    )
    centroids = _load_centroids(centroid_dir)
    sid_threshold = float(
        json.loads(sid_threshold_path.read_text(encoding="utf-8"))["threshold"]
    )
    verification_threshold = float(
        json.loads(verification_threshold_path.read_text(encoding="utf-8"))[
            "threshold"
        ]
    )
    sid_rows: list[dict[str, Any]] = []
    known_candidate_correct = 0
    known_correct = 0
    known_accepted = 0
    for row in known:
        candidate, similarity, latency_ms = _score_gallery(_vector(row), centroids)
        accepted = similarity >= sid_threshold
        prediction = candidate if accepted else "UNKNOWN"
        known_candidate_correct += candidate == row["speaker_id"]
        known_correct += prediction == row["speaker_id"]
        known_accepted += accepted
        sid_rows.append(
            {
                "audio_id": row["audio_id"],
                "query_type": "KNOWN",
                "true_speaker_id": row["speaker_id"],
                "candidate_speaker_id": candidate,
                "max_similarity": f"{similarity:.8f}",
                "threshold": f"{sid_threshold:.8f}",
                "predicted_speaker_id": prediction,
                "correct": str(prediction == row["speaker_id"]).lower(),
                "latency_ms": f"{latency_ms:.6f}",
            }
        )
    unknown_rejected = 0
    for row in unknown:
        candidate, similarity, latency_ms = _score_gallery(_vector(row), centroids)
        accepted = similarity >= sid_threshold
        prediction = candidate if accepted else "UNKNOWN"
        unknown_rejected += not accepted
        sid_rows.append(
            {
                "audio_id": row["audio_id"],
                "query_type": "UNKNOWN",
                "true_speaker_id": "UNKNOWN",
                "candidate_speaker_id": candidate,
                "max_similarity": f"{similarity:.8f}",
                "threshold": f"{sid_threshold:.8f}",
                "predicted_speaker_id": prediction,
                "correct": str(not accepted).lower(),
                "latency_ms": f"{latency_ms:.6f}",
            }
        )
    _write_csv(
        output_dir / "speaker_disjoint_sid_test_predictions.csv",
        sid_rows[0].keys(),
        sid_rows,
    )
    sid_metrics = {
        "protocol": "SPEAKER_V2_DISJOINT_OPEN_SET_SID_TEST",
        "threshold": sid_threshold,
        "threshold_source": sid_threshold_path.as_posix(),
        "threshold_selection_dataset_version": "v1",
        "threshold_tuning_on_v2_test": False,
        "known_query": {
            "count": len(known),
            "candidate_identification_accuracy": known_candidate_correct / len(known),
            "identification_accuracy": known_correct / len(known),
            "acceptance_rate": known_accepted / len(known),
            "false_unknown_rate": 1.0 - known_accepted / len(known),
        },
        "unknown_query": {
            "count": len(unknown),
            "unknown_rejection_rate": unknown_rejected / len(unknown),
            "false_known_rate": 1.0 - unknown_rejected / len(unknown),
        },
        "open_set": {
            "total_queries": len(known) + len(unknown),
            "correct": known_correct + unknown_rejected,
            "overall_accuracy": (
                (known_correct + unknown_rejected) / (len(known) + len(unknown))
            ),
        },
        "mean_cosine_sid_latency_ms": float(
            np.mean([float(row["latency_ms"]) for row in sid_rows])
        ),
    }
    _write_json(
        output_dir / "speaker_disjoint_sid_test_metrics.json",
        sid_metrics,
    )

    trials: list[dict[str, Any]] = []
    trial_index = 0
    for query_type, query_rows in (("KNOWN", known), ("UNKNOWN", unknown)):
        for row in query_rows:
            vector = _vector(row)
            for centroid_speaker in sorted(centroids):
                trial_index += 1
                genuine = (
                    query_type == "KNOWN"
                    and row["speaker_id"] == centroid_speaker
                )
                trial_type = (
                    "GENUINE"
                    if genuine
                    else "KNOWN_IMPOSTOR"
                    if query_type == "KNOWN"
                    else "UNKNOWN_IMPOSTOR"
                )
                score = float(centroids[centroid_speaker] @ vector)
                trials.append(
                    {
                        "trial_id": f"v2_trial_{trial_index:06d}",
                        "query_audio_id": row["audio_id"],
                        "query_speaker_id": row["speaker_id"],
                        "centroid_speaker_id": centroid_speaker,
                        "trial_type": trial_type,
                        "label": int(genuine),
                        "score": f"{score:.8f}",
                    }
                )
    _write_csv(
        output_dir / "speaker_disjoint_verification_test_trials.csv",
        trials[0].keys(),
        trials,
    )
    truth = np.asarray([int(row["label"]) for row in trials], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in trials], dtype=np.float64)
    accepted = scores >= verification_threshold
    predictions = [
        {
            **row,
            "threshold": f"{verification_threshold:.8f}",
            "verified": str(bool(decision)).lower(),
            "correct": str(bool(decision) == bool(label)).lower(),
        }
        for row, decision, label in zip(trials, accepted, truth)
    ]
    _write_csv(
        output_dir / "speaker_disjoint_verification_test_predictions.csv",
        predictions[0].keys(),
        predictions,
    )
    genuine_count = int(np.sum(truth == 1))
    non_genuine_count = int(np.sum(truth == 0))
    true_accept = int(np.sum(accepted & (truth == 1)))
    false_reject = int(np.sum(~accepted & (truth == 1)))
    false_accept = int(np.sum(accepted & (truth == 0)))
    true_reject = int(np.sum(~accepted & (truth == 0)))
    fixed_metrics = {
        "protocol": "SPEAKER_V2_VERIFICATION_FIXED_THRESHOLD",
        "threshold": verification_threshold,
        "threshold_source": verification_threshold_path.as_posix(),
        "threshold_selection_dataset_version": "v1",
        "threshold_tuned_on_v2_test": False,
        "test_trial_count": len(trials),
        "genuine_count": genuine_count,
        "non_genuine_count": non_genuine_count,
        "true_accept": true_accept,
        "false_reject": false_reject,
        "false_accept": false_accept,
        "true_reject": true_reject,
        "metrics": {
            "FAR": false_accept / non_genuine_count,
            "FRR": false_reject / genuine_count,
            "verification_accuracy": (true_accept + true_reject) / len(trials),
            "F1": float(f1_score(truth, accepted)),
        },
    }
    _write_json(
        output_dir
        / "speaker_disjoint_verification_fixed_threshold_metrics.json",
        fixed_metrics,
    )
    fpr, tpr, thresholds = roc_curve(truth, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    curve_metrics = {
        "protocol": "SPEAKER_V2_VERIFICATION_CURVE",
        "test_trial_count": len(trials),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": float(thresholds[eer_index]),
        "eer_far": float(fpr[eer_index]),
        "eer_frr": float(fnr[eer_index]),
        "eer_threshold_used_for_system": False,
        "system_threshold": verification_threshold,
    }
    _write_json(
        output_dir / "speaker_disjoint_verification_curve_metrics.json",
        curve_metrics,
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(fpr, tpr, label=f"AUC={curve_metrics['roc_auc']:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Speaker v2 verification ROC")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "sv_roc_curve.png", dpi=160)
    plt.close(figure)
    return {
        "sid": sid_metrics,
        "verification_fixed": fixed_metrics,
        "verification_curve": curve_metrics,
    }


def evaluate(
    embedding_metadata: Path,
    model_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = _read_csv(embedding_metadata)
    closed = _closed_set(
        rows,
        model_dir / "speaker_svm_linear.pkl",
        model_dir / "svm_closed_set_centroids",
        output_dir,
    )
    opened = _open_set_and_verification(
        rows,
        model_dir / "cosine_test_centroids",
        model_dir / "cosine_unknown_threshold.json",
        model_dir / "verification_threshold.json",
        output_dir,
    )
    summary = {
        "dataset_version": "speaker-v2",
        "closed_set": closed,
        **opened,
    }
    _write_json(output_dir / "speaker_v2_test_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-metadata",
        type=Path,
        default=Path("data/processed/v2/embedding_metadata.csv"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/experimental/v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/v2/test"),
    )
    args = parser.parse_args()
    summary = evaluate(args.embedding_metadata, args.model_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
