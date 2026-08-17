"""Reusable evaluation for ECAPA speaker identification and verification.

The functions in this module operate on already-extracted embeddings.  This
keeps protocol decisions testable and lets notebooks cache each audio embedding
once before applying the three task-specific decision layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import LinearSVC


UNKNOWN_SPEAKER = "UNKNOWN"


def _matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError(f"{name} must be a non-empty 1-D or 2-D array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _binary(labels: Any, *, name: str = "labels") -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if set(np.unique(values)) != {0, 1}:
        raise ValueError(f"{name} must contain both classes 0 and 1")
    return values


def _scores(values: Any, *, expected: int | None = None) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if not scores.size or not np.isfinite(scores).all():
        raise ValueError("scores must be non-empty and finite")
    if expected is not None and scores.size != expected:
        raise ValueError("labels and scores must have equal length")
    return scores


def l2_normalize(values: Any) -> np.ndarray:
    """Return row-wise L2-normalized embeddings; reject zero/non-finite rows."""

    matrix = _matrix(values, name="embeddings")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("embeddings contain a zero-norm row")
    normalized = matrix / norms
    return normalized[0] if np.asarray(values).ndim == 1 else normalized


def build_centroids(embeddings: Any, speaker_ids: Sequence[Any]) -> dict[str, np.ndarray]:
    """Build normalized mean centroids for every enrolled speaker."""

    vectors = l2_normalize(embeddings)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    speakers = np.asarray([str(value) for value in speaker_ids], dtype=object)
    if len(speakers) != len(vectors) or not len(speakers):
        raise ValueError("speaker_ids must match the number of embeddings")
    if any(not value for value in speakers):
        raise ValueError("speaker_ids must be non-empty")
    centroids = {}
    for speaker in sorted(set(speakers)):
        centroids[speaker] = l2_normalize(vectors[speakers == speaker].mean(axis=0))
    return centroids


def score_cosine_trials(
    embeddings_by_path: Mapping[str, Any],
    centroids: Mapping[str, Any],
    trials: Iterable[Mapping[str, Any]],
) -> np.ndarray:
    """Score claimed-centroid verification trials in input order."""

    normalized_centroids = {
        str(speaker): l2_normalize(vector)
        for speaker, vector in centroids.items()
    }
    output = []
    for trial in trials:
        speaker = str(trial.get("enrollment_speaker_id", ""))
        path = str(trial.get("query_audio_path", ""))
        if speaker not in normalized_centroids:
            raise ValueError(f"Missing enrollment centroid: {speaker}")
        if path not in embeddings_by_path:
            raise ValueError(f"Missing query embedding: {path}")
        query = l2_normalize(embeddings_by_path[path])
        if query.shape != normalized_centroids[speaker].shape:
            raise ValueError("Query and centroid dimensions do not match")
        output.append(float(query @ normalized_centroids[speaker]))
    if not output:
        raise ValueError("trials must not be empty")
    return np.asarray(output, dtype=np.float64)


def rates_at_threshold(labels: Any, scores: Any, threshold: float) -> dict[str, float]:
    """Compute FAR/FRR/TAR; equality is accepted (score >= threshold)."""

    binary = _binary(labels)
    values = _scores(scores, expected=len(binary))
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    accepted = values >= float(threshold)
    positives = binary == 1
    negatives = ~positives
    far = float(accepted[negatives].mean())
    frr = float((~accepted[positives]).mean())
    return {"threshold": float(threshold), "far": far, "frr": frr, "tar": 1.0 - frr}


def verification_metrics(labels: Any, scores: Any, p_target: float = 0.01) -> dict[str, float]:
    """Return threshold-independent EER/minDCF and validation thresholds."""

    binary = _binary(labels)
    values = _scores(scores, expected=len(binary))
    if not 0.0 < p_target < 1.0:
        raise ValueError("p_target must be between zero and one")
    fpr, tpr, thresholds = roc_curve(binary, values, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    dcf = p_target * fnr + (1.0 - p_target) * fpr
    min_dcf_index = int(np.argmin(dcf))
    far_mask = fpr <= 0.01
    return {
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": float(thresholds[eer_index]),
        "min_dcf": float(dcf[min_dcf_index] / min(p_target, 1.0 - p_target)),
        "min_dcf_threshold": float(thresholds[min_dcf_index]),
        "tar_at_far_1pct": float(tpr[far_mask].max()) if far_mask.any() else 0.0,
    }


def select_closed_set_svm(
    train_embeddings: Any,
    train_labels: Sequence[Any],
    validation_embeddings: Any,
    validation_labels: Sequence[Any],
    *,
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    class_weights: Sequence[str | None] = (None, "balanced"),
    random_state: int = 42,
) -> dict[str, Any]:
    """Select LinearSVC hyperparameters using validation labels only."""

    train_x = _matrix(train_embeddings, name="train_embeddings")
    valid_x = _matrix(validation_embeddings, name="validation_embeddings")
    train_y = np.asarray([str(value) for value in train_labels], dtype=object)
    valid_y = np.asarray([str(value) for value in validation_labels], dtype=object)
    if len(train_x) != len(train_y) or len(valid_x) != len(valid_y):
        raise ValueError("embedding and label lengths must match")
    if train_x.shape[1] != valid_x.shape[1]:
        raise ValueError("train and validation embedding dimensions must match")
    if len(set(train_y)) < 2 or not set(valid_y).issubset(set(train_y)):
        raise ValueError("closed-set labels require at least two train classes and no unseen validation class")
    candidates = []
    for c_value in c_values:
        if not np.isfinite(c_value) or c_value <= 0:
            raise ValueError("c_values must be positive and finite")
        for class_weight in class_weights:
            model = LinearSVC(
                C=float(c_value), class_weight=class_weight,
                random_state=random_state, dual="auto", max_iter=20_000,
            ).fit(train_x, train_y)
            predictions = model.predict(valid_x)
            macro_f1 = float(f1_score(valid_y, predictions, average="macro", zero_division=0))
            accuracy = float(accuracy_score(valid_y, predictions))
            candidates.append((macro_f1, accuracy, -float(c_value), class_weight is None, model, c_value, class_weight))
    best = max(candidates, key=lambda row: row[:4])
    return {
        "model": best[4],
        "selected_c": float(best[5]),
        "class_weight": best[6],
        "validation": {"macro_f1": best[0], "accuracy": best[1]},
    }


def evaluate_closed_set(model: Any, embeddings: Any, labels: Sequence[Any]) -> dict[str, Any]:
    """Evaluate a previously selected closed-set classifier."""

    features = _matrix(embeddings, name="test_embeddings")
    truth = np.asarray([str(value) for value in labels], dtype=object)
    if len(features) != len(truth):
        raise ValueError("test embedding and label lengths must match")
    predictions = np.asarray(model.predict(features), dtype=object)
    classes = np.asarray([str(value) for value in model.classes_], dtype=object)
    if not set(truth).issubset(set(classes)):
        raise ValueError("closed-set test labels contain an unseen class")
    recalls = recall_score(truth, predictions, labels=classes, average=None, zero_division=0)
    return {
        "predictions": predictions,
        "classes": classes,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=classes),
        "per_speaker_recall": {speaker: float(value) for speaker, value in zip(classes, recalls)},
        "metrics": {
            "accuracy": float(accuracy_score(truth, predictions)),
            "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        },
    }


def evaluate_closed_validation(train_embeddings: Any, train_labels: Sequence[Any], validation_embeddings: Any, validation_labels: Sequence[Any]) -> dict[str, Any]:
    """Fit classifier on closed-train and score closed-validation only."""
    selection = select_closed_set_svm(train_embeddings, train_labels, validation_embeddings, validation_labels)
    return {"metrics": selection["validation"], "model": selection["model"], "selected_c": selection["selected_c"], "class_weight": selection["class_weight"]}


def evaluate_verification_validation(embedding_cache: Mapping[str, Any], enrollment: Any, trials: Any) -> dict[str, Any]:
    centroids = build_centroids(_protocol_vectors(embedding_cache, enrollment), enrollment["speaker_id"])
    scores = score_cosine_trials(embedding_cache, centroids, trials.to_dict("records"))
    metrics = verification_metrics(trials["label"], scores)
    return {"metrics": metrics, "scores": scores}


def evaluate_open_validation(embedding_cache: Mapping[str, Any], gallery: Any, queries: Any) -> dict[str, Any]:
    centroids = build_centroids(_protocol_vectors(embedding_cache, gallery), gallery["speaker_id"])
    candidates, scores = predict_open_set(_protocol_vectors(embedding_cache, queries, "query_audio_path"), centroids, threshold=-1.0)
    threshold = select_open_set_threshold(queries["is_known"], scores)
    return {"metrics": open_set_metrics(queries["query_speaker_id"], queries["is_known"], candidates, scores, threshold=threshold), "scores": scores, "threshold": threshold}


def predict_open_set(
    query_embeddings: Any,
    centroids: Mapping[str, Any],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum-centroid identities, rejecting scores below threshold."""

    queries = l2_normalize(query_embeddings)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if not centroids:
        raise ValueError("gallery centroids must not be empty")
    speakers = np.asarray(sorted(str(value) for value in centroids), dtype=object)
    gallery = np.stack([l2_normalize(centroids[speaker]) for speaker in speakers])
    if queries.shape[1] != gallery.shape[1] or not np.isfinite(threshold):
        raise ValueError("query/gallery dimensions and threshold must be valid")
    similarity = queries @ gallery.T
    winners = similarity.argmax(axis=1)
    scores = similarity[np.arange(len(queries)), winners]
    identities = speakers[winners].copy()
    identities[scores < float(threshold)] = UNKNOWN_SPEAKER
    return identities, scores.astype(np.float64)


def select_open_set_threshold(is_known: Any, max_scores: Any) -> float:
    """Select the equal-error known/unknown rejection threshold on validation."""

    labels = _binary(is_known, name="is_known")
    scores = _scores(max_scores, expected=len(labels))
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    return float(thresholds[index])


def open_set_metrics(
    true_speaker_ids: Sequence[Any],
    is_known: Any,
    predicted_speaker_ids: Sequence[Any],
    max_scores: Any,
    *,
    threshold: float,
    far_target: float = 0.01,
) -> dict[str, float]:
    """Compute known ID, unknown rejection, AUROC, FAR/FRR, and DIR@FAR."""

    truth = np.asarray([str(value) for value in true_speaker_ids], dtype=object)
    predictions = np.asarray([str(value) for value in predicted_speaker_ids], dtype=object)
    known = _binary(is_known, name="is_known")
    scores = _scores(max_scores, expected=len(known))
    if len(truth) != len(known) or len(predictions) != len(known):
        raise ValueError("open-set arrays must have equal length")
    if not 0.0 <= far_target <= 1.0:
        raise ValueError("far_target must be between zero and one")
    known_mask = known == 1
    unknown_mask = ~known_mask
    accepted = scores >= float(threshold)
    candidate_correct = predictions[known_mask] == truth[known_mask]
    far = float(accepted[unknown_mask].mean())
    frr = float((~accepted[known_mask]).mean())
    unknown_scores = np.sort(scores[unknown_mask])[::-1]
    allowed_false_accepts = int(np.floor(far_target * len(unknown_scores)))
    fixed_threshold = (
        float("inf") if allowed_false_accepts == 0
        else float(np.nextafter(unknown_scores[allowed_false_accepts - 1], np.inf))
    )
    # With zero allowed false accepts, the highest unknown score can still be
    # rejected while known scores strictly above it contribute to DIR.
    if allowed_false_accepts == 0:
        fixed_threshold = float(np.nextafter(unknown_scores[0], np.inf))
    dir_value = float(((scores[known_mask] >= fixed_threshold) & candidate_correct).mean())
    return {
        "threshold": float(threshold),
        "known_identification_accuracy": float(
            (accepted[known_mask] & candidate_correct).mean()
        ),
        "unknown_rejection_rate": float((~accepted[unknown_mask]).mean()),
        "known_unknown_auroc": float(roc_auc_score(known, scores)),
        "far": far,
        "frr": frr,
        "dir_at_far_1pct": dir_value,
        "dir_far_target": float(far_target),
        "dir_threshold": fixed_threshold,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Metric value is not JSON serializable: {type(value).__name__}")


def write_metrics_json(path: str | Path, metrics: Mapping[str, Any]) -> None:
    """Serialize machine-readable metrics with stable formatting."""

    Path(path).write_text(
        json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _protocol_vectors(
    cache: Mapping[str, Any],
    protocol: Any,
    path_column: str = "audio_path",
) -> np.ndarray:
    paths = protocol[path_column].astype(str)
    missing = sorted(set(paths) - set(cache))
    if missing:
        raise ValueError(f"Missing cached protocol embeddings: {missing[:3]}")
    return np.stack([cache[str(path)] for path in paths])


def evaluate_three_tasks(
    model_name: str,
    embedding_cache: Mapping[str, Any],
    protocols: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate closed-set ID, verification, and open-set ID for one encoder.

    All model and threshold selection uses validation data. Test data is read
    only after selection. When ``output_dir`` is provided, task metrics and
    prediction artifacts are written under that model-specific directory.
    """

    required = {
        "closed_train",
        "closed_validation",
        "closed_test",
        "verification_validation_enrollment",
        "verification_validation_trials",
        "verification_test_enrollment",
        "verification_test_trials",
        "open_validation_gallery",
        "open_validation_queries",
        "open_test_gallery",
        "open_test_queries",
    }
    missing_protocols = sorted(required - set(protocols))
    if missing_protocols:
        raise ValueError(f"Missing three-task protocols: {missing_protocols}")

    closed_train = protocols["closed_train"]
    closed_validation = protocols["closed_validation"]
    closed_test = protocols["closed_test"]
    verification_validation_enrollment = protocols[
        "verification_validation_enrollment"
    ]
    verification_validation_trials = protocols["verification_validation_trials"]
    verification_test_enrollment = protocols["verification_test_enrollment"]
    verification_test_trials = protocols["verification_test_trials"]
    open_validation_gallery = protocols["open_validation_gallery"]
    open_validation_queries = protocols["open_validation_queries"]
    open_test_gallery = protocols["open_test_gallery"]
    open_test_queries = protocols["open_test_queries"]

    closed_selection = select_closed_set_svm(
        _protocol_vectors(embedding_cache, closed_train),
        closed_train["speaker_id"],
        _protocol_vectors(embedding_cache, closed_validation),
        closed_validation["speaker_id"],
    )
    closed_test_result = evaluate_closed_set(
        closed_selection["model"],
        _protocol_vectors(embedding_cache, closed_test),
        closed_test["speaker_id"],
    )
    closed_metrics = {
        "model": model_name,
        "selected_c": closed_selection["selected_c"],
        "class_weight": closed_selection["class_weight"],
        "validation": closed_selection["validation"],
        "test": closed_test_result["metrics"],
        "test_per_speaker_recall": closed_test_result["per_speaker_recall"],
    }

    validation_centroids = build_centroids(
        _protocol_vectors(embedding_cache, verification_validation_enrollment),
        verification_validation_enrollment["speaker_id"],
    )
    validation_verification_scores = score_cosine_trials(
        embedding_cache,
        validation_centroids,
        verification_validation_trials.to_dict("records"),
    )
    validation_verification = verification_metrics(
        verification_validation_trials["label"],
        validation_verification_scores,
    )
    verification_threshold = validation_verification["min_dcf_threshold"]
    test_centroids = build_centroids(
        _protocol_vectors(embedding_cache, verification_test_enrollment),
        verification_test_enrollment["speaker_id"],
    )
    test_verification_scores = score_cosine_trials(
        embedding_cache,
        test_centroids,
        verification_test_trials.to_dict("records"),
    )
    test_verification = verification_metrics(
        verification_test_trials["label"], test_verification_scores
    )
    test_verification.update(
        rates_at_threshold(
            verification_test_trials["label"],
            test_verification_scores,
            verification_threshold,
        )
    )
    verification_result = {
        "model": model_name,
        "validation": validation_verification,
        "test": test_verification,
    }

    validation_gallery_centroids = build_centroids(
        _protocol_vectors(embedding_cache, open_validation_gallery),
        open_validation_gallery["speaker_id"],
    )
    validation_candidates, validation_scores = predict_open_set(
        _protocol_vectors(
            embedding_cache, open_validation_queries, "query_audio_path"
        ),
        validation_gallery_centroids,
        threshold=-1.0,
    )
    open_threshold = select_open_set_threshold(
        open_validation_queries["is_known"], validation_scores
    )
    validation_open_metrics = open_set_metrics(
        open_validation_queries["query_speaker_id"],
        open_validation_queries["is_known"],
        validation_candidates,
        validation_scores,
        threshold=open_threshold,
    )
    test_gallery_centroids = build_centroids(
        _protocol_vectors(embedding_cache, open_test_gallery),
        open_test_gallery["speaker_id"],
    )
    test_candidates, test_scores = predict_open_set(
        _protocol_vectors(embedding_cache, open_test_queries, "query_audio_path"),
        test_gallery_centroids,
        threshold=-1.0,
    )
    test_open_metrics = open_set_metrics(
        open_test_queries["query_speaker_id"],
        open_test_queries["is_known"],
        test_candidates,
        test_scores,
        threshold=open_threshold,
    )
    open_result = {
        "model": model_name,
        "validation": validation_open_metrics,
        "test": test_open_metrics,
    }

    if output_dir is not None:
        import pandas as pd

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_metrics_json(destination / "verification_metrics.json", verification_result)
        verification_output = verification_test_trials.copy()
        verification_output["score"] = test_verification_scores
        verification_output["accepted"] = (
            test_verification_scores >= verification_threshold
        )
        verification_output.to_csv(
            destination / "verification_trial_scores.csv", index=False
        )

        write_metrics_json(destination / "closed_set_metrics.json", closed_metrics)
        closed_output = closed_test.copy()
        closed_output["predicted_speaker_id"] = closed_test_result["predictions"]
        closed_output.to_csv(destination / "closed_set_predictions.csv", index=False)
        pd.DataFrame(
            closed_test_result["confusion_matrix"],
            index=closed_test_result["classes"],
            columns=closed_test_result["classes"],
        ).to_csv(destination / "closed_set_confusion_matrix.csv")

        write_metrics_json(destination / "open_set_metrics.json", open_result)
        open_output = open_test_queries.copy()
        open_output["predicted_speaker_id"] = np.where(
            test_scores >= open_threshold, test_candidates, UNKNOWN_SPEAKER
        )
        open_output["max_score"] = test_scores
        open_output["threshold"] = open_threshold
        open_output.to_csv(destination / "open_set_predictions.csv", index=False)

    return {
        "model": model_name,
        "closed_set": closed_metrics,
        "verification": verification_result,
        "open_set": open_result,
    }


def summarize_three_task_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build comparable summary rows for frozen and fine-tuned encoders."""

    return [
        {
            "model": result["model"],
            "closed_set_test_accuracy": result["closed_set"]["test"]["accuracy"],
            "closed_set_test_macro_f1": result["closed_set"]["test"]["macro_f1"],
            "verification_test_eer": result["verification"]["test"]["eer"],
            "verification_test_min_dcf": result["verification"]["test"]["min_dcf"],
            "open_set_test_known_id_accuracy": result["open_set"]["test"][
                "known_identification_accuracy"
            ],
            "open_set_test_unknown_rejection": result["open_set"]["test"][
                "unknown_rejection_rate"
            ],
            "open_set_test_auroc": result["open_set"]["test"][
                "known_unknown_auroc"
            ],
        }
        for result in results
    ]
