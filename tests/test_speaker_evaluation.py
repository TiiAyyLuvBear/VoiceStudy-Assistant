"""Synthetic tests for reusable ECAPA three-task evaluation helpers."""

from __future__ import annotations

import json
import inspect

import numpy as np
import pandas as pd
import pytest

from src.speaker.evaluation import (
    build_centroids,
    evaluate_closed_set,
    evaluate_three_tasks,
    open_set_metrics,
    predict_open_set,
    rates_at_threshold,
    score_cosine_trials,
    select_closed_set_svm,
    select_open_set_threshold,
    summarize_three_task_results,
    verification_metrics,
    write_metrics_json,
)


def test_centroids_and_cosine_trials_have_predictable_scores() -> None:
    embeddings = {
        "a1": np.array([1.0, 0.0]),
        "a2": np.array([1.0, 0.0]),
        "b1": np.array([0.0, 1.0]),
        "query": np.array([1.0, 0.0]),
    }
    centroids = build_centroids(
        np.stack([embeddings["a1"], embeddings["a2"], embeddings["b1"]]),
        ["a", "a", "b"],
    )
    scores = score_cosine_trials(
        embeddings,
        centroids,
        [
            {"enrollment_speaker_id": "a", "query_audio_path": "query"},
            {"enrollment_speaker_id": "b", "query_audio_path": "query"},
        ],
    )
    np.testing.assert_allclose(scores, [1.0, 0.0])


def test_verification_metrics_and_threshold_equality() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    metrics = verification_metrics(labels, scores)
    assert metrics["eer"] == pytest.approx(0.0)
    rates = rates_at_threshold(labels, scores, threshold=0.8)
    assert rates == {"threshold": 0.8, "far": 0.0, "frr": 0.0, "tar": 1.0}


def test_closed_set_svm_selects_on_validation_and_evaluates_test_once() -> None:
    assert "test" not in inspect.signature(select_closed_set_svm).parameters
    train_x = np.array([
        [2.0, 0.0], [1.0, 0.1], [0.0, 2.0], [0.1, 1.0],
    ])
    train_y = np.array(["a", "a", "b", "b"])
    valid_x = np.array([[1.0, 0.0], [0.0, 1.0]])
    valid_y = np.array(["a", "b"])
    selection = select_closed_set_svm(
        train_x, train_y, valid_x, valid_y, c_values=(0.01, 1.0)
    )
    assert set(selection) == {"model", "selected_c", "class_weight", "validation"}
    result = evaluate_closed_set(selection["model"], valid_x, valid_y)
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["macro_f1"] == 1.0
    assert result["confusion_matrix"].shape == (2, 2)


def test_open_set_rejection_metrics_and_boundary() -> None:
    centroids = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    queries = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    identities, scores = predict_open_set(queries, centroids, threshold=1.0)
    assert identities.tolist() == ["a", "b", "UNKNOWN"]
    threshold = select_open_set_threshold([1, 1, 0], scores)
    metrics = open_set_metrics(
        ["a", "b", "intruder"], [1, 1, 0], identities, scores, threshold=1.0
    )
    assert threshold == pytest.approx(1.0)
    assert metrics["known_identification_accuracy"] == 1.0
    assert metrics["unknown_rejection_rate"] == 1.0
    assert metrics["dir_at_far_1pct"] == 1.0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: build_centroids([[0.0, 0.0]], ["a"]), "zero"),
        (lambda: verification_metrics([1, 1], [0.2, 0.3]), "both classes"),
        (lambda: rates_at_threshold([1, 0], [0.2, np.nan], 0.2), "finite"),
        (
            lambda: score_cosine_trials(
                {"q": np.array([1.0, 0.0])},
                {"a": np.array([1.0, 0.0])},
                [{"enrollment_speaker_id": "missing", "query_audio_path": "q"}],
            ),
            "centroid",
        ),
    ],
)
def test_invalid_inputs_are_rejected(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_metrics_json_serializes_numpy_values(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    write_metrics_json(path, {"score": np.float32(0.5), "count": np.int64(2)})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "score": 0.5,
        "count": 2,
    }


def test_three_task_evaluator_writes_one_model_artifact_set(tmp_path) -> None:
    vectors = {
        "a1": [1.0, 0.0], "a2": [0.9, 0.1],
        "b1": [0.0, 1.0], "b2": [0.1, 0.9],
        "av": [1.0, 0.0], "bv": [0.0, 1.0],
        "at": [1.0, 0.0], "bt": [0.0, 1.0],
        "vea": [1.0, 0.0], "veb": [0.0, 1.0],
        "vqa": [1.0, 0.0], "vqb": [0.0, 1.0],
        "tea": [1.0, 0.0], "teb": [0.0, 1.0],
        "tqa": [1.0, 0.0], "tqb": [0.0, 1.0],
        "vunknown": [-1.0, 0.0], "tunknown": [-1.0, 0.0],
    }
    cache = {key: np.asarray(value) for key, value in vectors.items()}

    def frame(paths, speakers):
        return pd.DataFrame({"audio_path": paths, "speaker_id": speakers})

    def trials(prefix):
        return pd.DataFrame([
            {"enrollment_speaker_id": "a", "query_audio_path": f"{prefix}qa", "label": 1},
            {"enrollment_speaker_id": "b", "query_audio_path": f"{prefix}qa", "label": 0},
            {"enrollment_speaker_id": "b", "query_audio_path": f"{prefix}qb", "label": 1},
            {"enrollment_speaker_id": "a", "query_audio_path": f"{prefix}qb", "label": 0},
        ])

    def open_queries(prefix):
        return pd.DataFrame({
            "query_audio_path": [f"{prefix}qa", f"{prefix}qb", f"{prefix}unknown"],
            "query_speaker_id": ["a", "b", "intruder"],
            "is_known": [1, 1, 0],
        })

    protocols = {
        "closed_train": frame(["a1", "a2", "b1", "b2"], ["a", "a", "b", "b"]),
        "closed_validation": frame(["av", "bv"], ["a", "b"]),
        "closed_test": frame(["at", "bt"], ["a", "b"]),
        "verification_validation_enrollment": frame(["vea", "veb"], ["a", "b"]),
        "verification_validation_trials": trials("v"),
        "verification_test_enrollment": frame(["tea", "teb"], ["a", "b"]),
        "verification_test_trials": trials("t"),
        "open_validation_gallery": frame(["vea", "veb"], ["a", "b"]),
        "open_validation_queries": open_queries("v"),
        "open_test_gallery": frame(["tea", "teb"], ["a", "b"]),
        "open_test_queries": open_queries("t"),
    }

    result = evaluate_three_tasks(
        "synthetic ECAPA", cache, protocols, output_dir=tmp_path
    )
    summary = summarize_three_task_results([result])

    assert summary[0]["closed_set_test_accuracy"] == 1.0
    assert summary[0]["verification_test_eer"] == 0.0
    assert summary[0]["open_set_test_unknown_rejection"] == 1.0
    for filename in (
        "closed_set_metrics.json",
        "verification_metrics.json",
        "open_set_metrics.json",
    ):
        assert (tmp_path / filename).is_file()
