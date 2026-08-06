"""Score validation trials and select an application verification threshold."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TRIALS_PATH = Path(
    "experiments/validation/"
    "speaker_disjoint_verification_validation_trials.csv"
)
SCORES_PATH = Path(
    "experiments/validation/"
    "speaker_disjoint_verification_validation_scores.csv"
)
RESULTS_PATH = Path("experiments/validation/verification_threshold_results.csv")
CONFIG_PATH = Path("models/experimental/verification_threshold.json")
EMBEDDING_METADATA = Path("data/metadata/embedding_metadata.csv")
CENTROID_DIR = Path("models/experimental/cosine_validation_centroids")

SCORE_FIELDS = (
    "trial_id",
    "protocol",
    "query_audio_id",
    "query_speaker_id",
    "query_split",
    "claimed_speaker_id",
    "trial_type",
    "label",
    "cosine_similarity",
)

RESULT_FIELDS = (
    "threshold",
    "genuine_count",
    "impostor_count",
    "true_accept",
    "false_reject",
    "true_reject",
    "false_accept",
    "FAR",
    "FRR",
    "accuracy",
    "F1",
    "eer_gap",
    "selected",
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


def _load_normalized(path: Path) -> np.ndarray:
    vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"Invalid embedding: {path}")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError(f"Zero embedding: {path}")
    return vector / norm


def score_validation_trials(
    trials_path: Path = TRIALS_PATH,
    metadata_path: Path = EMBEDDING_METADATA,
    centroid_dir: Path = CENTROID_DIR,
    output_path: Path = SCORES_PATH,
) -> list[dict[str, Any]]:
    """Recompute cosine similarity for every validation verification trial."""

    trials = _read_csv(trials_path)
    metadata = {
        row["audio_id"]: row
        for row in _read_csv(metadata_path)
    }
    centroids = {
        path.stem: _load_normalized(path)
        for path in sorted(centroid_dir.glob("*.npy"))
    }
    if not centroids:
        raise FileNotFoundError(f"No validation centroids found in: {centroid_dir}")

    scores: list[dict[str, Any]] = []
    for trial in trials:
        if trial["protocol"] != "COSINE_VALIDATION":
            raise ValueError(f"Non-validation protocol in trial {trial['trial_id']}")
        query_row = metadata.get(trial["query_audio_id"])
        if query_row is None:
            raise ValueError(f"Missing query metadata for {trial['trial_id']}")
        if query_row["protocol"] != "COSINE_VALIDATION":
            raise ValueError(f"Query protocol leakage in {trial['trial_id']}")
        if query_row["split"] not in (
            "cosine_validation_query",
            "cosine_validation_unknown",
        ):
            raise ValueError(f"Query split leakage in {trial['trial_id']}")
        claimed = trial["claimed_speaker_id"]
        if claimed not in centroids:
            raise ValueError(f"Missing claimed centroid for {trial['trial_id']}")

        query = _load_normalized(Path(query_row["embedding_path"]))
        similarity = float(query @ centroids[claimed])
        genuine = trial["trial_type"] == "GENUINE"
        scores.append(
            {
                "trial_id": trial["trial_id"],
                "protocol": trial["protocol"],
                "query_audio_id": trial["query_audio_id"],
                "query_speaker_id": trial["query_speaker_id"],
                "query_split": trial["query_split"],
                "claimed_speaker_id": claimed,
                "trial_type": trial["trial_type"],
                "label": 1 if genuine else 0,
                "cosine_similarity": f"{similarity:.8f}",
            }
        )
    _write_csv(output_path, SCORE_FIELDS, scores)
    return scores


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    values = np.unique(scores)
    if not values.size:
        raise ValueError("At least one verification score is required")
    if values.size == 1:
        return np.asarray(
            [
                np.nextafter(values[0], -np.inf),
                np.nextafter(values[0], np.inf),
            ]
        )
    return np.concatenate(
        (
            [np.nextafter(values[0], -np.inf)],
            (values[:-1] + values[1:]) / 2.0,
            [np.nextafter(values[-1], np.inf)],
        )
    )


def select_verification_threshold(
    scores_path: Path = SCORES_PATH,
    results_path: Path = RESULTS_PATH,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Select the validation threshold closest to balanced FAR and FRR."""

    rows = _read_csv(scores_path)
    scores = np.asarray([float(row["cosine_similarity"]) for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows])
    genuine_count = int(np.sum(labels == 1))
    impostor_count = int(np.sum(labels == 0))
    if not genuine_count or not impostor_count:
        raise ValueError("Both genuine and impostor validation trials are required")

    evaluated: list[dict[str, Any]] = []
    ranking: list[tuple[float, float, float, float, float]] = []
    for threshold in _candidate_thresholds(scores):
        accepted = scores >= threshold
        true_accept = int(np.sum(accepted & (labels == 1)))
        false_reject = genuine_count - true_accept
        false_accept = int(np.sum(accepted & (labels == 0)))
        true_reject = impostor_count - false_accept
        far = false_accept / impostor_count
        frr = false_reject / genuine_count
        accuracy = (true_accept + true_reject) / len(labels)
        denominator = 2 * true_accept + false_accept + false_reject
        f1 = (2 * true_accept / denominator) if denominator else 0.0
        gap = abs(far - frr)
        evaluated.append(
            {
                "threshold": f"{float(threshold):.8f}",
                "genuine_count": genuine_count,
                "impostor_count": impostor_count,
                "true_accept": true_accept,
                "false_reject": false_reject,
                "true_reject": true_reject,
                "false_accept": false_accept,
                "FAR": f"{far:.8f}",
                "FRR": f"{frr:.8f}",
                "accuracy": f"{accuracy:.8f}",
                "F1": f"{f1:.8f}",
                "eer_gap": f"{gap:.8f}",
                "selected": "false",
            }
        )
        ranking.append((gap, (far + frr) / 2.0, -f1, -accuracy, threshold))

    selected_index = min(range(len(evaluated)), key=lambda index: ranking[index])
    evaluated[selected_index]["selected"] = "true"
    _write_csv(results_path, RESULT_FIELDS, evaluated)
    selected = evaluated[selected_index]
    config = {
        "protocol": "APPLICATION_SV",
        "threshold_type": "cosine_verification_threshold",
        "threshold": float(selected["threshold"]),
        "comparison": "cosine_similarity >= threshold means verified",
        "selection_criterion": (
            "minimize absolute FAR-FRR gap; tie-break by lower mean error, "
            "higher F1, higher accuracy, then smaller threshold"
        ),
        "validation_scores_source": str(scores_path),
        "genuine_count": genuine_count,
        "impostor_count": impostor_count,
        "metrics": {
            "FAR": float(selected["FAR"]),
            "FRR": float(selected["FRR"]),
            "accuracy": float(selected["accuracy"]),
            "F1": float(selected["F1"]),
            "eer_gap": float(selected["eer_gap"]),
            "approximate_EER": (
                float(selected["FAR"]) + float(selected["FRR"])
            )
            / 2.0,
        },
    }
    _write_json(config_path, config)
    return config


def main() -> int:
    scores = score_validation_trials()
    config = select_verification_threshold()
    summary = {
        "score_count": len(scores),
        "selected_threshold": config["threshold"],
        "metrics": config["metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
