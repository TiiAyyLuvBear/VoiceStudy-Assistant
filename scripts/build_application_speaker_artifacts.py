"""Select application SID threshold and build speaker-disjoint SV trials."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


KNOWN_SCORES = Path("experiments/validation/cosine_validation_known_scores.csv")
UNKNOWN_SCORES = Path("experiments/validation/cosine_validation_unknown_scores.csv")
THRESHOLD_RESULTS = Path(
    "experiments/validation/cosine_unknown_threshold_results.csv"
)
THRESHOLD_CONFIG = Path("models/experimental/cosine_unknown_threshold.json")
EMBEDDING_METADATA = Path("data/metadata/embedding_metadata.csv")
COSINE_CENTROID_DIR = Path("models/experimental/cosine_validation_centroids")
SV_TRIALS = Path(
    "experiments/validation/"
    "speaker_disjoint_verification_validation_trials.csv"
)

COSINE_PROTOCOL = "COSINE_VALIDATION"
KNOWN_SPLIT = "cosine_validation_query"
UNKNOWN_SPLIT = "cosine_validation_unknown"

THRESHOLD_FIELDS = (
    "threshold",
    "known_count",
    "unknown_count",
    "known_accepted",
    "known_rejected",
    "unknown_rejected",
    "unknown_accepted",
    "known_acceptance_rate",
    "false_unknown_rate",
    "unknown_rejection_rate",
    "false_known_rate",
    "overall_detection_accuracy",
    "selected",
)

TRIAL_FIELDS = (
    "trial_id",
    "protocol",
    "query_audio_id",
    "query_speaker_id",
    "query_split",
    "claimed_speaker_id",
    "trial_type",
    "cosine_similarity",
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


def _candidate_thresholds(known: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    values = np.unique(np.concatenate((known, unknown)))
    if not values.size:
        raise ValueError("At least one validation similarity is required")
    if values.size == 1:
        return np.asarray(
            [
                np.nextafter(values[0], -np.inf),
                np.nextafter(values[0], np.inf),
            ]
        )
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        (
            [np.nextafter(values[0], -np.inf)],
            midpoints,
            [np.nextafter(values[-1], np.inf)],
        )
    )


def select_unknown_threshold(
    known_scores_path: Path = KNOWN_SCORES,
    unknown_scores_path: Path = UNKNOWN_SCORES,
    results_path: Path = THRESHOLD_RESULTS,
    config_path: Path = THRESHOLD_CONFIG,
) -> dict[str, Any]:
    """Choose threshold from max similarities on validation only."""

    known_rows = _read_csv(known_scores_path)
    unknown_rows = _read_csv(unknown_scores_path)
    known = np.asarray([float(row["max_similarity"]) for row in known_rows])
    unknown = np.asarray([float(row["max_similarity"]) for row in unknown_rows])
    if not known.size or not unknown.size:
        raise ValueError("Known and unknown validation scores must both be non-empty")
    if not np.isfinite(known).all() or not np.isfinite(unknown).all():
        raise ValueError("Validation similarities contain NaN or infinity")

    evaluated: list[dict[str, Any]] = []
    rank_values: list[tuple[float, float, float, float]] = []
    for threshold in _candidate_thresholds(known, unknown):
        known_accepted = int(np.sum(known >= threshold))
        known_rejected = int(known.size - known_accepted)
        unknown_rejected = int(np.sum(unknown < threshold))
        unknown_accepted = int(unknown.size - unknown_rejected)
        known_acceptance = known_accepted / known.size
        false_unknown = known_rejected / known.size
        unknown_rejection = unknown_rejected / unknown.size
        false_known = unknown_accepted / unknown.size
        overall = (known_accepted + unknown_rejected) / (
            known.size + unknown.size
        )
        evaluated.append(
            {
                "threshold": f"{float(threshold):.8f}",
                "known_count": known.size,
                "unknown_count": unknown.size,
                "known_accepted": known_accepted,
                "known_rejected": known_rejected,
                "unknown_rejected": unknown_rejected,
                "unknown_accepted": unknown_accepted,
                "known_acceptance_rate": f"{known_acceptance:.8f}",
                "false_unknown_rate": f"{false_unknown:.8f}",
                "unknown_rejection_rate": f"{unknown_rejection:.8f}",
                "false_known_rate": f"{false_known:.8f}",
                "overall_detection_accuracy": f"{overall:.8f}",
                "selected": "false",
            }
        )
        rank_values.append(
            (overall, unknown_rejection, known_acceptance, float(threshold))
        )

    best_index = max(
        range(len(evaluated)),
        key=lambda index: (
            rank_values[index][0],
            rank_values[index][1],
            rank_values[index][2],
            -rank_values[index][3],
        ),
    )
    evaluated[best_index]["selected"] = "true"
    _write_csv(results_path, THRESHOLD_FIELDS, evaluated)

    selected = evaluated[best_index]
    config = {
        "protocol": "APPLICATION_SID",
        "threshold_type": "cosine_unknown_threshold",
        "threshold": float(selected["threshold"]),
        "comparison": "cosine_similarity >= threshold means KNOWN",
        "selection_criterion": (
            "maximize overall detection accuracy; tie-break by unknown "
            "rejection, known acceptance, then smaller threshold"
        ),
        "known_validation_source": str(known_scores_path),
        "unknown_validation_source": str(unknown_scores_path),
        "known_count": int(selected["known_count"]),
        "unknown_count": int(selected["unknown_count"]),
        "metrics": {
            "known_acceptance_rate": float(selected["known_acceptance_rate"]),
            "false_unknown_rate": float(selected["false_unknown_rate"]),
            "unknown_rejection_rate": float(
                selected["unknown_rejection_rate"]
            ),
            "false_known_rate": float(selected["false_known_rate"]),
            "overall_detection_accuracy": float(
                selected["overall_detection_accuracy"]
            ),
        },
    }
    _write_json(config_path, config)
    return config


def _load_normalized(path: Path) -> np.ndarray:
    vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"Invalid embedding: {path}")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError(f"Zero embedding: {path}")
    return vector / norm


def build_speaker_disjoint_validation_trials(
    metadata_path: Path = EMBEDDING_METADATA,
    centroid_dir: Path = COSINE_CENTROID_DIR,
    output_path: Path = SV_TRIALS,
    *,
    include_unknown: bool = True,
) -> list[dict[str, Any]]:
    """Build genuine/impostor trials from cosine validation speakers only."""

    metadata = _read_csv(metadata_path)
    known = [
        row
        for row in metadata
        if row["protocol"] == COSINE_PROTOCOL and row["split"] == KNOWN_SPLIT
    ]
    unknown = [
        row
        for row in metadata
        if row["protocol"] == COSINE_PROTOCOL and row["split"] == UNKNOWN_SPLIT
    ]
    centroid_paths = sorted(centroid_dir.glob("*.npy"))
    centroids = {
        path.stem: _load_normalized(path)
        for path in centroid_paths
    }
    if len(centroids) < 2:
        raise ValueError("At least two validation enrolled centroids are required")

    trials: list[dict[str, Any]] = []

    def append_trial(
        row: dict[str, str],
        claimed_speaker: str,
        trial_type: str,
    ) -> None:
        query = _load_normalized(Path(row["embedding_path"]))
        score = float(query @ centroids[claimed_speaker])
        trials.append(
            {
                "trial_id": f"SVVAL{len(trials) + 1:04d}",
                "protocol": COSINE_PROTOCOL,
                "query_audio_id": row["audio_id"],
                "query_speaker_id": row["speaker_id"],
                "query_split": row["split"],
                "claimed_speaker_id": claimed_speaker,
                "trial_type": trial_type,
                "cosine_similarity": f"{score:.8f}",
            }
        )

    enrolled_speakers = sorted(centroids)
    for row in known:
        true_speaker = row["speaker_id"]
        if true_speaker not in centroids:
            raise ValueError(f"Missing genuine centroid for {true_speaker}")
        append_trial(row, true_speaker, "GENUINE")
        for impostor in enrolled_speakers:
            if impostor != true_speaker:
                append_trial(row, impostor, "IMPOSTOR")

    if include_unknown:
        for row in unknown:
            for claimed_speaker in enrolled_speakers:
                append_trial(row, claimed_speaker, "UNKNOWN_IMPOSTOR")

    _write_csv(output_path, TRIAL_FIELDS, trials)
    return trials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-unknown-trials", action="store_true")
    args = parser.parse_args()
    threshold = select_unknown_threshold()
    trials = build_speaker_disjoint_validation_trials(
        include_unknown=not args.no_unknown_trials
    )
    summary = {
        "selected_threshold": threshold["threshold"],
        "threshold_metrics": threshold["metrics"],
        "trial_count": len(trials),
        "genuine_count": sum(row["trial_type"] == "GENUINE" for row in trials),
        "impostor_count": sum(row["trial_type"] == "IMPOSTOR" for row in trials),
        "unknown_impostor_count": sum(
            row["trial_type"] == "UNKNOWN_IMPOSTOR" for row in trials
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
