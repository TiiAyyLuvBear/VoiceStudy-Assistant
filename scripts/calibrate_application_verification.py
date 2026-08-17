"""Calibrate application SV threshold from held-out command-audio trials."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.preprocessing import preprocess_audio
from src.speaker.embedding import ECAPAEmbeddingExtractor, extract_embedding


DEMO_DATA = PROJECT_ROOT / "experiments/system/demo_enrollment_data.csv"
CENTROID_DIR = PROJECT_ROOT / "models/application/user_embeddings"
RESULTS = PROJECT_ROOT / "experiments/system/application_verification_threshold_results.csv"
THRESHOLD = PROJECT_ROOT / "models/application/application_verification_threshold.json"


def _metrics(scores: list[tuple[float, int]], threshold: float) -> dict:
    genuine = [score for score, label in scores if label == 1]
    impostor = [score for score, label in scores if label == 0]
    false_reject = sum(score < threshold for score in genuine)
    false_accept = sum(score >= threshold for score in impostor)
    true_accept = len(genuine) - false_reject
    true_reject = len(impostor) - false_accept
    far = false_accept / len(impostor)
    frr = false_reject / len(genuine)
    accuracy = (true_accept + true_reject) / len(scores)
    precision = true_accept / (true_accept + false_accept) if true_accept + false_accept else 0.0
    recall = true_accept / len(genuine)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "FAR": far,
        "FRR": frr,
        "accuracy": accuracy,
        "F1": f1,
        "eer_gap": abs(far - frr),
        "mean_error": (far + frr) / 2,
    }


def calibrate() -> dict:
    with DEMO_DATA.open("r", encoding="utf-8-sig", newline="") as stream:
        heldout = [
            row for row in csv.DictReader(stream) if row["role"] == "HELDOUT_QUERY"
        ]
    if len(heldout) != 15:
        raise RuntimeError(f"Expected 15 held-out queries; found {len(heldout)}")

    preprocess_audio(PROJECT_ROOT / heldout[0]["audio_path"])
    extractor = ECAPAEmbeddingExtractor.from_config()
    centroids = {
        path.stem: np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        for path in sorted(CENTROID_DIR.glob("user_*.npy"))
        if path.stem in {"user_001", "user_002", "user_003"}
    }
    if set(centroids) != {"user_001", "user_002", "user_003"}:
        raise RuntimeError("Three application demo centroids are required")

    scores: list[tuple[float, int]] = []
    trial_rows = []
    for row in heldout:
        audio, sample_rate = preprocess_audio(PROJECT_ROOT / row["audio_path"])
        embedding, _, _ = extract_embedding(
            audio, sample_rate=sample_rate, extractor=extractor
        )
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query /= np.linalg.norm(query)
        for candidate_user_id, centroid in centroids.items():
            similarity = float(query @ (centroid / np.linalg.norm(centroid)))
            label = int(candidate_user_id == row["user_id"])
            scores.append((similarity, label))
            trial_rows.append(
                {
                    "recording_id": row["recording_id"],
                    "source_user_id": row["user_id"],
                    "candidate_user_id": candidate_user_id,
                    "label": label,
                    "similarity": f"{similarity:.8f}",
                }
            )

    unique = sorted({score for score, _ in scores})
    candidates = [unique[0] - 1e-6]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    candidates.append(unique[-1] + 1e-6)
    results = [_metrics(scores, threshold) for threshold in candidates]
    selected = min(
        results,
        key=lambda row: (
            row["eer_gap"],
            row["mean_error"],
            -row["F1"],
            -row["accuracy"],
            row["threshold"],
        ),
    )
    output_rows = []
    for row in results:
        output_rows.append(
            {
                **{key: f"{value:.8f}" for key, value in row.items()},
                "selected": str(row is selected).lower(),
            }
        )
    with RESULTS.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    threshold_document = {
        "protocol": "APPLICATION_SV",
        "threshold_type": "cosine_similarity",
        "threshold": round(selected["threshold"], 8),
        "comparison": "cosine_similarity >= threshold means VERIFIED",
        "source": "experiments/system/application_verification_threshold_results.csv",
        "selection": "application_validation",
        "selection_criterion": (
            "minimize absolute FAR-FRR gap; tie-break by mean error, F1, "
            "accuracy, then smaller threshold"
        ),
        "genuine_count": sum(label == 1 for _, label in scores),
        "impostor_count": sum(label == 0 for _, label in scores),
        "metrics": {
            key: round(selected[key], 8)
            for key in ("FAR", "FRR", "accuracy", "F1", "eer_gap", "mean_error")
        },
        "application_audio_calibrated": True,
        "calibration_data": (
            "15 held-out validation command audios from user_001..003; "
            "not used in enrollment centroids"
        ),
        "threshold_tuned_on_v2_test": False,
        "speaker_v2_modified": False,
    }
    THRESHOLD.write_text(
        json.dumps(threshold_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    trials_path = RESULTS.with_name("application_verification_validation_trials.csv")
    with trials_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    return threshold_document


def main() -> int:
    document = calibrate()
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

