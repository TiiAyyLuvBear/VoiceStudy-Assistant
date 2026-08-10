"""Week 3 — Speaker-disjoint known query SID test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

QUERY_MANIFEST = ROOT / "data/processed/v1/metadata/cosine_test_query.csv"
EMBEDDING_METADATA = ROOT / "data/metadata/embedding_metadata.csv"
CENTROID_DIR = ROOT / "models/experimental/cosine_test_centroids"
THRESHOLD_CONFIG = ROOT / "models/experimental/cosine_unknown_threshold.json"

OUTPUT_PREDICTIONS = ROOT / "experiments/test/speaker_disjoint_sid_test_predictions.csv"
OUTPUT_METRICS = ROOT / "experiments/test/speaker_disjoint_sid_test_metrics.json"

EXPECTED_THRESHOLD = 0.51307271


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_embedding(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p

    if not p.is_file():
        raise FileNotFoundError(f"Embedding not found: {p}")

    x = np.asarray(np.load(p, allow_pickle=False), dtype=np.float32).reshape(-1)

    if not np.isfinite(x).all():
        raise ValueError(f"Invalid embedding: {p}")

    norm = np.linalg.norm(x)
    if norm <= np.finfo(np.float32).eps:
        raise ValueError(f"Zero embedding: {p}")

    return x / norm


def main():
    print("# WEEK 3 — SPEAKER-DISJOINT KNOWN QUERY TEST\n")

    queries = read_csv(QUERY_MANIFEST)
    metadata = read_csv(EMBEDDING_METADATA)

    embedding_map = {
        row["audio_id"]: row["embedding_path"]
        for row in metadata
    }

    if not CENTROID_DIR.is_dir():
        raise FileNotFoundError(f"Centroid directory not found: {CENTROID_DIR}")

    centroid_paths = sorted(CENTROID_DIR.glob("test_enrolled_spk_*.npy"))

    if not centroid_paths:
        raise FileNotFoundError(
            f"No test centroids found in: {CENTROID_DIR}"
        )

    centroids = {
        p.stem: load_embedding(str(p))
        for p in centroid_paths
    }

    threshold_data = json.loads(
        THRESHOLD_CONFIG.read_text(encoding="utf-8")
    )

    threshold = float(threshold_data["threshold"])

    if not np.isclose(threshold, EXPECTED_THRESHOLD, atol=1e-8):
        raise ValueError(
            f"Unexpected unknown threshold: {threshold}. "
            f"Expected locked threshold: {EXPECTED_THRESHOLD}"
        )

    print(f"Query samples       : {len(queries)}")
    print(f"Test centroids      : {len(centroids)}")
    print(f"Unknown threshold   : {threshold:.8f}")
    print("Threshold applied   : YES")
    print()

    predictions = []

    for row in queries:
        audio_id = row["audio_id"]
        true_speaker = row["normalized_speaker_id"]

        if audio_id not in embedding_map:
            raise KeyError(
                f"Missing embedding metadata for audio_id: {audio_id}"
            )

        query = load_embedding(embedding_map[audio_id])

        scores = {
            speaker: float(query @ centroid)
            for speaker, centroid in centroids.items()
        }

        candidate = max(scores, key=scores.get)
        max_similarity = scores[candidate]

        if max_similarity >= threshold:
            predicted = candidate
            decision = "KNOWN"
        else:
            predicted = "UNKNOWN"
            decision = "UNKNOWN"

        correct = predicted == true_speaker

        predictions.append({
            "audio_id": audio_id,
            "true_speaker_id": true_speaker,
            "candidate_speaker_id": candidate,
            "max_similarity": f"{max_similarity:.8f}",
            "threshold": f"{threshold:.8f}",
            "decision": decision,
            "predicted_speaker_id": predicted,
            "correct": str(correct).lower(),
        })

    known_count = len(predictions)

    correct_known = sum(
        r["true_speaker_id"] == r["predicted_speaker_id"]
        for r in predictions
    )

    accepted_known = sum(
        r["decision"] == "KNOWN"
        for r in predictions
    )

    false_unknown = sum(
        r["decision"] == "UNKNOWN"
        for r in predictions
    )

    known_identification_accuracy = (
        correct_known / known_count if known_count else 0.0
    )

    known_acceptance_rate = (
        accepted_known / known_count if known_count else 0.0
    )

    false_unknown_rate = (
        false_unknown / known_count if known_count else 0.0
    )

    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PREDICTIONS.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(predictions[0].keys()),
        )
        writer.writeheader()
        writer.writerows(predictions)

    metrics = {
        "protocol": "WEEK3_SPEAKER_DISJOINT_KNOWN_QUERY_TEST",
        "query_count": known_count,
        "centroid_count": len(centroids),
        "unknown_threshold": threshold,
        "threshold_source": str(THRESHOLD_CONFIG),
        "threshold_tuning_on_test": False,
        "metrics": {
            "known_identification_accuracy": known_identification_accuracy,
            "known_acceptance_rate": known_acceptance_rate,
            "false_unknown_rate": false_unknown_rate,
        },
        "manifest": str(QUERY_MANIFEST),
        "centroid_dir": str(CENTROID_DIR),
    }

    OUTPUT_METRICS.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("RESULTS")
    print("-" * 60)
    print(f"Known query samples       : {known_count}")
    print(f"Correct identification    : {correct_known}/{known_count}")
    print(f"Known identification acc. : {known_identification_accuracy:.6f}")
    print(f"Known acceptance rate     : {known_acceptance_rate:.6f}")
    print(f"False-unknown rate        : {false_unknown_rate:.6f}")

    print()
    print("Artifacts:")
    print(OUTPUT_PREDICTIONS)
    print(OUTPUT_METRICS)

    print("\nWEEK 3 — STEP 5 COMPLETE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())