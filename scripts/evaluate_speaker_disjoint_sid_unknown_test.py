"""Week 3 — Speaker-disjoint unknown query SID test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

UNKNOWN_MANIFEST = ROOT / "data/processed/v1/metadata/cosine_test_unknown.csv"
EMBEDDING_METADATA = ROOT / "data/metadata/embedding_metadata.csv"
CENTROID_DIR = ROOT / "models/experimental/cosine_test_centroids"
THRESHOLD_CONFIG = ROOT / "models/experimental/cosine_unknown_threshold.json"

KNOWN_METRICS = ROOT / "experiments/test/speaker_disjoint_sid_test_metrics.json"

OUTPUT_PREDICTIONS = ROOT / "experiments/test/speaker_disjoint_sid_unknown_predictions.csv"
OUTPUT_METRICS = ROOT / "experiments/test/speaker_disjoint_sid_open_set_metrics.json"

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
    print("# WEEK 3 — SPEAKER-DISJOINT UNKNOWN QUERY TEST\n")

    unknown_rows = read_csv(UNKNOWN_MANIFEST)
    metadata = read_csv(EMBEDDING_METADATA)

    embedding_map = {
        row["audio_id"]: row["embedding_path"]
        for row in metadata
    }

    centroid_paths = sorted(
        CENTROID_DIR.glob("test_enrolled_spk_*.npy")
    )

    if not centroid_paths:
        raise FileNotFoundError(
            f"No test centroids found: {CENTROID_DIR}"
        )

    centroids = {
        p.stem: load_embedding(str(p))
        for p in centroid_paths
    }

    threshold_config = json.loads(
        THRESHOLD_CONFIG.read_text(encoding="utf-8")
    )

    threshold = float(threshold_config["threshold"])

    if not np.isclose(threshold, EXPECTED_THRESHOLD, atol=1e-8):
        raise ValueError(
            f"Threshold mismatch: {threshold} "
            f"(expected {EXPECTED_THRESHOLD})"
        )

    print(f"Unknown query samples : {len(unknown_rows)}")
    print(f"Test centroids        : {len(centroids)}")
    print(f"Unknown threshold     : {threshold:.8f}")
    print("Threshold tuned on test: NO")
    print()

    predictions = []

    for row in unknown_rows:
        audio_id = row["audio_id"]

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

        accepted_as_known = max_similarity >= threshold
        decision = "KNOWN" if accepted_as_known else "UNKNOWN"

        predictions.append({
            "audio_id": audio_id,
            "unknown_speaker_id": row["speaker_id"],
            "candidate_speaker_id": candidate,
            "max_similarity": f"{max_similarity:.8f}",
            "threshold": f"{threshold:.8f}",
            "decision": decision,
            "correct_rejection": str(not accepted_as_known).lower(),
        })

    total = len(predictions)

    unknown_rejected = sum(
        r["decision"] == "UNKNOWN"
        for r in predictions
    )

    false_known = sum(
        r["decision"] == "KNOWN"
        for r in predictions
    )

    unknown_rejection_rate = (
        unknown_rejected / total
        if total else 0.0
    )

    false_known_rate = (
        false_known / total
        if total else 0.0
    )

    # ---------------------------------------------------------
    # Load known-query results from Step 5
    # ---------------------------------------------------------

    if not KNOWN_METRICS.is_file():
        raise FileNotFoundError(
            f"Step 5 metrics not found: {KNOWN_METRICS}"
        )

    known_metrics = json.loads(
        KNOWN_METRICS.read_text(encoding="utf-8")
    )

    known_count = int(
        known_metrics["query_count"]
    )

    known_correct = round(
        known_metrics["metrics"]["known_identification_accuracy"]
        * known_count
    )

    open_set_correct = (
        known_correct + unknown_rejected
    )

    open_set_total = (
        known_count + total
    )

    open_set_accuracy = (
        open_set_correct / open_set_total
        if open_set_total else 0.0
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    OUTPUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    # ---------------------------------------------------------
    # Save combined open-set metrics
    # ---------------------------------------------------------

    metrics = {
        "protocol": "WEEK3_SPEAKER_DISJOINT_OPEN_SET_SID_TEST",
        "threshold": threshold,
        "threshold_source": str(THRESHOLD_CONFIG),
        "threshold_tuning_on_test": False,
        "known_query": {
            "count": known_count,
            "correct_identification": known_correct,
            "identification_accuracy": known_metrics["metrics"][
                "known_identification_accuracy"
            ],
            "acceptance_rate": known_metrics["metrics"][
                "known_acceptance_rate"
            ],
            "false_unknown_rate": known_metrics["metrics"][
                "false_unknown_rate"
            ],
        },
        "unknown_query": {
            "count": total,
            "rejected_as_unknown": unknown_rejected,
            "false_known": false_known,
            "unknown_rejection_rate": unknown_rejection_rate,
            "false_known_rate": false_known_rate,
        },
        "open_set": {
            "total_queries": open_set_total,
            "correct": open_set_correct,
            "overall_accuracy": open_set_accuracy,
        },
        "manifests": {
            "known": str(
                ROOT / "data/processed/v1/metadata/cosine_test_query.csv"
            ),
            "unknown": str(UNKNOWN_MANIFEST),
        },
        "centroid_dir": str(CENTROID_DIR),
    }

    OUTPUT_METRICS.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("RESULTS")
    print("-" * 60)
    print(f"Unknown queries          : {total}")
    print(f"Rejected as UNKNOWN      : {unknown_rejected}")
    print(f"False KNOWN              : {false_known}")
    print(f"Unknown rejection rate   : {unknown_rejection_rate:.6f}")
    print(f"False-known rate         : {false_known_rate:.6f}")
    print()
    print(f"Open-set total queries   : {open_set_total}")
    print(f"Open-set correct         : {open_set_correct}")
    print(f"Open-set overall accuracy: {open_set_accuracy:.6f}")

    print()
    print("Artifacts:")
    print(OUTPUT_PREDICTIONS)
    print(OUTPUT_METRICS)

    print("\nWEEK 3 — STEP 6 COMPLETE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())