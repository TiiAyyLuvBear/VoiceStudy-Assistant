"""Benchmark Speaker v2 stages without reading or writing v1 artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from scripts.benchmark_speaker_latency import (
    benchmark,
    benchmark_ecapa,
    benchmark_preprocessing,
    cosine,
    load_centroids,
    load_embedding,
    read_csv,
    resolve_audio_path,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_MANIFEST = ROOT / "data/processed/v2/metadata/svm_closed_set_test.csv"
EMBEDDING_METADATA = ROOT / "data/processed/v2/embedding_metadata.csv"
MODEL_DIR = ROOT / "models/experimental/v2"
OUTPUT = ROOT / "experiments/v2/test/speaker_latency_results.csv"


def main() -> int:
    test_rows = read_csv(TEST_MANIFEST)
    embedding_rows = read_csv(EMBEDDING_METADATA)
    embedding_map = {
        row["audio_id"]: Path(row["embedding_path"])
        for row in embedding_rows
    }
    sample = test_rows[0]
    audio_path = resolve_audio_path(sample["audio_path"])
    query = load_embedding(embedding_map[sample["audio_id"]])
    payload = joblib.load(MODEL_DIR / "speaker_svm_linear.pkl")
    model = payload["model"]
    closed = load_centroids(MODEL_DIR / "svm_closed_set_centroids")
    disjoint = load_centroids(MODEL_DIR / "cosine_test_centroids")
    sid_threshold = float(
        json.loads(
            (MODEL_DIR / "cosine_unknown_threshold.json").read_text(
                encoding="utf-8"
            )
        )["threshold"]
    )
    verification_threshold = float(
        json.loads(
            (MODEL_DIR / "verification_threshold.json").read_text(
                encoding="utf-8"
            )
        )["threshold"]
    )
    first_test_centroid = next(iter(disjoint.values()))
    results = [
        ("preprocessing", benchmark_preprocessing(audio_path)),
        ("ecapa_embedding", benchmark_ecapa(audio_path)),
        (
            "linear_svm_inference",
            benchmark(lambda: model.predict(query.reshape(1, -1)), repeats=1000),
        ),
        (
            "closed_set_cosine_sid",
            benchmark(
                lambda: max(cosine(query, value) for value in closed.values()),
                repeats=1000,
            ),
        ),
        (
            "speaker_disjoint_cosine_sid",
            benchmark(
                lambda: max(cosine(query, value) for value in disjoint.values()),
                repeats=1000,
            ),
        ),
        (
            "unknown_detection",
            benchmark(
                lambda: max(cosine(query, value) for value in disjoint.values())
                >= sid_threshold,
                repeats=1000,
            ),
        ),
        (
            "speaker_verification",
            benchmark(
                lambda: cosine(query, first_test_centroid)
                >= verification_threshold,
                repeats=1000,
            ),
        ),
        (
            "application_cosine_sid",
            benchmark(
                lambda: max(cosine(query, value) for value in disjoint.values()),
                repeats=1000,
            ),
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "mean_latency_ms"])
        for stage, latency in results:
            writer.writerow([stage, "" if latency is None else f"{latency:.6f}"])
    print(json.dumps(dict(results), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
