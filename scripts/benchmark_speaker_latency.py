"""WEEK 3 — Benchmark speaker-system latency."""

from __future__ import annotations

import sys
import time
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_MANIFEST = PROJECT_ROOT / "data/processed/v1/metadata/svm_closed_set_test.csv"
EMBEDDING_METADATA = PROJECT_ROOT / "data/metadata/embedding_metadata.csv"
SVM_MODEL = PROJECT_ROOT / "models/experimental/speaker_svm_linear.pkl"
COSINE_CENTROIDS = PROJECT_ROOT / "models/experimental/svm_closed_set_centroids"
TEST_CENTROIDS = PROJECT_ROOT / "models/experimental/cosine_test_centroids"
APPLICATION_CONFIG = PROJECT_ROOT / "models/application/application_sid_config.json"
OUTPUT = PROJECT_ROOT / "experiments/test/speaker_latency_results.csv"

PROCESSED_AUDIO_ROOT = PROJECT_ROOT / "data/processed/v1/audio"
RAW_AUDIO_ROOT = PROJECT_ROOT / "data/audio"

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def resolve_audio_path(value: str) -> Path:
    p = Path(str(value).strip())
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            PROJECT_ROOT / p,
            PROCESSED_AUDIO_ROOT / p,
            PROCESSED_AUDIO_ROOT / p.name,
            RAW_AUDIO_ROOT / p,
            RAW_AUDIO_ROOT / p.name,
        ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Audio file not found. Tried:\n" +
        "\n".join(str(x) for x in candidates)
    )

def load_embedding(path: Path) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f"Invalid embedding: {path}")
    return x

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(a @ b)

def benchmark(fn, repeats: int = 1000) -> float:
    for _ in range(min(10, repeats)):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1000.0 / repeats

def benchmark_preprocessing(audio_path: Path) -> float:
    try:
        from src.audio.preprocessing import preprocess_audio
    except Exception as e:
        print(f"[WARN] preprocessing benchmark skipped: {e}")
        return None

    try:
        def run():
            preprocess_audio(str(audio_path))

        return benchmark(run, repeats=20)
    except Exception as e:
        print(f"[WARN] preprocessing benchmark skipped: {e}")
        return None

def benchmark_ecapa(audio_path: Path) -> float:
    try:
        from src.audio.preprocessing import preprocess_audio
        from src.speaker.embedding import ECAPAEmbeddingExtractor
    except Exception as e:
        print(f"[WARN] ECAPA benchmark skipped: {e}")
        return None

    try:
        config_path = PROJECT_ROOT / "config.yaml"
        extractor = ECAPAEmbeddingExtractor.from_config(config_path)

        audio, sample_rate = preprocess_audio(str(audio_path))

        def run():
            extractor.extract(audio, sample_rate=sample_rate)

        return benchmark(run, repeats=20)
    except Exception as e:
        print(f"[WARN] ECAPA benchmark skipped: {e}")
        return None

def load_centroids(directory: Path) -> dict[str, np.ndarray]:
    centroids = {}
    if not directory.is_dir():
        return centroids

    for path in sorted(directory.glob("*.npy")):
        centroids[path.stem] = load_embedding(path)

    return centroids

def main() -> int:
    print("# WEEK 3 — SPEAKER LATENCY BENCHMARK")
    print()

    if not TEST_MANIFEST.is_file():
        raise FileNotFoundError(f"Test manifest does not exist: {TEST_MANIFEST}")

    if not EMBEDDING_METADATA.is_file():
        raise FileNotFoundError(f"Embedding metadata does not exist: {EMBEDDING_METADATA}")

    if not SVM_MODEL.is_file():
        raise FileNotFoundError(f"SVM model does not exist: {SVM_MODEL}")

    test_rows = read_csv(TEST_MANIFEST)
    embedding_rows = read_csv(EMBEDDING_METADATA)

    embedding_map = {
        str(row["audio_id"]): row["embedding_path"]
        for row in embedding_rows
    }

    if not test_rows:
        raise ValueError("Test manifest is empty.")

    sample = test_rows[0]
    audio_path = resolve_audio_path(sample["audio_path"])

    embedding_path_value = embedding_map.get(str(sample["audio_id"]))
    if not embedding_path_value:
        raise KeyError(f"No embedding found for audio_id={sample['audio_id']}")

    embedding_path = Path(embedding_path_value)
    if not embedding_path.is_absolute():
        embedding_path = PROJECT_ROOT / embedding_path

    if not embedding_path.is_file():
        raise FileNotFoundError(f"Embedding file does not exist: {embedding_path}")

    query_embedding = load_embedding(embedding_path)

    bundle = joblib.load(SVM_MODEL)
    model = bundle["model"]

    svm_result = model.predict(query_embedding.reshape(1, -1))[0]

    closed_centroids = load_centroids(COSINE_CENTROIDS)
    test_centroids = load_centroids(TEST_CENTROIDS)

    if not closed_centroids:
        raise ValueError(f"No closed-set centroids found: {COSINE_CENTROIDS}")

    if not test_centroids:
        raise ValueError(f"No speaker-disjoint test centroids found: {TEST_CENTROIDS}")

    closed_result = max(
        closed_centroids,
        key=lambda speaker: cosine(query_embedding, closed_centroids[speaker])
    )

    disjoint_result = max(
        test_centroids,
        key=lambda speaker: cosine(query_embedding, test_centroids[speaker])
    )

    threshold = None
    if APPLICATION_CONFIG.is_file():
        import json
        config = json.loads(APPLICATION_CONFIG.read_text(encoding="utf-8"))
        threshold = config.get("sid", {}).get("threshold")
        if threshold is None:
            threshold = config.get("application_sid_threshold")
        if threshold is None:
            threshold = config.get("threshold")

    if threshold is None:
        threshold = 0.51307271

    max_test_similarity = max(
        cosine(query_embedding, centroid)
        for centroid in test_centroids.values()
    )

    unknown_result = max_test_similarity >= float(threshold)

    sv_threshold = 0.37547598
    verification_score = cosine(
        query_embedding,
        next(iter(test_centroids.values()))
    )
    verification_result = verification_score >= sv_threshold

    preprocessing_ms = benchmark_preprocessing(audio_path)
    ecapa_ms = benchmark_ecapa(audio_path)

    svm_ms = benchmark(
        lambda: model.predict(query_embedding.reshape(1, -1)),
        repeats=1000,
    )

    closed_cosine_ms = benchmark(
        lambda: max(
            cosine(query_embedding, centroid)
            for centroid in closed_centroids.values()
        ),
        repeats=1000,
    )

    disjoint_cosine_ms = benchmark(
        lambda: max(
            cosine(query_embedding, centroid)
            for centroid in test_centroids.values()
        ),
        repeats=1000,
    )

    unknown_ms = benchmark(
        lambda: max(
            cosine(query_embedding, centroid)
            for centroid in test_centroids.values()
        ) >= float(threshold),
        repeats=1000,
    )

    verification_ms = benchmark(
        lambda: cosine(
            query_embedding,
            next(iter(test_centroids.values()))
        ) >= sv_threshold,
        repeats=1000,
    )

    application_ms = benchmark(
        lambda: max(
            cosine(query_embedding, centroid)
            for centroid in test_centroids.values()
        ),
        repeats=1000,
    )

    results = [
        ("preprocessing", preprocessing_ms),
        ("ecapa_embedding", ecapa_ms),
        ("linear_svm_inference", svm_ms),
        ("closed_set_cosine_sid", closed_cosine_ms),
        ("speaker_disjoint_cosine_sid", disjoint_cosine_ms),
        ("unknown_detection", unknown_ms),
        ("speaker_verification", verification_ms),
        ("application_cosine_sid", application_ms),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "mean_latency_ms"])
        for stage, latency in results:
            writer.writerow([
                stage,
                "" if latency is None else f"{latency:.6f}",
            ])

    print("## RESULTS")
    print()

    for stage, latency in results:
        if latency is None:
            print(f"{stage:35s} SKIPPED")
        else:
            print(f"{stage:35s} {latency:.3f} ms")

    print()
    print("Artifact:")
    print(OUTPUT)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())