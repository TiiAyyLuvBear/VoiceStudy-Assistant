"""Create speaker-disjoint test enrollment centroids."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SELECTED_SPEAKERS = ROOT / "data/metadata/selected_test_enrolled_speakers.csv"
EMBEDDING_METADATA = ROOT / "data/metadata/embedding_metadata.csv"
OUTPUT_DIR = ROOT / "models/experimental/cosine_test_centroids"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_embedding(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p

    if not p.is_file():
        raise FileNotFoundError(f"Embedding not found: {p}")

    x = np.asarray(np.load(p, allow_pickle=False), dtype=np.float32).reshape(-1)

    if not np.isfinite(x).all():
        raise ValueError(f"Embedding contains NaN/Inf: {p}")

    norm = np.linalg.norm(x)
    if norm <= np.finfo(np.float32).eps:
        raise ValueError(f"Zero embedding: {p}")

    return x / norm


def main() -> int:
    print("# WEEK 3 — CREATE SPEAKER-DISJOINT TEST GALLERY\n")

    selected = read_csv(SELECTED_SPEAKERS)
    embedding_meta = read_csv(EMBEDDING_METADATA)

    required_selected = {
        "speaker_id",
        "normalized_speaker_id",
        "protocol",
        "role",
        "project_split",
    }
    missing = required_selected - set(selected[0].keys())
    if missing:
        raise KeyError(f"Selected speaker manifest missing: {sorted(missing)}")

    required_embedding = {"audio_id", "embedding_path"}
    missing = required_embedding - set(embedding_meta[0].keys())
    if missing:
        raise KeyError(f"Embedding metadata missing: {sorted(missing)}")

    # Mapping được lấy trực tiếp từ selected_test_enrolled_speakers.csv.
    speaker_mapping = {
        row["speaker_id"]: row["normalized_speaker_id"]
        for row in selected
        if row["protocol"] == "COSINE_TEST"
        and row["role"] == "ENROLLED"
        and row["project_split"] == "TEST"
    }

    if not speaker_mapping:
        raise ValueError("No COSINE_TEST enrolled speakers found.")

    embedding_by_audio = {
        row["audio_id"]: row["embedding_path"]
        for row in embedding_meta
    }

    enrollment_manifest = ROOT / "data/processed/v1/metadata/cosine_test_enrollment.csv"
    enrollment_rows = read_csv(enrollment_manifest)

    print(f"Enrollment samples: {len(enrollment_rows)}")
    print(f"Test enrolled speakers: {len(speaker_mapping)}")
    print("Speakers:")

    for speaker_id, normalized_id in speaker_mapping.items():
        print(f"{normalized_id} -> {speaker_id}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created = 0

    for speaker_id, normalized_id in speaker_mapping.items():
        rows = [r for r in enrollment_rows if r["speaker_id"] == speaker_id]

        if not rows:
            raise ValueError(
                f"No enrollment samples found for speaker: {speaker_id}"
            )

        embeddings = []

        for row in rows:
            audio_id = row["audio_id"]

            if audio_id not in embedding_by_audio:
                raise KeyError(
                    f"Missing embedding metadata for audio_id: {audio_id}"
                )

            embeddings.append(
                load_embedding(embedding_by_audio[audio_id])
            )

        matrix = np.stack(embeddings)
        centroid = matrix.mean(axis=0)

        norm = np.linalg.norm(centroid)
        if norm <= np.finfo(np.float32).eps:
            raise ValueError(
                f"Centroid has zero norm: {normalized_id}"
            )

        centroid = centroid / norm

        output_path = OUTPUT_DIR / f"{normalized_id}.npy"
        np.save(output_path, centroid)

        print(
            f"{normalized_id}: "
            f"{len(embeddings)} embeddings -> {output_path.name}"
        )

        created += 1

    print("\nGallery created successfully.")
    print(f"Centroids created: {created}")
    print(f"Output: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())