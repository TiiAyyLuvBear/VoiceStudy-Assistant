from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUERY = ROOT / "data/processed/v1/metadata/cosine_test_query.csv"
UNKNOWN = ROOT / "data/processed/v1/metadata/cosine_test_unknown.csv"
EMBED_META = ROOT / "data/metadata/embedding_metadata.csv"
CENTROID_DIR = ROOT / "models/experimental/cosine_test_centroids"
OUTPUT = ROOT / "experiments/test/speaker_disjoint_verification_test_trials.csv"

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_embedding(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    x = np.asarray(np.load(p, allow_pickle=False), dtype=np.float32).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError(f"Invalid embedding: {p}")
    norm = np.linalg.norm(x)
    if norm <= np.finfo(np.float32).eps:
        raise ValueError(f"Zero embedding: {p}")
    return x / norm

def main():
    print("# WEEK 3 — CREATE SPEAKER-DISJOINT SV TEST TRIALS\n")

    query_rows = read_csv(QUERY)
    unknown_rows = read_csv(UNKNOWN)
    metadata = read_csv(EMBED_META)

    embedding_map = {r["audio_id"]: r["embedding_path"] for r in metadata}

    centroid_paths = sorted(CENTROID_DIR.glob("test_enrolled_spk_*.npy"))
    if len(centroid_paths) < 2:
        raise ValueError("Expected at least 2 test enrolled centroids.")

    centroids = {p.stem: load_embedding(p) for p in centroid_paths}
    speakers = sorted(centroids)

    print(f"Known query samples : {len(query_rows)}")
    print(f"Unknown query samples: {len(unknown_rows)}")
    print(f"Test centroids      : {len(speakers)}")
    print("Speakers:")
    for s in speakers:
        print(f"  {s}")

    trials = []

    def add_trial(row, claimed, trial_type):
        audio_id = row["audio_id"]
        if audio_id not in embedding_map:
            raise KeyError(f"Missing embedding for {audio_id}")

        query = load_embedding(embedding_map[audio_id])
        score = float(query @ centroids[claimed])

        trials.append({
            "trial_id": f"SVTEST{len(trials)+1:04d}",
            "protocol": "SPEAKER_DISJOINT_VERIFICATION_TEST",
            "query_audio_id": audio_id,
            "query_speaker_id": row["speaker_id"],
            "query_split": row.get("split_name", row.get("project_split", "")),
            "claimed_speaker_id": claimed,
            "trial_type": trial_type,
            "cosine_similarity": f"{score:.8f}",
        })

    for row in query_rows:
        true_speaker = row["normalized_speaker_id"]
        if true_speaker not in centroids:
            raise ValueError(
                f"Query speaker {true_speaker} has no test centroid."
            )

        add_trial(row, true_speaker, "GENUINE")

        for speaker in speakers:
            if speaker != true_speaker:
                add_trial(row, speaker, "IMPOSTOR")

    for row in unknown_rows:
        for speaker in speakers:
            add_trial(row, speaker, "UNKNOWN_IMPOSTOR")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "trial_id",
        "protocol",
        "query_audio_id",
        "query_speaker_id",
        "query_split",
        "claimed_speaker_id",
        "trial_type",
        "cosine_similarity",
    ]

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trials)

    genuine = sum(r["trial_type"] == "GENUINE" for r in trials)
    impostor = sum(r["trial_type"] == "IMPOSTOR" for r in trials)
    unknown_impostor = sum(r["trial_type"] == "UNKNOWN_IMPOSTOR" for r in trials)

    print("\nTRIAL SUMMARY")
    print("-" * 50)
    print(f"Genuine trials          : {genuine}")
    print(f"Impostor trials         : {impostor}")
    print(f"Unknown impostor trials : {unknown_impostor}")
    print(f"Total trials            : {len(trials)}")
    print(f"\nOutput: {OUTPUT}")
    print("\nWEEK 3 — STEP 7 COMPLETE")

if __name__ == "__main__":
    main()