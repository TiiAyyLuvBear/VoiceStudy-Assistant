"""Create expanded Week 3 speaker-disjoint splits with duplicate checks."""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# CONFIG
# ============================================================

# Đổi đường dẫn nếu cần
SOURCE_MANIFEST = PROJECT_ROOT / "data/metadata/data_inventory.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/metadata/new"

ENROLLMENT_N = 5
SVM_SPEAKERS = 10
COSINE_VAL_ENROLLED = 2
COSINE_VAL_UNKNOWN = 2
COSINE_TEST_ENROLLED = 2
COSINE_TEST_UNKNOWN = 2

TOTAL_SPEAKERS = (
    SVM_SPEAKERS
    + COSINE_VAL_ENROLLED
    + COSINE_VAL_UNKNOWN
    + COSINE_TEST_ENROLLED
    + COSINE_TEST_UNKNOWN
)

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / f"{name}_new.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved: {path}")
    return path


def require_columns(df: pd.DataFrame) -> None:
    required = {"audio_id", "speaker_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if "normalized_speaker_id" not in df.columns:
        df["normalized_speaker_id"] = df["speaker_id"].astype(str)


def split_70_15_15(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(np.floor(n * 0.70))
    n_val = int(np.floor(n * 0.15))
    n_test = n - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Not enough samples for 70/15/15 split: {n} samples"
        )

    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train:n_train + n_val].copy(),
        df.iloc[n_train + n_val:].copy(),
    )


def check_no_duplicate_audio(files: dict[str, pd.DataFrame]) -> None:
    seen = {}
    duplicates = []

    for name, df in files.items():
        for audio_id in df["audio_id"].astype(str):
            if audio_id in seen:
                duplicates.append(
                    (audio_id, seen[audio_id], name)
                )
            else:
                seen[audio_id] = name

    if duplicates:
        print("\nDUPLICATE AUDIO FOUND:")
        for audio_id, first, second in duplicates[:20]:
            print(f"  {audio_id}: {first} <-> {second}")
        raise ValueError(
            f"Found {len(duplicates)} duplicated audio_id(s) across output files."
        )

    print("\nPASS: No duplicated audio_id across all 10 files.")


def check_speaker_disjoint(
    groups: dict[str, set[str]]
) -> None:
    names = list(groups.keys())
    errors = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            overlap = groups[a] & groups[b]
            if overlap:
                errors.append(
                    (a, b, sorted(overlap))
                )

    if errors:
        print("\nSPEAKER-DISJOINT CHECK FAILED:")
        for a, b, overlap in errors:
            print(f"  {a} <-> {b}: {overlap}")
        raise ValueError("Speaker-disjoint protocol violated.")

    print("PASS: Speaker-disjoint protocol verified.")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("# WEEK 3 — CREATE EXPANDED SPEAKER SPLITS")
    print()

    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Source manifest not found:\n{SOURCE_MANIFEST}"
        )

    df = pd.read_csv(
        SOURCE_MANIFEST,
        encoding="utf-8-sig",
    )

    require_columns(df)

    print(f"Source manifest : {SOURCE_MANIFEST}")
    print(f"Total valid rows: {len(df)}")
    print()

    # --------------------------------------------------------
    # Filter valid audio
    # --------------------------------------------------------

    if "is_valid" in df.columns:
        valid_values = df["is_valid"].astype(str).str.lower()
        df = df[valid_values.isin(["true", "1", "yes"])].copy()

    df["audio_id"] = df["audio_id"].astype(str)
    df["speaker_id"] = df["speaker_id"].astype(str)
    df["normalized_speaker_id"] = df["normalized_speaker_id"].astype(str)

    df = df.drop_duplicates(
        subset=["audio_id"]
    ).copy()

    print(f"Valid unique audio: {len(df)}")
    print()

    # --------------------------------------------------------
    # Speaker statistics
    # --------------------------------------------------------

    speaker_counts = (
        df.groupby("speaker_id")
        .size()
        .sort_values(ascending=False)
    )

    if len(speaker_counts) < TOTAL_SPEAKERS:
        raise ValueError(
            f"Need {TOTAL_SPEAKERS} speakers, "
            f"but only {len(speaker_counts)} available."
        )

    # Every selected speaker needs at least 6 samples.
    eligible = speaker_counts[speaker_counts >= ENROLLMENT_N + 1]

    if len(eligible) < TOTAL_SPEAKERS:
        raise ValueError(
            f"Not enough speakers with >= {ENROLLMENT_N + 1} audio samples. "
            f"Eligible: {len(eligible)}"
        )

    selected_ids = eligible.head(TOTAL_SPEAKERS).index.tolist()

    print("Selected 18 speakers:")
    for i, speaker in enumerate(selected_ids, 1):
        print(
            f"{i:02d}. {speaker} "
            f"({int(speaker_counts[speaker])} audio)"
        )

    print()

    # --------------------------------------------------------
    # Assign speaker roles
    # --------------------------------------------------------

    svm_speakers = selected_ids[:10]
    val_enrolled_speakers = selected_ids[10:12]
    val_unknown_speakers = selected_ids[12:14]
    test_enrolled_speakers = selected_ids[14:16]
    test_unknown_speakers = selected_ids[16:18]

    print("Speaker allocation:")
    print(f"SVM enrolled      : {len(svm_speakers)}")
    print(f"Cosine val known  : {len(val_enrolled_speakers)}")
    print(f"Cosine val unknown: {len(val_unknown_speakers)}")
    print(f"Cosine test known : {len(test_enrolled_speakers)}")
    print(f"Cosine test unknown: {len(test_unknown_speakers)}")
    print()

    # --------------------------------------------------------
    # Create SVM splits
    # --------------------------------------------------------

    svm_enrollment_rows = []
    svm_train_rows = []
    svm_validation_rows = []
    svm_test_rows = []

    rng = np.random.default_rng(RANDOM_SEED)

    for speaker in svm_speakers:
        rows = df[
            df["speaker_id"] == speaker
        ].copy()

        rows = rows.sample(
            frac=1,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

        if len(rows) < ENROLLMENT_N + 3:
            raise ValueError(
                f"Speaker {speaker} has too few samples: {len(rows)}"
            )

        enrollment = rows.iloc[:ENROLLMENT_N].copy()
        remaining = rows.iloc[ENROLLMENT_N:].copy()

        train, validation, test = split_70_15_15(
            remaining
        )

        enrollment["protocol"] = "SVM_CLOSED_SET"
        enrollment["role"] = "ENROLLED"
        enrollment["project_split"] = "ENROLLMENT"
        enrollment["split_name"] = (
            f"exp_svm_spk_{svm_speakers.index(speaker)+1:04d}"
        )

        train["protocol"] = "SVM_CLOSED_SET"
        train["role"] = "TRAIN"
        train["project_split"] = "TRAIN"
        train["split_name"] = (
            f"exp_svm_spk_{svm_speakers.index(speaker)+1:04d}"
        )

        validation["protocol"] = "SVM_CLOSED_SET"
        validation["role"] = "VALIDATION"
        validation["project_split"] = "VALIDATION"
        validation["split_name"] = (
            f"exp_svm_spk_{svm_speakers.index(speaker)+1:04d}"
        )

        test["protocol"] = "SVM_CLOSED_SET"
        test["role"] = "TEST"
        test["project_split"] = "TEST"
        test["split_name"] = (
            f"exp_svm_spk_{svm_speakers.index(speaker)+1:04d}"
        )

        svm_enrollment_rows.append(enrollment)
        svm_train_rows.append(train)
        svm_validation_rows.append(validation)
        svm_test_rows.append(test)

    svm_enrollment = pd.concat(
        svm_enrollment_rows,
        ignore_index=True,
    )

    svm_train = pd.concat(
        svm_train_rows,
        ignore_index=True,
    )

    svm_validation = pd.concat(
        svm_validation_rows,
        ignore_index=True,
    )

    svm_test = pd.concat(
        svm_test_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Create cosine validation split
    # --------------------------------------------------------

    cosine_val_enrollment_rows = []
    cosine_val_query_rows = []
    cosine_val_unknown_rows = []

    for index, speaker in enumerate(
        val_enrolled_speakers,
        1,
    ):
        rows = df[
            df["speaker_id"] == speaker
        ].copy()

        rows = rows.sample(
            frac=1,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

        enrollment = rows.iloc[:ENROLLMENT_N].copy()
        query = rows.iloc[ENROLLMENT_N:].copy()

        enrollment["protocol"] = "COSINE_VALIDATION"
        enrollment["role"] = "ENROLLED"
        enrollment["project_split"] = "VALIDATION"
        enrollment["split_name"] = (
            f"cosine_validation_spk_{index:04d}"
        )

        query["protocol"] = "COSINE_VALIDATION"
        query["role"] = "QUERY"
        query["project_split"] = "VALIDATION"
        query["split_name"] = (
            f"cosine_validation_spk_{index:04d}"
        )

        cosine_val_enrollment_rows.append(enrollment)
        cosine_val_query_rows.append(query)

    for index, speaker in enumerate(
        val_unknown_speakers,
        1,
    ):
        rows = df[
            df["speaker_id"] == speaker
        ].copy()

        rows = rows.sample(
            frac=1,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

        rows["protocol"] = "COSINE_VALIDATION"
        rows["role"] = "UNKNOWN"
        rows["project_split"] = "VALIDATION"
        rows["split_name"] = (
            f"cosine_validation_unknown_{index:04d}"
        )

        cosine_val_unknown_rows.append(rows)

    cosine_validation_enrollment = pd.concat(
        cosine_val_enrollment_rows,
        ignore_index=True,
    )

    cosine_validation_query = pd.concat(
        cosine_val_query_rows,
        ignore_index=True,
    )

    cosine_validation_unknown = pd.concat(
        cosine_val_unknown_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Create cosine test split
    # --------------------------------------------------------

    cosine_test_enrollment_rows = []
    cosine_test_query_rows = []
    cosine_test_unknown_rows = []

    for index, speaker in enumerate(
        test_enrolled_speakers,
        1,
    ):
        rows = df[
            df["speaker_id"] == speaker
        ].copy()

        rows = rows.sample(
            frac=1,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

        enrollment = rows.iloc[:ENROLLMENT_N].copy()
        query = rows.iloc[ENROLLMENT_N:].copy()

        enrollment["protocol"] = "COSINE_TEST"
        enrollment["role"] = "ENROLLED"
        enrollment["project_split"] = "TEST"
        enrollment["split_name"] = (
            f"cosine_test_spk_{index:04d}"
        )

        query["protocol"] = "COSINE_TEST"
        query["role"] = "QUERY"
        query["project_split"] = "TEST"
        query["split_name"] = (
            f"cosine_test_spk_{index:04d}"
        )

        cosine_test_enrollment_rows.append(enrollment)
        cosine_test_query_rows.append(query)

    for index, speaker in enumerate(
        test_unknown_speakers,
        1,
    ):
        rows = df[
            df["speaker_id"] == speaker
        ].copy()

        rows = rows.sample(
            frac=1,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

        rows["protocol"] = "COSINE_TEST"
        rows["role"] = "UNKNOWN"
        rows["project_split"] = "TEST"
        rows["split_name"] = (
            f"cosine_test_unknown_{index:04d}"
        )

        cosine_test_unknown_rows.append(rows)

    cosine_test_enrollment = pd.concat(
        cosine_test_enrollment_rows,
        ignore_index=True,
    )

    cosine_test_query = pd.concat(
        cosine_test_query_rows,
        ignore_index=True,
    )

    cosine_test_unknown = pd.concat(
        cosine_test_unknown_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save exactly 10 files
    # --------------------------------------------------------

    files = {
        "svm_closed_set_enrollment": svm_enrollment,
        "svm_closed_set_train": svm_train,
        "svm_closed_set_validation": svm_validation,
        "svm_closed_set_test": svm_test,
        "cosine_validation_enrollment": cosine_validation_enrollment,
        "cosine_validation_query": cosine_validation_query,
        "cosine_validation_unknown": cosine_validation_unknown,
        "cosine_test_enrollment": cosine_test_enrollment,
        "cosine_test_query": cosine_test_query,
        "cosine_test_unknown": cosine_test_unknown,
    }

    print("# OUTPUT")
    print()

    for name, output_df in files.items():
        save_csv(
            output_df,
            name,
        )

    # --------------------------------------------------------
    # Speaker-disjoint validation
    # --------------------------------------------------------

    speaker_groups = {
        "SVM": set(svm_speakers),
        "COSINE_VALIDATION_ENROLLED": set(val_enrolled_speakers),
        "COSINE_VALIDATION_UNKNOWN": set(val_unknown_speakers),
        "COSINE_TEST_ENROLLED": set(test_enrolled_speakers),
        "COSINE_TEST_UNKNOWN": set(test_unknown_speakers),
    }

    print()
    print("# SPEAKER-DISJOINT CHECK")
    check_speaker_disjoint(
        speaker_groups
    )

    # --------------------------------------------------------
    # Audio duplicate validation
    # --------------------------------------------------------

    print()
    print("# AUDIO DUPLICATE CHECK")
    check_no_duplicate_audio(
        files
    )

    # --------------------------------------------------------
    # Enrollment count validation
    # --------------------------------------------------------

    print()
    print("# ENROLLMENT CHECK")

    expected_enrollment = {
        "svm_closed_set_enrollment": 10 * ENROLLMENT_N,
        "cosine_validation_enrollment": 2 * ENROLLMENT_N,
        "cosine_test_enrollment": 2 * ENROLLMENT_N,
    }

    for name, expected in expected_enrollment.items():
        actual = len(files[name])

        if actual != expected:
            raise ValueError(
                f"{name}: expected {expected}, got {actual}"
            )

        print(
            f"PASS: {name}: {actual} samples"
        )

    # --------------------------------------------------------
    # Per-speaker SVM split summary
    # --------------------------------------------------------

    print()
    print("# SVM SPLIT SUMMARY")

    for speaker in svm_speakers:
        e = int(
            (svm_enrollment["speaker_id"] == speaker).sum()
        )
        t = int(
            (svm_train["speaker_id"] == speaker).sum()
        )
        v = int(
            (svm_validation["speaker_id"] == speaker).sum()
        )
        te = int(
            (svm_test["speaker_id"] == speaker).sum()
        )

        print(
            f"{speaker}: "
            f"enroll={e}, "
            f"train={t}, "
            f"val={v}, "
            f"test={te}, "
            f"total={e+t+v+te}"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("# FINAL SUMMARY")
    print(f"Speakers selected : {TOTAL_SPEAKERS}")
    print(f"SVM speakers      : {len(svm_speakers)}")
    print(f"Validation speakers: 4")
    print(f"Test speakers     : 4")
    print()
    print("Audio counts:")

    for name, output_df in files.items():
        print(
            f"{name:35s}: {len(output_df)}"
        )

    print()
    print(
        "WEEK 3 — EXPANDED SPEAKER SPLITS COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())