"""
Evaluate cosine-similarity closed-set baseline on the frozen Week 3 test split.

Week 3 — Step 3
----------------
Protocol:
    COSINE_CLOSED_SET_TEST

Input:
    data/processed/v1/metadata/svm_closed_set_test.csv

Frozen cosine gallery:
    models/experimental/svm_closed_set_centroids/

Embedding metadata:
    data/metadata/embedding_metadata.csv

Outputs:
    experiments/test/cosine_closed_set_predictions.csv
    experiments/test/cosine_closed_set_metrics.json
    experiments/test/closed_set_svm_vs_cosine.csv

Important:
    - This is CLOSED-SET evaluation.
    - Every test sample belongs to one of the 10 known speakers.
    - DO NOT apply the unknown threshold.
    - The unknown threshold is only for speaker-disjoint/open-set SID.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# PROJECT ROOT
# ============================================================
#
# File location:
#
#   PROJECT_ROOT/
#       scripts/
#           evaluate/
#               cosine_closed_set_test.py
#
# Therefore:
#   parents[0] = evaluate
#   parents[1] = scripts
#   parents[2] = PROJECT_ROOT
#
# This avoids depending on the current PowerShell directory.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUTS
# ============================================================

TEST_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v1"
    / "metadata"
    / "svm_closed_set_test.csv"
)

EMBEDDING_METADATA = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "embedding_metadata.csv"
)

COSINE_CENTROID_DIR = (
    PROJECT_ROOT
    / "models"
    / "experimental"
    / "svm_closed_set_centroids"
)

SVM_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "test"
    / "svm_closed_set_predictions.csv"
)

SVM_METRICS = (
    PROJECT_ROOT
    / "experiments"
    / "test"
    / "svm_closed_set_metrics.json"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "test"
    / "cosine_closed_set_predictions.csv"
)

OUTPUT_METRICS = (
    PROJECT_ROOT
    / "experiments"
    / "test"
    / "cosine_closed_set_metrics.json"
)

OUTPUT_COMPARISON = (
    PROJECT_ROOT
    / "experiments"
    / "test"
    / "closed_set_svm_vs_cosine.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_SPEAKER_COUNT = 10
EXPECTED_EMBEDDING_DIM = 192

PROTOCOL = "WEEK3_COSINE_CLOSED_SET_TEST"


# ============================================================
# HELPERS
# ============================================================

def normalize_speaker_id(value: Any) -> str:
    """
    Normalize speaker IDs so that IDs coming from CSV files
    are compared consistently.

    Important:
        pandas may interpret numeric-looking IDs as integers.
        Therefore everything is converted to string.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Avoid values such as "12345.0" if pandas inferred numeric.
    if value.endswith(".0"):
        try:
            numeric = float(value)
            if numeric.is_integer():
                value = str(int(numeric))
        except ValueError:
            pass

    return value


def resolve_path(path_value: str | Path) -> Path:
    """
    Resolve a path robustly.

    Relative paths are interpreted relative to PROJECT_ROOT,
    not the current working directory.
    """

    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_embedding(path_value: str | Path) -> np.ndarray:
    """
    Load and validate a single ECAPA embedding.
    """

    path = resolve_path(path_value)

    if not path.is_file():
        raise FileNotFoundError(
            f"Embedding file does not exist:\n{path}"
        )

    embedding = np.asarray(
        np.load(
            path,
            allow_pickle=False,
        ),
        dtype=np.float32,
    ).reshape(-1)

    if embedding.ndim != 1:
        raise ValueError(
            f"Embedding must be 1-D: {path}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            f"Embedding contains NaN/Inf:\n{path}"
        )

    if len(embedding) != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch:\n"
            f"File: {path}\n"
            f"Expected: {EXPECTED_EMBEDDING_DIM}\n"
            f"Got: {len(embedding)}"
        )

    return embedding


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalize an embedding or centroid.
    """

    vector = np.asarray(
        vector,
        dtype=np.float32,
    ).reshape(-1)

    norm = float(np.linalg.norm(vector))

    if norm <= np.finfo(np.float32).eps:
        raise ValueError(
            "Cannot normalize zero vector."
        )

    return vector / norm


def read_json(path: Path) -> dict[str, Any]:
    """
    Read JSON if it exists.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file does not exist:\n{path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# LOAD TEST MANIFEST
# ============================================================

def load_test_manifest() -> pd.DataFrame:

    print("Test manifest:")
    print(TEST_MANIFEST)
    print()

    if not TEST_MANIFEST.is_file():

        raise FileNotFoundError(
            "Test manifest does not exist:\n"
            f"{TEST_MANIFEST}\n\n"
            "Expected file:\n"
            "data/processed/v1/metadata/"
            "svm_closed_set_test.csv"
        )

    df = pd.read_csv(
        TEST_MANIFEST,
        encoding="utf-8-sig",
    )

    if df.empty:
        raise ValueError(
            "Test manifest is empty."
        )

    print(
        f"Test samples: {len(df)}"
    )

    print()
    print("Manifest columns:")
    print(df.columns.tolist())
    print()

    required = {
        "audio_id",
        "normalized_speaker_id",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            "Test manifest is missing columns:\n"
            f"{sorted(missing)}"
        )

    df["audio_id"] = (
        df["audio_id"]
        .apply(normalize_speaker_id)
    )

    df["normalized_speaker_id"] = (
        df["normalized_speaker_id"]
        .apply(normalize_speaker_id)
    )

    return df


# ============================================================
# RESOLVE EMBEDDING PATHS
# ============================================================

def resolve_embedding_column(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:

    # --------------------------------------------------------
    # Case 1:
    # Test manifest already contains embedding_path
    # --------------------------------------------------------

    if "embedding_path" in df.columns:

        print(
            "Embedding source: "
            "embedding_path in test manifest"
        )

        return df, "embedding_path"

    # --------------------------------------------------------
    # Case 2:
    # Need to merge embedding_metadata.csv
    # --------------------------------------------------------

    if not EMBEDDING_METADATA.is_file():

        raise FileNotFoundError(
            "Test manifest does not contain "
            "'embedding_path', and embedding metadata "
            "does not exist:\n"
            f"{EMBEDDING_METADATA}"
        )

    embedding_metadata = pd.read_csv(
        EMBEDDING_METADATA,
        encoding="utf-8-sig",
        dtype={
            "audio_id": str,
        },
    )

    print(
        "Embedding source:"
        " data/metadata/embedding_metadata.csv"
    )

    print(
        "Embedding metadata columns:"
    )

    print(
        embedding_metadata.columns.tolist()
    )

    print()

    required = {
        "audio_id",
        "embedding_path",
    }

    missing = (
        required
        - set(embedding_metadata.columns)
    )

    if missing:

        raise KeyError(
            "embedding_metadata.csv is missing "
            f"columns: {sorted(missing)}"
        )

    embedding_metadata["audio_id"] = (
        embedding_metadata["audio_id"]
        .apply(normalize_speaker_id)
    )

    embedding_metadata = (
        embedding_metadata[
            [
                "audio_id",
                "embedding_path",
            ]
        ]
        .drop_duplicates(
            subset=["audio_id"],
            keep="first",
        )
    )

    # --------------------------------------------------------
    # Merge using string audio_id on both sides.
    #
    # This prevents:
    #
    # object vs int64
    #
    # merge errors.
    # --------------------------------------------------------

    df = df.merge(
        embedding_metadata,
        on="audio_id",
        how="left",
        validate="one_to_one",
    )

    return df, "embedding_path"


# ============================================================
# LOAD FROZEN COSINE CENTROIDS
# ============================================================

def load_centroids() -> dict[str, np.ndarray]:

    print(
        "Frozen cosine centroids:"
    )

    print(
        COSINE_CENTROID_DIR
    )

    print()

    if not COSINE_CENTROID_DIR.is_dir():

        raise FileNotFoundError(
            "Cosine centroid directory does not exist:\n"
            f"{COSINE_CENTROID_DIR}"
        )

    centroid_files = sorted(
        COSINE_CENTROID_DIR.glob("*.npy")
    )

    if not centroid_files:

        raise FileNotFoundError(
            "No .npy centroid files found in:\n"
            f"{COSINE_CENTROID_DIR}"
        )

    centroids: dict[str, np.ndarray] = {}

    for path in centroid_files:

        speaker_id = normalize_speaker_id(
            path.stem
        )

        centroid = load_embedding(path)

        centroid = l2_normalize(
            centroid
        )

        centroids[speaker_id] = centroid

    print(
        f"Number of speakers : "
        f"{len(centroids)}"
    )

    print(
        f"Embedding dimension: "
        f"{EXPECTED_EMBEDDING_DIM}"
    )

    print("Speakers:")

    for speaker in sorted(centroids):

        print(
            speaker
        )

    print()

    if len(centroids) != EXPECTED_SPEAKER_COUNT:

        raise ValueError(
            "Unexpected number of cosine centroids.\n"
            f"Expected: {EXPECTED_SPEAKER_COUNT}\n"
            f"Found: {len(centroids)}"
        )

    return centroids


# ============================================================
# COSINE CLOSED-SET PREDICTION
# ============================================================

def predict_closed_set(
    df: pd.DataFrame,
    embedding_col: str,
    centroids: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:

    speakers = sorted(
        centroids.keys()
    )

    centroid_matrix = np.vstack(
        [
            centroids[speaker]
            for speaker in speakers
        ]
    ).astype(
        np.float32
    )

    embeddings = []
    true_labels = []

    rows = []

    for _, row in df.iterrows():

        embedding = load_embedding(
            row[embedding_col]
        )

        embedding = l2_normalize(
            embedding
        )

        similarities = (
            centroid_matrix
            @ embedding
        )

        best_index = int(
            np.argmax(similarities)
        )

        predicted_speaker = (
            speakers[best_index]
        )

        best_similarity = float(
            similarities[best_index]
        )

        true_speaker = normalize_speaker_id(
            row["normalized_speaker_id"]
        )

        embeddings.append(
            embedding
        )

        true_labels.append(
            true_speaker
        )

        rows.append(
            {
                "audio_id": normalize_speaker_id(
                    row["audio_id"]
                ),
                "true_speaker_id": true_speaker,
                "predicted_speaker_id":
                    predicted_speaker,
                "max_similarity":
                    best_similarity,
                "correct":
                    bool(
                        predicted_speaker
                        == true_speaker
                    ),
            }
        )

    return (
        np.asarray(
            embeddings,
            dtype=np.float32,
        ),
        np.asarray(
            true_labels,
            dtype=str,
        ),
        rows,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[
    dict[str, Any],
    list[str],
    np.ndarray,
]:

    labels = sorted(
        set(y_true)
        | set(y_pred)
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    per_speaker = {}

    for speaker in sorted(
        set(y_true)
    ):

        mask = (
            y_true == speaker
        )

        count = int(
            np.sum(mask)
        )

        correct = int(
            np.sum(
                y_pred[mask]
                == y_true[mask]
            )
        )

        speaker_accuracy = (
            correct / count
            if count > 0
            else 0.0
        )

        per_speaker[speaker] = {
            "count": count,
            "correct": correct,
            "accuracy":
                float(
                    speaker_accuracy
                ),
        }

    metrics = {

        "accuracy":
            float(accuracy),

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "per_speaker_accuracy":
            per_speaker,

    }

    return (
        metrics,
        labels,
        cm,
    )


# ============================================================
# LOAD SVM RESULTS
# ============================================================

def load_svm_results() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
]:

    if not SVM_PREDICTIONS.is_file():

        raise FileNotFoundError(
            "SVM closed-set predictions do not exist:\n"
            f"{SVM_PREDICTIONS}\n\n"
            "Step 2 must be completed before Step 3."
        )

    svm_df = pd.read_csv(
        SVM_PREDICTIONS,
        encoding="utf-8-sig",
        dtype={
            "audio_id": str,
            "predicted_speaker_id": str,
        },
    )

    if "audio_id" not in svm_df.columns:

        raise KeyError(
            "SVM predictions are missing "
            "'audio_id'."
        )

    if "predicted_speaker_id" not in svm_df.columns:

        raise KeyError(
            "SVM predictions are missing "
            "'predicted_speaker_id'."
        )

    svm_df["audio_id"] = (
        svm_df["audio_id"]
        .apply(normalize_speaker_id)
    )

    svm_df["predicted_speaker_id"] = (
        svm_df["predicted_speaker_id"]
        .apply(normalize_speaker_id)
    )

    svm_predictions = {}

    for _, row in svm_df.iterrows():

        audio_id = normalize_speaker_id(
            row["audio_id"]
        )

        svm_predictions[audio_id] = {
            "predicted_speaker_id":
                normalize_speaker_id(
                    row["predicted_speaker_id"]
                ),
        }

    svm_metrics = {}

    if SVM_METRICS.is_file():

        svm_metrics = read_json(
            SVM_METRICS
        )

    return (
        svm_predictions,
        svm_metrics,
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    df: pd.DataFrame,
    prediction_rows: list[dict[str, Any]],
) -> None:

    prediction_lookup = {
        row["audio_id"]: row
        for row in prediction_rows
    }

    output = df.copy()

    output["cosine_predicted_speaker_id"] = (
        output["audio_id"]
        .map(
            lambda x:
                prediction_lookup[
                    normalize_speaker_id(x)
                ]["predicted_speaker_id"]
        )
    )

    output["cosine_max_similarity"] = (
        output["audio_id"]
        .map(
            lambda x:
                prediction_lookup[
                    normalize_speaker_id(x)
                ]["max_similarity"]
        )
    )

    output["cosine_correct"] = (
        output["audio_id"]
        .map(
            lambda x:
                prediction_lookup[
                    normalize_speaker_id(x)
                ]["correct"]
        )
    )

    OUTPUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
        encoding="utf-8",
    )


# ============================================================
# SAVE SVM VS COSINE COMPARISON
# ============================================================

def save_comparison(
    prediction_rows: list[dict[str, Any]],
    svm_predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:

    rows = []

    svm_correct_count = 0
    cosine_correct_count = 0
    both_correct = 0
    svm_only_correct = 0
    cosine_only_correct = 0
    both_wrong = 0

    for row in prediction_rows:

        audio_id = normalize_speaker_id(
            row["audio_id"]
        )

        true_speaker = normalize_speaker_id(
            row["true_speaker_id"]
        )

        cosine_prediction = normalize_speaker_id(
            row["predicted_speaker_id"]
        )

        cosine_correct = bool(
            cosine_prediction
            == true_speaker
        )

        if audio_id not in svm_predictions:

            raise ValueError(
                "SVM predictions do not contain "
                f"audio_id: {audio_id}"
            )

        svm_prediction = normalize_speaker_id(
            svm_predictions[audio_id][
                "predicted_speaker_id"
            ]
        )

        svm_correct = bool(
            svm_prediction
            == true_speaker
        )

        if svm_correct:
            svm_correct_count += 1

        if cosine_correct:
            cosine_correct_count += 1

        if svm_correct and cosine_correct:
            both_correct += 1

        elif svm_correct and not cosine_correct:
            svm_only_correct += 1

        elif not svm_correct and cosine_correct:
            cosine_only_correct += 1

        else:
            both_wrong += 1

        rows.append(
            {
                "audio_id":
                    audio_id,

                "true_speaker_id":
                    true_speaker,

                "svm_predicted_speaker_id":
                    svm_prediction,

                "cosine_predicted_speaker_id":
                    cosine_prediction,

                "svm_correct":
                    svm_correct,

                "cosine_correct":
                    cosine_correct,

                "agreement":
                    bool(
                        svm_prediction
                        == cosine_prediction
                    ),

                "cosine_max_similarity":
                    float(
                        row["max_similarity"]
                    ),
            }
        )

    fieldnames = [
        "audio_id",
        "true_speaker_id",
        "svm_predicted_speaker_id",
        "cosine_predicted_speaker_id",
        "svm_correct",
        "cosine_correct",
        "agreement",
        "cosine_max_similarity",
    ]

    OUTPUT_COMPARISON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df = pd.DataFrame(
        rows,
        columns=fieldnames,
    )

    comparison_df.to_csv(
        OUTPUT_COMPARISON,
        index=False,
        encoding="utf-8",
    )

    total = len(rows)

    summary = {

        "test_sample_count":
            total,

        "svm_accuracy":
            float(
                svm_correct_count / total
                if total
                else 0.0
            ),

        "cosine_accuracy":
            float(
                cosine_correct_count / total
                if total
                else 0.0
            ),

        "accuracy_difference_cosine_minus_svm":
            float(
                (
                    cosine_correct_count
                    - svm_correct_count
                )
                / total
                if total
                else 0.0
            ),

        "both_correct":
            int(both_correct),

        "svm_only_correct":
            int(svm_only_correct),

        "cosine_only_correct":
            int(cosine_only_correct),

        "both_wrong":
            int(both_wrong),

        "agreement_count":
            int(
                comparison_df[
                    "agreement"
                ].sum()
            ),

        "agreement_rate":
            float(
                comparison_df[
                    "agreement"
                ].mean()
                if total
                else 0.0
            ),

    }

    return summary


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print(
        "# WEEK 3 — COSINE CLOSED-SET BASELINE"
    )

    print()

    # --------------------------------------------------------
    # 1. Show project paths
    # --------------------------------------------------------

    print(
        "PROJECT ROOT:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    print(
        "TEST MANIFEST:"
    )

    print(
        TEST_MANIFEST
    )

    print(
        "EXISTS:",
        TEST_MANIFEST.exists(),
    )

    print()

    if not TEST_MANIFEST.is_file():

        raise FileNotFoundError(
            "Test manifest does not exist:\n"
            f"{TEST_MANIFEST}"
        )

    # --------------------------------------------------------
    # 2. Load manifest
    # --------------------------------------------------------

    df = load_test_manifest()

    # --------------------------------------------------------
    # 3. Resolve embedding paths
    # --------------------------------------------------------

    df, embedding_col = (
        resolve_embedding_column(df)
    )

    missing = df[
        df[embedding_col].isna()
        |
        (
            df[embedding_col]
            .astype(str)
            .str.strip()
            == ""
        )
    ]

    if not missing.empty:

        missing_ids = (
            missing["audio_id"]
            .astype(str)
            .tolist()
        )

        raise FileNotFoundError(
            "Could not resolve embedding paths "
            f"for {len(missing_ids)} test samples.\n"
            f"Missing audio_ids: "
            f"{missing_ids[:20]}"
        )

    print(
        f"Resolved embeddings: "
        f"{df[embedding_col].notna().sum()}"
        f"/{len(df)}"
    )

    print()

    # --------------------------------------------------------
    # 4. Load frozen centroids
    # --------------------------------------------------------

    centroids = load_centroids()

    # --------------------------------------------------------
    # 5. Validate closed-set condition
    # --------------------------------------------------------

    test_speakers = sorted(
        set(
            df[
                "normalized_speaker_id"
            ]
            .apply(
                normalize_speaker_id
            )
        )
    )

    centroid_speakers = sorted(
        centroids.keys()
    )

    print(
        "Closed-set validation:"
    )

    print(
        "Test speakers     :",
        len(test_speakers),
    )

    print(
        "Centroid speakers :",
        len(centroid_speakers),
    )

    print()

    missing_centroids = (
        set(test_speakers)
        - set(centroid_speakers)
    )

    extra_centroids = (
        set(centroid_speakers)
        - set(test_speakers)
    )

    if missing_centroids:

        raise ValueError(
            "Test speakers without cosine centroids:\n"
            f"{sorted(missing_centroids)}"
        )

    if extra_centroids:

        raise ValueError(
            "Cosine centroids contain speakers "
            "not present in test manifest:\n"
            f"{sorted(extra_centroids)}"
        )

    # --------------------------------------------------------
    # 6. IMPORTANT:
    #    CLOSED-SET DOES NOT USE UNKNOWN THRESHOLD
    # --------------------------------------------------------

    print(
        "Unknown threshold:"
    )

    print(
        "NOT APPLIED"
    )

    print()

    # --------------------------------------------------------
    # 7. Predict
    # --------------------------------------------------------

    (
        _,
        y_true,
        prediction_rows,
    ) = predict_closed_set(
        df,
        embedding_col,
        centroids,
    )

    y_pred = np.asarray(
        [
            row[
                "predicted_speaker_id"
            ]
            for row in prediction_rows
        ],
        dtype=str,
    )

    # --------------------------------------------------------
    # 8. Metrics
    # --------------------------------------------------------

    (
        metrics,
        labels,
        cm,
    ) = calculate_metrics(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # 9. Save predictions
    # --------------------------------------------------------

    save_predictions(
        df,
        prediction_rows,
    )

    # --------------------------------------------------------
    # 10. Load SVM results
    # --------------------------------------------------------

    (
        svm_predictions,
        svm_metrics,
    ) = load_svm_results()

    # --------------------------------------------------------
    # 11. Compare SVM vs Cosine
    # --------------------------------------------------------

    comparison_summary = (
        save_comparison(
            prediction_rows,
            svm_predictions,
        )
    )

    # --------------------------------------------------------
    # 12. Save metrics
    # --------------------------------------------------------

    metrics_output = {

        "protocol":
            PROTOCOL,

        "test_sample_count":
            int(len(df)),

        "embedding_dim":
            EXPECTED_EMBEDDING_DIM,

        "num_classes":
            len(centroids),

        "unknown_threshold_applied":
            False,

        "threshold_policy":
            "Closed-set evaluation only; "
            "no unknown threshold is applied.",

        "manifest":
            str(
                TEST_MANIFEST.relative_to(
                    PROJECT_ROOT
                )
            ),

        "centroid_directory":
            str(
                COSINE_CENTROID_DIR.relative_to(
                    PROJECT_ROOT
                )
            ),

        "metrics":
            metrics,

        "labels":
            labels,

        "confusion_matrix":
            cm.tolist(),

        "svm_comparison":
            comparison_summary,

        "svm_metrics_source":
            str(
                SVM_METRICS.relative_to(
                    PROJECT_ROOT
                )
            )
            if SVM_METRICS.is_file()
            else None,

    }

    write_json(
        OUTPUT_METRICS,
        metrics_output,
    )

    # --------------------------------------------------------
    # 13. Final report
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "RESULTS — COSINE CLOSED-SET BASELINE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Accuracy        : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Macro Precision : "
        f"{metrics['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall    : "
        f"{metrics['macro_recall']:.6f}"
    )

    print(
        f"Macro F1        : "
        f"{metrics['macro_f1']:.6f}"
    )

    print()

    print(
        "Per-speaker accuracy:"
    )

    for speaker, result in (
        metrics[
            "per_speaker_accuracy"
        ]
        .items()
    ):

        print(
            f"  {speaker}: "
            f"{result['correct']}/"
            f"{result['count']} "
            f"("
            f"{result['accuracy']:.4f}"
            f")"
        )

    print()

    print(
        "SVM vs Cosine:"
    )

    print(
        f"  SVM accuracy     : "
        f"{comparison_summary['svm_accuracy']:.6f}"
    )

    print(
        f"  Cosine accuracy  : "
        f"{comparison_summary['cosine_accuracy']:.6f}"
    )

    print(
        f"  Difference       : "
        f"{comparison_summary['accuracy_difference_cosine_minus_svm']:.6f}"
    )

    print(
        f"  Both correct     : "
        f"{comparison_summary['both_correct']}"
    )

    print(
        f"  SVM only correct : "
        f"{comparison_summary['svm_only_correct']}"
    )

    print(
        f"  Cosine only      : "
        f"{comparison_summary['cosine_only_correct']}"
    )

    print(
        f"  Both wrong       : "
        f"{comparison_summary['both_wrong']}"
    )

    print(
        f"  Agreement rate   : "
        f"{comparison_summary['agreement_rate']:.6f}"
    )

    print()

    print(
        "Artifacts:"
    )

    print(
        OUTPUT_PREDICTIONS
    )

    print(
        OUTPUT_METRICS
    )

    print(
        OUTPUT_COMPARISON
    )

    print()

    print(
        "WEEK 3 — STEP 3 COMPLETE"
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )