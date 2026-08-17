"""Evaluate Linear SVM on the frozen Week 3 closed-set test split."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "experimental"
    / "speaker_svm_linear.pkl"
)

EMBEDDING_METADATA = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "embedding_metadata.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "test"
    / "svm_closed_set_predictions.csv"
)

OUTPUT_METRICS = (
    PROJECT_ROOT
    / "experiments"
    / "test"
    / "svm_closed_set_metrics.json"
)

OUTPUT_CONFUSION_MATRIX = (
    PROJECT_ROOT
    / "experiments"
    / "test"
    / "svm_closed_set_confusion_matrix.png"
)


# ============================================================
# HELPERS
# ============================================================

def load_embedding(path: str) -> np.ndarray:
    """Load and validate one ECAPA embedding."""

    embedding_path = Path(path)

    if not embedding_path.is_absolute():
        embedding_path = PROJECT_ROOT / embedding_path

    if not embedding_path.is_file():
        raise FileNotFoundError(
            f"Embedding file not found:\n{embedding_path}"
        )

    x = np.asarray(
        np.load(
            embedding_path,
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    x = x.reshape(-1)

    if not np.isfinite(x).all():
        raise ValueError(
            f"Embedding contains NaN/Inf:\n{embedding_path}"
        )

    return x


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Write JSON artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("# WEEK 3 — SVM CLOSED-SET TEST")
    print()

    # ========================================================
    # 1. LOAD TEST MANIFEST
    # ========================================================

    if not TEST_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Test manifest does not exist:\n{TEST_MANIFEST}"
        )

    df = pd.read_csv(
        TEST_MANIFEST,
        encoding="utf-8-sig",
    )

    print(
        f"Test samples: {len(df)}"
    )

    print()
    print("Manifest columns:")
    print(df.columns.tolist())
    print()

    if df.empty:
        raise ValueError(
            "The SVM closed-set test manifest is empty."
        )


    # ========================================================
    # 2. LOAD FROZEN SVM
    # ========================================================

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"SVM model does not exist:\n{MODEL_PATH}"
        )

    bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(bundle, dict):
        raise TypeError(
            "Expected speaker_svm_linear.pkl "
            "to contain a dictionary."
        )

    if "model" not in bundle:
        raise KeyError(
            "SVM bundle does not contain key 'model'."
        )

    model = bundle["model"]

    model_type = bundle.get(
        "model_type",
        type(model).__name__,
    )

    selected_c = bundle.get(
        "selected_C",
        getattr(model, "C", None),
    )

    embedding_dim = bundle.get(
        "embedding_dim",
        None,
    )

    classes = bundle.get(
        "classes",
        getattr(model, "classes_", None),
    )

    if classes is None:
        raise KeyError(
            "Could not determine SVM classes."
        )

    print("Frozen SVM:")
    print(
        f"Model type : {model_type}"
    )
    print(
        f"C          : {selected_c}"
    )
    print(
        f"Embedding  : {embedding_dim}"
    )
    print(
        f"Classes    : {len(classes)}"
    )
    print()


    # ========================================================
    # 3. RESOLVE EMBEDDING PATHS
    # ========================================================

    if "embedding_path" in df.columns:

        embedding_col = "embedding_path"

    else:

        print(
            "Test manifest does not contain "
            "'embedding_path'."
        )

        print(
            "Loading embedding metadata:"
        )

        print(
            EMBEDDING_METADATA
        )

        if not EMBEDDING_METADATA.is_file():
            raise FileNotFoundError(
                "The test manifest does not contain "
                "'embedding_path', and the embedding "
                "metadata file does not exist:\n"
                f"{EMBEDDING_METADATA}"
            )

        embedding_metadata = pd.read_csv(
            EMBEDDING_METADATA,
            encoding="utf-8-sig",
        )

        required_columns = {
            "audio_id",
            "embedding_path",
        }

        missing_columns = (
            required_columns
            - set(embedding_metadata.columns)
        )

        if missing_columns:
            raise KeyError(
                "embedding_metadata.csv is missing "
                f"columns: {sorted(missing_columns)}"
            )

        embedding_metadata = (
            embedding_metadata[
                [
                    "audio_id",
                    "embedding_path",
                ]
            ]
            .drop_duplicates(
                subset=["audio_id"]
            )
        )

        df = df.merge(
            embedding_metadata,
            on="audio_id",
            how="left",
            validate="one_to_one",
        )

        embedding_col = "embedding_path"


    # ========================================================
    # 4. VALIDATE EMBEDDING PATHS
    # ========================================================

    missing_embeddings = df[
        df[embedding_col].isna()
        |
        (
            df[embedding_col]
            .astype(str)
            .str.strip()
            == ""
        )
    ]

    if not missing_embeddings.empty:

        missing_ids = (
            missing_embeddings[
                "audio_id"
            ]
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

    print(
        "Unique embedding paths: "
        f"{df[embedding_col].nunique()}"
    )

    print()


    # ========================================================
    # 5. LOAD TEST EMBEDDINGS
    # ========================================================

    X = []
    y_true = []

    for _, row in df.iterrows():

        embedding_path = Path(
            str(
                row[embedding_col]
            )
        )

        if not embedding_path.is_absolute():
            embedding_path = (
                PROJECT_ROOT
                / embedding_path
            )

        if not embedding_path.is_file():
            raise FileNotFoundError(
                "Embedding file does not exist:\n"
                f"{embedding_path}"
            )

        embedding = np.asarray(
            np.load(
                embedding_path,
                allow_pickle=False,
            ),
            dtype=np.float32,
        )

        embedding = embedding.reshape(-1)

        # --------------------------------------------
        # Check embedding dimension
        # --------------------------------------------

        if embedding_dim is not None:

            if (
                len(embedding)
                != int(embedding_dim)
            ):
                raise ValueError(
                    f"Embedding dimension mismatch "
                    f"for audio_id="
                    f"{row['audio_id']}: "
                    f"expected "
                    f"{embedding_dim}, "
                    f"got "
                    f"{len(embedding)}"
                )

        # --------------------------------------------
        # Check NaN / Inf
        # --------------------------------------------

        if not np.isfinite(
            embedding
        ).all():

            raise ValueError(
                "Embedding contains NaN/Inf:\n"
                f"{embedding_path}"
            )

        X.append(
            embedding
        )

        # --------------------------------------------
        # Ground truth
        # --------------------------------------------

        if (
            "normalized_speaker_id"
            not in df.columns
        ):
            raise KeyError(
                "Test manifest does not contain "
                "'normalized_speaker_id'."
            )

        y_true.append(
            str(
                row[
                    "normalized_speaker_id"
                ]
            )
        )


    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y_true = np.asarray(
        y_true,
        dtype=str,
    )


    # ========================================================
    # 6. SANITY CHECK
    # ========================================================

    print(
        f"Embedding matrix shape: "
        f"{X.shape}"
    )

    if embedding_dim is not None:

        expected_shape = (
            len(df),
            int(embedding_dim),
        )

        if X.shape != expected_shape:
            raise ValueError(
                "Unexpected embedding matrix "
                f"shape: {X.shape}; "
                f"expected {expected_shape}"
            )

    print()


    # ========================================================
    # 7. CLOSED-SET SVM PREDICTION
    # ========================================================
    #
    # IMPORTANT:
    #
    # Week 3 closed-set test MUST NOT apply
    # unknown-speaker threshold.
    #
    # Every test sample is forced to one of
    # the 10 known SVM classes.
    #
    # ========================================================

    y_pred = model.predict(
        X
    )

    y_pred = np.asarray(
        y_pred,
        dtype=str,
    )


    # ========================================================
    # 8. METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )


    # ========================================================
    # 9. LABELS + CONFUSION MATRIX
    # ========================================================

    labels = sorted(
        set(y_true)
        |
        set(y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )


    # ========================================================
    # 10. PER-SPEAKER ACCURACY
    # ========================================================

    per_speaker = {}

    for speaker in sorted(
        set(y_true)
    ):

        mask = (
            y_true
            == speaker
        )

        count = int(
            mask.sum()
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
            "accuracy": float(
                speaker_accuracy
            ),
        }


    # ========================================================
    # 11. SAVE PREDICTIONS
    # ========================================================

    prediction_df = df.copy()

    prediction_df[
        "predicted_speaker_id"
    ] = y_pred

    prediction_df[
        "correct"
    ] = (
        y_true
        == y_pred
    )

    OUTPUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
        encoding="utf-8",
    )


    # ========================================================
    # 12. SAVE METRICS JSON
    # ========================================================

    metrics = {

        "protocol":
            "WEEK3_SVM_CLOSED_SET_TEST",

        "test_sample_count":
            int(len(df)),

        "model_type":
            model_type,

        "C":
            float(selected_c),

        "embedding_dim":
            int(embedding_dim),

        "num_classes":
            int(len(classes)),

        "unknown_threshold_applied":
            False,

        "metrics": {

            "accuracy":
                float(accuracy),

            "macro_precision":
                float(
                    macro_precision
                ),

            "macro_recall":
                float(
                    macro_recall
                ),

            "macro_f1":
                float(
                    macro_f1
                ),
        },

        "per_speaker_accuracy":
            per_speaker,

        "labels":
            labels,

        "manifest":
            str(
                TEST_MANIFEST
            ),

        "model":
            str(
                MODEL_PATH
            ),
    }

    write_json(
        OUTPUT_METRICS,
        metrics,
    )


    # ========================================================
    # 13. CONFUSION MATRIX IMAGE
    # ========================================================

    OUTPUT_CONFUSION_MATRIX.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.title(
        "SVM Closed-Set Test Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(labels)),
        labels,
    )

    plt.xlabel(
        "Predicted speaker"
    )

    plt.ylabel(
        "True speaker"
    )

    for i in range(
        len(labels)
    ):

        for j in range(
            len(labels)
        ):

            plt.text(
                j,
                i,
                str(
                    int(
                        cm[i, j]
                    )
                ),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_CONFUSION_MATRIX,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # ========================================================
    # 14. TERMINAL REPORT
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "SVM CLOSED-SET TEST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy        : "
        f"{accuracy:.6f}"
    )

    print(
        f"Macro Precision : "
        f"{macro_precision:.6f}"
    )

    print(
        f"Macro Recall    : "
        f"{macro_recall:.6f}"
    )

    print(
        f"Macro F1        : "
        f"{macro_f1:.6f}"
    )

    print()

    print(
        "Per-speaker accuracy:"
    )

    for (
        speaker,
        result
    ) in per_speaker.items():

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
        "Unknown threshold applied: "
        "FALSE"
    )

    print()

    print(
        "Artifacts:"
    )

    print(
        f"  Predictions : "
        f"{OUTPUT_PREDICTIONS}"
    )

    print(
        f"  Metrics     : "
        f"{OUTPUT_METRICS}"
    )

    print(
        f"  Confusion   : "
        f"{OUTPUT_CONFUSION_MATRIX}"
    )

    print(
        "=" * 60
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )