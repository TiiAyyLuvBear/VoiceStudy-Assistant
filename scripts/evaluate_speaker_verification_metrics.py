from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = ROOT / "experiments/test/speaker_disjoint_verification_test_predictions.csv"
THRESHOLD_CONFIG = ROOT / "models/experimental/verification_threshold.json"

FIXED_METRICS = ROOT / "experiments/test/speaker_disjoint_verification_fixed_threshold_metrics.json"
CURVE_METRICS = ROOT / "experiments/test/speaker_disjoint_verification_curve_metrics.json"
ROC_PLOT = ROOT / "experiments/test/sv_roc_curve.png"


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    print("# WEEK 3 — SPEAKER VERIFICATION METRICS\n")

    rows = read_csv(PREDICTIONS)
    config = json.loads(THRESHOLD_CONFIG.read_text(encoding="utf-8"))

    threshold = float(config["threshold"])

    if not rows:
        raise ValueError("Verification predictions are empty.")

    scores = np.asarray(
        [float(r["cosine_similarity"]) for r in rows],
        dtype=float,
    )

    trial_types = np.asarray(
        [r["trial_type"] for r in rows],
        dtype=str,
    )

    genuine_mask = trial_types == "GENUINE"
    non_genuine_mask = ~genuine_mask

    verified = scores >= threshold

    tp = int(np.sum(genuine_mask & verified))
    fn = int(np.sum(genuine_mask & ~verified))
    fp = int(np.sum(non_genuine_mask & verified))
    tn = int(np.sum(non_genuine_mask & ~verified))

    genuine_count = int(np.sum(genuine_mask))
    non_genuine_count = int(np.sum(non_genuine_mask))

    far = fp / non_genuine_count if non_genuine_count else 0.0
    frr = fn / genuine_count if genuine_count else 0.0
    accuracy = (tp + tn) / len(rows)

    y_true_binary = genuine_mask.astype(int)
    y_pred_binary = verified.astype(int)

    f1 = f1_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0,
    )

    fixed_metrics = {
        "protocol": "WEEK3_SPEAKER_VERIFICATION_FIXED_THRESHOLD",
        "threshold": threshold,
        "threshold_source": str(THRESHOLD_CONFIG),
        "threshold_tuned_on_test": False,
        "test_trial_count": len(rows),
        "genuine_count": genuine_count,
        "non_genuine_count": non_genuine_count,
        "true_accept": tp,
        "false_reject": fn,
        "false_accept": fp,
        "true_reject": tn,
        "metrics": {
            "FAR": float(far),
            "FRR": float(frr),
            "verification_accuracy": float(accuracy),
            "F1": float(f1),
        },
        "note": "EER threshold is not used to modify the system.",
    }

    FIXED_METRICS.parent.mkdir(parents=True, exist_ok=True)
    FIXED_METRICS.write_text(
        json.dumps(
            fixed_metrics,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # ROC / AUC
    # ---------------------------------------------------------

    fpr, tpr, roc_thresholds = roc_curve(
        y_true_binary,
        scores,
    )

    roc_auc = float(
        roc_auc_score(
            y_true_binary,
            scores,
        )
    )

    fnr = 1.0 - tpr

    eer_index = int(
        np.argmin(
            np.abs(
                fpr - fnr
            )
        )
    )

    eer_threshold = float(
        roc_thresholds[eer_index]
    )

    eer_far = float(
        fpr[eer_index]
    )

    eer_frr = float(
        fnr[eer_index]
    )

    eer = float(
        (eer_far + eer_frr) / 2.0
    )

    curve_metrics = {
        "protocol": "WEEK3_SPEAKER_VERIFICATION_CURVE",
        "test_trial_count": len(rows),
        "genuine_count": genuine_count,
        "non_genuine_count": non_genuine_count,
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "eer_far": eer_far,
        "eer_frr": eer_frr,
        "eer_threshold_used_for_system": False,
        "system_threshold": threshold,
        "note": "EER is descriptive only. The frozen verification threshold remains unchanged.",
    }

    CURVE_METRICS.parent.mkdir(parents=True, exist_ok=True)
    CURVE_METRICS.write_text(
        json.dumps(
            curve_metrics,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # ROC curve
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Speaker Verification ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ROC_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        ROC_PLOT,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print("FIXED THRESHOLD RESULTS")
    print("-" * 60)
    print(f"Threshold            : {threshold:.8f}")
    print(f"Genuine              : {genuine_count}")
    print(f"Non-genuine          : {non_genuine_count}")
    print(f"True Accept          : {tp}")
    print(f"False Reject         : {fn}")
    print(f"False Accept         : {fp}")
    print(f"True Reject          : {tn}")
    print(f"FAR                  : {far:.6f}")
    print(f"FRR                  : {frr:.6f}")
    print(f"Verification Accuracy: {accuracy:.6f}")
    print(f"F1                   : {f1:.6f}")

    print()
    print("CURVE RESULTS")
    print("-" * 60)
    print(f"ROC-AUC              : {roc_auc:.6f}")
    print(f"EER                  : {eer:.6f}")
    print(f"EER threshold        : {eer_threshold:.8f}")
    print(f"EER FAR              : {eer_far:.6f}")
    print(f"EER FRR              : {eer_frr:.6f}")
    print("EER threshold used   : NO")

    print()
    print("Artifacts:")
    print(FIXED_METRICS)
    print(CURVE_METRICS)
    print(ROC_PLOT)

    print("\nWEEK 3 — STEP 9 COMPLETE")


if __name__ == "__main__":
    main()