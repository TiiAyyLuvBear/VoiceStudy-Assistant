from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

TRIALS = ROOT / "experiments/test/speaker_disjoint_verification_test_trials.csv"
THRESHOLD_CONFIG = ROOT / "models/experimental/verification_threshold.json"
OUTPUT = ROOT / "experiments/test/speaker_disjoint_verification_test_predictions.csv"


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    print("# WEEK 3 — SPEAKER VERIFICATION TEST\n")

    trials = read_csv(TRIALS)

    if not trials:
        raise ValueError("Verification trial file is empty.")

    config = json.loads(
        THRESHOLD_CONFIG.read_text(encoding="utf-8")
    )

    threshold = float(config["threshold"])

    print(f"Test trials       : {len(trials)}")
    print(f"Verification threshold: {threshold:.8f}")
    print("Threshold tuned on test: NO")
    print()

    predictions = []

    for row in trials:
        score = float(row["cosine_similarity"])
        trial_type = row["trial_type"]

        verified = score >= threshold

        if trial_type == "GENUINE":
            correct = verified
        else:
            correct = not verified

        if trial_type == "GENUINE":
            result = "TRUE_ACCEPT" if verified else "FALSE_REJECT"
        else:
            result = "FALSE_ACCEPT" if verified else "TRUE_REJECT"

        predictions.append({
            **row,
            "threshold": f"{threshold:.8f}",
            "verified": str(verified).lower(),
            "result": result,
            "correct": str(correct).lower(),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fields = list(predictions[0].keys())

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)

    genuine = [r for r in predictions if r["trial_type"] == "GENUINE"]
    impostor = [r for r in predictions if r["trial_type"] == "IMPOSTOR"]
    unknown = [r for r in predictions if r["trial_type"] == "UNKNOWN_IMPOSTOR"]

    true_accept = sum(r["result"] == "TRUE_ACCEPT" for r in genuine)
    false_reject = sum(r["result"] == "FALSE_REJECT" for r in genuine)

    false_accept_impostor = sum(
        r["result"] == "FALSE_ACCEPT" for r in impostor
    )
    true_reject_impostor = sum(
        r["result"] == "TRUE_REJECT" for r in impostor
    )

    false_accept_unknown = sum(
        r["result"] == "FALSE_ACCEPT" for r in unknown
    )
    true_reject_unknown = sum(
        r["result"] == "TRUE_REJECT" for r in unknown
    )

    correct = sum(r["correct"] == "true" for r in predictions)
    accuracy = correct / len(predictions)

    print("RESULTS")
    print("-" * 60)

    print(f"Genuine trials          : {len(genuine)}")
    print(f"  True accept           : {true_accept}")
    print(f"  False reject          : {false_reject}")

    print()

    print(f"Impostor trials         : {len(impostor)}")
    print(f"  True reject           : {true_reject_impostor}")
    print(f"  False accept          : {false_accept_impostor}")

    print()

    print(f"Unknown impostor trials : {len(unknown)}")
    print(f"  True reject           : {true_reject_unknown}")
    print(f"  False accept          : {false_accept_unknown}")

    print()

    print(f"Correct trials          : {correct}/{len(predictions)}")
    print(f"Verification accuracy   : {accuracy:.6f}")

    print()
    print("Artifacts:")
    print(OUTPUT)

    print("\nWEEK 3 — STEP 8 COMPLETE")


if __name__ == "__main__":
    main()