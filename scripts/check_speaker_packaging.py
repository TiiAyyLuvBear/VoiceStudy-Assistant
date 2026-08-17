from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SVM = ROOT / "models/experimental/speaker_svm_linear.pkl"
SVM_CENTROIDS = ROOT / "models/experimental/svm_closed_set_centroids"
UNKNOWN_THRESHOLD = ROOT / "models/experimental/cosine_unknown_threshold.json"
VERIFICATION_THRESHOLD = ROOT / "models/experimental/verification_threshold.json"
USER_EMBEDDINGS = ROOT / "models/application/user_embeddings"
APPLICATION_THRESHOLD = ROOT / "models/application/application_sid_threshold.json"
OUTPUT = ROOT / "experiments/test/speaker_packaging_check.md"


def check_file(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "MISSING"

    try:
        if path.suffix == ".pkl":
            obj = joblib.load(path)
            return True, f"LOAD OK ({type(obj).__name__})"

        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return True, f"LOAD OK ({type(data).__name__})"

        return True, "EXISTS"

    except Exception as exc:
        return False, f"LOAD FAILED: {exc}"


def check_centroids(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "DIRECTORY MISSING"

    files = sorted(path.glob("*.npy"))
    if not files:
        return False, "NO .npy FILES"

    errors = []
    for file in files:
        try:
            x = np.asarray(np.load(file, allow_pickle=False), dtype=np.float32)
            if x.ndim != 1 or not np.isfinite(x).all():
                errors.append(f"{file.name}: invalid embedding")
            elif x.size != 192:
                errors.append(f"{file.name}: dimension={x.size}, expected=192")
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")

    if errors:
        return False, "; ".join(errors)

    return True, f"LOAD OK ({len(files)} centroids, 192-D)"


def check_embeddings(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "DIRECTORY MISSING"

    files = sorted(path.glob("*.npy"))
    if not files:
        return False, "NO .npy FILES"

    errors = []
    for file in files:
        try:
            x = np.asarray(np.load(file, allow_pickle=False), dtype=np.float32)
            if x.ndim != 1 or not np.isfinite(x).all():
                errors.append(f"{file.name}: invalid embedding")
            elif x.size != 192:
                errors.append(f"{file.name}: dimension={x.size}, expected=192")
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")

    if errors:
        return False, "; ".join(errors)

    return True, f"LOAD OK ({len(files)} embeddings, 192-D)"


def main() -> int:
    print("# WEEK 3 — SPEAKER PACKAGING CHECK")
    print()

    results = []

    ok, detail = check_file(SVM)
    results.append(("speaker_svm_linear.pkl", SVM, ok, detail))

    ok, detail = check_centroids(SVM_CENTROIDS)
    results.append(("svm_closed_set_centroids", SVM_CENTROIDS, ok, detail))

    ok, detail = check_file(UNKNOWN_THRESHOLD)
    results.append(("cosine_unknown_threshold.json", UNKNOWN_THRESHOLD, ok, detail))

    ok, detail = check_file(VERIFICATION_THRESHOLD)
    results.append(("verification_threshold.json", VERIFICATION_THRESHOLD, ok, detail))

    ok, detail = check_embeddings(USER_EMBEDDINGS)
    results.append(("user_embeddings", USER_EMBEDDINGS, ok, detail))

    ok, detail = check_file(APPLICATION_THRESHOLD)
    results.append(("application_sid_threshold.json", APPLICATION_THRESHOLD, ok, detail))

    lines = [
        "# WEEK 3 — SPEAKER PACKAGING CHECK",
        "",
        "## Summary",
        "",
    ]

    all_ok = True

    for name, path, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(f"{status:<6} {name:<40} {detail}")
        lines.append(f"- **{name}** — `{status}` — {detail}")
        lines.append(f"  - Path: `{path.relative_to(ROOT)}`")

    lines += [
        "",
        "## Final Status",
        "",
        f"**{'PASS' if all_ok else 'FAIL'}**",
        "",
        "Packaging check only; no model retraining or test-threshold tuning performed.",
        "",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"Report: {OUTPUT}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())