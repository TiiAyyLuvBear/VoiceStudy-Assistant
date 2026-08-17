"""Check that all experimental Speaker v2 artifacts load successfully."""

from __future__ import annotations

from pathlib import Path

from scripts.check_speaker_packaging import check_centroids, check_file


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/experimental/v2"
OUTPUT = ROOT / "experiments/v2/test/speaker_packaging_check.md"


def main() -> int:
    targets = [
        ("speaker_svm_linear.pkl", MODEL_DIR / "speaker_svm_linear.pkl", check_file),
        (
            "svm_closed_set_centroids",
            MODEL_DIR / "svm_closed_set_centroids",
            check_centroids,
        ),
        (
            "cosine_test_centroids",
            MODEL_DIR / "cosine_test_centroids",
            check_centroids,
        ),
        (
            "cosine_unknown_threshold.json",
            MODEL_DIR / "cosine_unknown_threshold.json",
            check_file,
        ),
        (
            "verification_threshold.json",
            MODEL_DIR / "verification_threshold.json",
            check_file,
        ),
    ]
    results = []
    for name, path, checker in targets:
        ok, detail = checker(path)
        results.append((name, path, ok, detail))
    all_ok = all(item[2] for item in results)
    lines = [
        "# Speaker v2 packaging check",
        "",
        *[
            f"- **{name}** - **{'PASS' if ok else 'FAIL'}** - {detail} "
            f"(`{path.relative_to(ROOT)}`)"
            for name, path, ok, detail in results
        ],
        "",
        f"Final status: **{'PASS' if all_ok else 'FAIL'}**",
        "",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
