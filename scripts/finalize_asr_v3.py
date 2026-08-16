"""Finalize ASR v3 after its one-time locked test and freeze provenance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.utils import sha256_file


def metric_comparison(baseline: float, fine_tuned: float) -> dict[str, float]:
    if baseline <= 0:
        raise ValueError("Baseline metric must be positive")
    absolute = fine_tuned - baseline
    return {
        "baseline": baseline,
        "fine_tuned": fine_tuned,
        "absolute_change": absolute,
        "absolute_percentage_point_change": absolute * 100,
        "relative_change": absolute / baseline,
        "relative_improvement": (baseline - fine_tuned) / baseline,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("reports/asr/v3/baseline_original/baseline_original_metrics.json"))
    parser.add_argument("--final-test", type=Path, default=Path("reports/asr/v3/final_test/test_metrics.json"))
    parser.add_argument("--lock", type=Path, default=Path("models/experimental/asr/v3/locked_model.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/processed/v3/asr_finetune_manifest.json"))
    parser.add_argument("--training-summary", type=Path, default=Path("models/experimental/asr/v3/training_summary.json"))
    parser.add_argument("--comparison", type=Path, default=Path("reports/asr/v3/comparison.json"))
    parser.add_argument("--final-manifest", type=Path, default=Path("models/experimental/asr/v3/final_manifest.json"))
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()

    if (args.comparison.exists() or args.final_manifest.exists()) and not args.repair:
        raise FileExistsError("ASR v3 has already been finalized")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    final = json.loads(args.final_test.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    dataset_text = args.dataset_manifest.read_text(encoding="utf-8")
    dataset = json.loads(dataset_text.removesuffix("\\n"))
    training = json.loads(args.training_summary.read_text(encoding="utf-8"))

    if baseline.get("status") != "FROZEN_BASELINE":
        raise ValueError("Baseline is not frozen")
    if lock.get("status") != "LOCKED_BEFORE_FINAL_TEST":
        raise ValueError("Model was not locked before final test")
    if final.get("role") != "test" or final.get("sample_count") != 249:
        raise ValueError("Final test is incomplete or has the wrong role")
    if final.get("failure_count") != 0:
        raise ValueError("Final test contains failed predictions")
    if final.get("lock_sha256") != sha256_file(args.lock):
        raise ValueError("Final test does not reference the current locked model")
    if training.get("test_split_accessed") is not False:
        raise ValueError("Training summary reports test access")

    comparison = {
        "schema_version": 1,
        "dataset_version": "asr-v3",
        "baseline_model": baseline["model"],
        "fine_tuned_model": final["model_version"],
        "sample_count": final["sample_count"],
        "wer": metric_comparison(float(baseline["wer"]), float(final["wer"])),
        "cer": metric_comparison(float(baseline["cer"]), float(final["cer"])),
        "word_edits": {
            "baseline": baseline["word_edits"],
            "fine_tuned": final["word_edits"],
            "reduction": baseline["word_edits"] - final["word_edits"],
        },
        "char_edits": {
            "baseline": baseline["char_edits"],
            "fine_tuned": final["char_edits"],
            "reduction": baseline["char_edits"] - final["char_edits"],
        },
        "protocol": {
            "same_test_audio": True,
            "same_reference_count": True,
            "same_faster_whisper_decode_settings": True,
            "test_used_for_training_or_checkpoint_selection": False,
            "final_test_runs": 1,
        },
    }
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    args.comparison.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset["freeze_status"] = "FROZEN"
    dataset["frozen_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    dataset["final_evaluation"] = {
        "locked_model": args.lock.as_posix(),
        "locked_model_sha256": sha256_file(args.lock),
        "baseline_metrics": args.baseline.as_posix(),
        "baseline_metrics_sha256": sha256_file(args.baseline),
        "final_test_metrics": args.final_test.as_posix(),
        "final_test_metrics_sha256": sha256_file(args.final_test),
        "comparison": args.comparison.as_posix(),
        "comparison_sha256": sha256_file(args.comparison),
        "final_test_runs": 1,
    }
    args.dataset_manifest.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    final_manifest = {
        "schema_version": 1,
        "dataset_version": "asr-v3",
        "model_version": final["model_version"],
        "status": "FROZEN_FINAL_TEST_COMPLETED",
        "frozen_at": dataset["frozen_at"],
        "best_epoch": training["selection"]["best_epoch"],
        "best_validation_loss": training["selection"]["best_validation_loss"],
        "artifacts": {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in (
                ("dataset_manifest", args.dataset_manifest),
                ("training_summary", args.training_summary),
                ("locked_model", args.lock),
                ("baseline_metrics", args.baseline),
                ("final_test_metrics", args.final_test),
                ("comparison", args.comparison),
            )
        },
        "test_protocol": comparison["protocol"],
    }
    args.final_manifest.write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
