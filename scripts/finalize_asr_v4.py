"""Compare original, LoRA v3 and wider LoRA v4, then freeze ASR v4."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.utils import canonical_csv_sha256, sha256_file


def _change(reference: float, candidate: float) -> dict[str, float]:
    if reference <= 0:
        raise ValueError("Reference metric must be positive")
    return {
        "reference": reference,
        "candidate": candidate,
        "absolute_change": candidate - reference,
        "percentage_point_change": (candidate - reference) * 100,
        "relative_improvement": (reference - candidate) / reference,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("reports/asr/v3/baseline_original/baseline_original_metrics.json"),
    )
    parser.add_argument(
        "--v3", type=Path, default=Path("reports/asr/v3/final_test/test_metrics.json")
    )
    parser.add_argument(
        "--v4", type=Path, default=Path("reports/asr/v4/final_test/test_metrics.json")
    )
    parser.add_argument(
        "--lock", type=Path, default=Path("models/experimental/asr/v4/locked_model.json")
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/processed/v4/asr_finetune_manifest.json"),
    )
    parser.add_argument(
        "--test-split",
        type=Path,
        default=Path("data/processed/v4/metadata/asr_finetune_test.csv"),
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("models/experimental/asr/v4/training_summary.json"),
    )
    parser.add_argument(
        "--comparison", type=Path, default=Path("reports/asr/v4/comparison.json")
    )
    parser.add_argument(
        "--final-manifest",
        type=Path,
        default=Path("models/experimental/asr/v4/final_manifest.json"),
    )
    args = parser.parse_args()

    if args.comparison.exists() or args.final_manifest.exists():
        raise FileExistsError("ASR v4 has already been finalized")
    original = json.loads(args.original.read_text(encoding="utf-8"))
    v3 = json.loads(args.v3.read_text(encoding="utf-8"))
    v4 = json.loads(args.v4.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    training = json.loads(args.training_summary.read_text(encoding="utf-8"))

    expected_split = canonical_csv_sha256(args.test_split)
    split_hashes = {
        original["artifacts"]["test"]["canonical_csv_sha256"],
        v3["split_canonical_csv_sha256"],
        v4["split_canonical_csv_sha256"],
        expected_split,
    }
    if len(split_hashes) != 1:
        raise ValueError("The three model results do not use the same test split")
    if {original["sample_count"], v3["sample_count"], v4["sample_count"]} != {249}:
        raise ValueError("Expected 249 test samples for every model")
    if v3.get("failure_count") or v4.get("failure_count"):
        raise ValueError("Fine-tuned evaluation contains failed predictions")
    if lock.get("status") != "LOCKED_BEFORE_TEST":
        raise ValueError("V4 model was not locked before test")
    if v4.get("lock_sha256") != sha256_file(args.lock):
        raise ValueError("V4 result does not reference the current model lock")
    if training.get("test_split_accessed") is not False:
        raise ValueError("Training summary reports test access")

    models = [
        {
            "label": "Whisper Small original (model used by ASR v2; no fine-tune)",
            "model_version": original["model"],
            "wer": original["wer"], "cer": original["cer"],
            "word_edits": original["word_edits"], "char_edits": original["char_edits"],
        },
        {
            "label": "Whisper Small LoRA v3 (r8, q_proj/v_proj)",
            "model_version": v3["model_version"],
            "wer": v3["wer"], "cer": v3["cer"],
            "word_edits": v3["word_edits"], "char_edits": v3["char_edits"],
        },
        {
            "label": "Whisper Small wider LoRA v4 (r32, attention + fc1/fc2)",
            "model_version": v4["model_version"],
            "wer": v4["wer"], "cer": v4["cer"],
            "word_edits": v4["word_edits"], "char_edits": v4["char_edits"],
        },
    ]
    comparison = {
        "schema_version": 1,
        "dataset_version": "asr-v4",
        "sample_count": 249,
        "test_canonical_csv_sha256": expected_split,
        "models": models,
        "v4_vs_original": {
            "wer": _change(float(original["wer"]), float(v4["wer"])),
            "cer": _change(float(original["cer"]), float(v4["cer"])),
        },
        "v4_vs_lora_v3": {
            "wer": _change(float(v3["wer"]), float(v4["wer"])),
            "cer": _change(float(v3["cer"]), float(v4["cer"])),
        },
        "protocol": {
            "same_test_audio": True,
            "same_search_settings": True,
            "same_runtime_precision": False,
            "runtime_note": (
                "Original and LoRA v3 reference metrics used CPU int8; the Colab v4 "
                "test uses CUDA float16. Beam/VAD/search settings remain the same."
            ),
            "v4_test_used_for_training_or_checkpoint_selection": False,
            "v4_model_test_runs": 1,
            "test_is_fresh_holdout": False,
            "interpretation": (
                "Controlled comparison on the frozen v3 test reused by v4. Since earlier "
                "v3 results were already known, this is not a new unbiased holdout estimate."
            ),
        },
    }
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    args.comparison.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    frozen_at = datetime.now().astimezone().isoformat(timespec="seconds")
    dataset["freeze_status"] = "FROZEN"
    dataset["frozen_at"] = frozen_at
    dataset["final_evaluation"] = {
        "locked_model": args.lock.as_posix(),
        "locked_model_sha256": sha256_file(args.lock),
        "v4_test_metrics": args.v4.as_posix(),
        "v4_test_metrics_sha256": sha256_file(args.v4),
        "comparison": args.comparison.as_posix(),
        "comparison_sha256": sha256_file(args.comparison),
        "v4_model_test_runs": 1,
    }
    args.dataset_manifest.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    final_manifest = {
        "schema_version": 1,
        "dataset_version": "asr-v4",
        "model_version": v4["model_version"],
        "status": "FROZEN_COMPARISON_COMPLETED",
        "frozen_at": frozen_at,
        "selection": training["selection"],
        "artifacts": {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in (
                ("dataset_manifest", args.dataset_manifest),
                ("training_summary", args.training_summary),
                ("locked_model", args.lock),
                ("original_metrics", args.original),
                ("v3_metrics", args.v3),
                ("v4_metrics", args.v4),
                ("comparison", args.comparison),
            )
        },
        "test_protocol": comparison["protocol"],
    }
    args.final_manifest.write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
