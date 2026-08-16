"""Re-score the frozen Whisper Small v2 predictions against the identical v3 test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.asr.metrics import calculate_corpus_error_rates, calculate_error_rates
from src.asr.text_normalizer import normalize_asr_text
from src.utils import canonical_csv_sha256, sha256_file


PREDICTION_FIELDS = (
    "audio_id",
    "audio_path",
    "reference_transcript",
    "hypothesis",
    "normalized_reference",
    "normalized_hypothesis",
    "wer",
    "cer",
    "model",
    "language",
    "success",
    "source_prediction_file",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare_baseline(
    *,
    test_path: Path,
    source_predictions_path: Path,
    source_config_path: Path,
    model_binary_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    test_rows = _read(test_path)
    source_rows = _read(source_predictions_path)
    by_id = {row["audio_id"]: row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("Duplicate audio_id in source predictions")
    if set(by_id) != {row["audio_id"] for row in test_rows}:
        raise ValueError("v2 prediction IDs do not exactly match the v3 test IDs")

    output_rows: list[dict[str, str]] = []
    for test in test_rows:
        source = by_id[test["audio_id"]]
        if source.get("success", "").casefold() != "true":
            raise ValueError(f"Source prediction failed for {test['audio_id']}")
        if Path(source["audio_path"]).resolve() != Path(test["audio_path"]).resolve():
            raise ValueError(f"Audio path mismatch for {test['audio_id']}")
        hypothesis = source["hypothesis"]
        reference = test["transcript"]
        rates = calculate_error_rates(reference, hypothesis)
        output_rows.append(
            {
                "audio_id": test["audio_id"],
                "audio_path": test["audio_path"],
                "reference_transcript": reference,
                "hypothesis": hypothesis,
                "normalized_reference": normalize_asr_text(reference),
                "normalized_hypothesis": normalize_asr_text(hypothesis),
                "wer": f"{rates['wer']:.6f}",
                "cer": f"{rates['cer']:.6f}",
                "model": source["model"],
                "language": source["language"],
                "success": "true",
                "source_prediction_file": source_predictions_path.as_posix(),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "baseline_original_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    corpus = calculate_corpus_error_rates(
        [row["reference_transcript"] for row in output_rows],
        [row["hypothesis"] for row in output_rows],
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "FROZEN_BASELINE",
        "dataset_version": "asr-v3",
        "model": "whisper-small-original",
        "inference_reused": True,
        "reuse_reason": "v2 and v3 test contain exactly the same 249 audio IDs and paths",
        "test_used_for_finetune_decisions": False,
        "sample_count": len(output_rows),
        **corpus,
        "artifacts": {
            "test": {
                "path": test_path.as_posix(),
                "sha256": sha256_file(test_path),
                "canonical_csv_sha256": canonical_csv_sha256(test_path),
            },
            "source_predictions": {
                "path": source_predictions_path.as_posix(),
                "sha256": sha256_file(source_predictions_path),
            },
            "source_config": {
                "path": source_config_path.as_posix(),
                "sha256": sha256_file(source_config_path),
            },
            "ctranslate2_model": {
                "path": model_binary_path.as_posix(),
                "sha256": sha256_file(model_binary_path),
            },
            "predictions": {
                "path": predictions_path.as_posix(),
                "sha256": sha256_file(predictions_path),
                "canonical_csv_sha256": canonical_csv_sha256(predictions_path),
            },
        },
    }
    summary_path = output_dir / "baseline_original_metrics.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/v3/metadata/asr_finetune_test.csv"),
    )
    parser.add_argument(
        "--source-predictions",
        type=Path,
        default=Path("reports/asr/v2/asr_test_predictions.csv"),
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("reports/asr/v2/asr_test_config.json"),
    )
    parser.add_argument(
        "--model-binary",
        type=Path,
        default=Path(
            "models/cache/whisper/models--Systran--faster-whisper-small/"
            "snapshots/536b0662742c02347bc0e980a01041f333bce120/model.bin"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/asr/v3/baseline_original")
    )
    args = parser.parse_args()
    summary = prepare_baseline(
        test_path=args.test,
        source_predictions_path=args.source_predictions,
        source_config_path=args.source_config,
        model_binary_path=args.model_binary,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
