"""Evaluate the locked ASR v3 CTranslate2 model on validation or final test."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel

from src.asr.metrics import calculate_corpus_error_rates, calculate_error_rates
from src.asr.text_normalizer import normalize_asr_text
from src.utils import canonical_csv_sha256, sha256_file


FIELDS = (
    "audio_id",
    "audio_path",
    "reference_transcript",
    "hypothesis",
    "normalized_reference",
    "normalized_hypothesis",
    "wer",
    "cer",
    "latency_ms",
    "success",
    "error",
)


def _verify_lock(lock_path: Path, model_dir: Path, split_path: Path, role: str) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_FINAL_TEST":
        raise ValueError("Model is not locked before final test")
    if Path(lock["model_directory"]).resolve() != model_dir.resolve():
        raise ValueError("Model directory differs from lock")
    for relative, expected in lock["model_files"].items():
        if sha256_file(model_dir / relative) != expected:
            raise ValueError(f"Locked model file changed: {relative}")
    expected_split = lock["datasets"][role]["canonical_csv_sha256"]
    if canonical_csv_sha256(split_path) != expected_split:
        raise ValueError(f"{role} split differs from lock")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("validation", "test"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/experimental/asr/v3/ctranslate2"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("models/experimental/asr/v3/locked_model.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()

    if args.role == "test" and not args.confirm_final_test:
        parser.error("Final test requires --confirm-final-test")
    predictions_path = args.output_dir / f"{args.role}_predictions.csv"
    metrics_path = args.output_dir / f"{args.role}_metrics.json"
    if predictions_path.exists() or metrics_path.exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation")
    lock = _verify_lock(args.lock, args.model_dir, args.split, args.role)

    with args.split.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    model = WhisperModel(
        str(args.model_dir),
        device="cpu",
        compute_type="int8",
        cpu_threads=0,
        num_workers=1,
        local_files_only=True,
    )
    predictions: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        try:
            segments, _ = model.transcribe(
                row["audio_path"],
                language="vi",
                task="transcribe",
                beam_size=10,
                vad_filter=True,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            hypothesis = " ".join(
                segment.text.strip() for segment in segments if segment.text.strip()
            )
            success = bool(hypothesis)
            error = "" if success else "empty_transcript"
        except Exception as exc:
            hypothesis = ""
            success = False
            error = str(exc)
        rates = calculate_error_rates(row["transcript"], hypothesis)
        predictions.append(
            {
                "audio_id": row["audio_id"],
                "audio_path": row["audio_path"],
                "reference_transcript": row["transcript"],
                "hypothesis": hypothesis,
                "normalized_reference": normalize_asr_text(row["transcript"]),
                "normalized_hypothesis": normalize_asr_text(hypothesis),
                "wer": f"{rates['wer']:.6f}",
                "cer": f"{rates['cer']:.6f}",
                "latency_ms": f"{(time.perf_counter() - started) * 1000:.3f}",
                "success": str(success).lower(),
                "error": error,
            }
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(predictions)
        print(f"[{index}/{len(rows)}] {row['audio_id']} success={success}", flush=True)

    corpus = calculate_corpus_error_rates(
        [row["reference_transcript"] for row in predictions],
        [row["hypothesis"] for row in predictions],
    )
    latencies = [float(row["latency_ms"]) for row in predictions]
    metrics = {
        "schema_version": 1,
        "dataset_version": "asr-v3",
        "model_version": lock["model_version"],
        "role": args.role,
        "sample_count": len(predictions),
        "successful_count": sum(row["success"] == "true" for row in predictions),
        "failure_count": sum(row["success"] != "true" for row in predictions),
        **corpus,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "lock_sha256": sha256_file(args.lock),
        "split_canonical_csv_sha256": canonical_csv_sha256(args.split),
        "predictions_sha256": sha256_file(predictions_path),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 1 if metrics["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
