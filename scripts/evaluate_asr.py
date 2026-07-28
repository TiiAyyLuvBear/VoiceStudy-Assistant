"""Chạy Whisper trên ASR split và tính WER, CER, latency có checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from src.asr.metrics import calculate_corpus_error_rates, calculate_error_rates
from src.asr.whisper_model import get_asr_model
from src.nlu.text_normalizer import normalize_text


PREDICTION_FIELDS = (
    "audio_id",
    "audio_path",
    "original_split",
    "reference_transcript",
    "hypothesis",
    "normalized_reference",
    "normalized_hypothesis",
    "wer",
    "cer",
    "latency_ms",
    "model",
    "language",
    "success",
    "error",
)


def _column(fieldnames: list[str], *aliases: str) -> str:
    lookup = {name.lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    raise ValueError(f"Missing column; expected one of: {', '.join(aliases)}")


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_previous(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["audio_id"]: row for row in csv.DictReader(stream)}


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_file", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/asr"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if not args.split_file.is_file():
        raise FileNotFoundError(f"ASR split does not exist: {args.split_file}")
    with args.split_file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        id_column = _column(fields, "audio_id", "id", "file_id")
        path_column = _column(fields, "audio_path", "file_path", "path")
        text_column = _column(fields, "reference_transcript", "transcript", "text")
        split_column = next(
            (name for name in ("original_split", "split") if name in fields), None
        )
        source_rows = list(reader)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        source_rows = source_rows[: args.limit]

    prediction_path = args.output_dir / f"{args.split_file.stem}_predictions.csv"
    summary_path = args.output_dir / f"{args.split_file.stem}_summary.json"
    previous = _load_previous(prediction_path) if args.resume else {}
    asr = get_asr_model(args.config)
    predictions: list[dict[str, str]] = []

    for index, source in enumerate(source_rows, start=1):
        audio_id = source[id_column]
        cached = previous.get(audio_id)
        if cached and cached.get("success", "").lower() == "true":
            predictions.append(cached)
            print(f"[{index}/{len(source_rows)}] resume {audio_id}")
            continue

        reference = source[text_column]
        result = asr.transcribe(source[path_column])
        hypothesis = result["transcript"]
        rates = calculate_error_rates(reference, hypothesis)
        prediction = {
            "audio_id": audio_id,
            "audio_path": source[path_column],
            "original_split": source[split_column] if split_column else "",
            "reference_transcript": reference,
            "hypothesis": hypothesis,
            "normalized_reference": normalize_text(reference),
            "normalized_hypothesis": normalize_text(hypothesis),
            "wer": f"{rates['wer']:.6f}",
            "cer": f"{rates['cer']:.6f}",
            "latency_ms": f"{result['latency_ms']:.3f}",
            "model": result["model"],
            "language": result["language"],
            "success": str(result["success"]).lower(),
            "error": result["error"] or "",
        }
        predictions.append(prediction)
        _write_predictions(prediction_path, predictions)
        print(
            f"[{index}/{len(source_rows)}] {audio_id}: "
            f"success={result['success']} latency={result['latency_ms']:.1f}ms"
        )

    _write_predictions(prediction_path, predictions)
    references = [row["reference_transcript"] for row in predictions]
    hypotheses = [row["hypothesis"] for row in predictions]
    corpus = calculate_corpus_error_rates(references, hypotheses)
    latencies = [float(row["latency_ms"]) for row in predictions]
    failures = sum(row["success"].lower() != "true" for row in predictions)
    summary = {
        "split_file": str(args.split_file),
        "prediction_file": str(prediction_path),
        "sample_count": len(predictions),
        "successful_count": len(predictions) - failures,
        "failure_count": failures,
        "wer": corpus["wer"],
        "cer": corpus["cer"],
        "word_edits": corpus["word_edits"],
        "reference_words": corpus["reference_words"],
        "char_edits": corpus["char_edits"],
        "reference_chars": corpus["reference_chars"],
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile_95(latencies),
        "model": predictions[0]["model"] if predictions else "",
        "language": predictions[0]["language"] if predictions else "vi",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
