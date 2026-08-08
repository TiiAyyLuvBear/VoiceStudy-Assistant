"""Run the frozen Week 3 command text and audio-to-NLU test protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.week3_test_config import APPLICATION_LABELS, sha256_file, verify_snapshot
from src.asr.metrics import calculate_error_rates
from src.nlu.command_parser import parse_command
from src.pipeline.asr_nlu import ASRNLUPipelineResult, run_asr_nlu_pipeline


TEXT_FIELDS = (
    "command_id",
    "transcript",
    "reference_date",
    "expected_intent",
    "predicted_intent",
    "intent_correct",
    "expected_title",
    "predicted_title",
    "title_exact_match",
    "expected_date",
    "predicted_date",
    "date_exact_match",
    "expected_time",
    "predicted_time",
    "time_exact_match",
    "entity_exact_match",
    "missing_fields",
)

AUDIO_FIELDS = (
    "recording_id",
    "command_id",
    "speaker_id",
    "audio_path",
    "manifest_status",
    "expected_transcript",
    "whisper_transcript",
    "expected_intent",
    "predicted_intent",
    "intent_correct",
    "expected_title",
    "predicted_title",
    "title_exact_match",
    "expected_date",
    "predicted_date",
    "date_exact_match",
    "expected_time",
    "predicted_time",
    "time_exact_match",
    "entity_exact_match",
    "missing_fields",
    "asr_wer",
    "asr_cer",
    "latency_ms",
    "model",
    "language",
    "evaluated",
    "asr_success",
    "error",
    "test_config_sha256",
)

OOS_FIELDS = (
    "command_id",
    "transcript",
    "expected_intent",
    "text_predicted_intent",
    "text_correctly_rejected",
    "audio_available",
    "whisper_transcript",
    "whisper_predicted_intent",
    "whisper_correctly_rejected",
    "error",
)

ENTITY_FIELDS = (
    "command_id",
    "expected_intent",
    "entity_bearing_command",
    "expected_title",
    "text_predicted_title",
    "text_title_exact_match",
    "whisper_predicted_title",
    "whisper_title_exact_match",
    "expected_date",
    "text_predicted_date",
    "text_date_exact_match",
    "whisper_predicted_date",
    "whisper_date_exact_match",
    "expected_time",
    "text_predicted_time",
    "text_time_exact_match",
    "whisper_predicted_time",
    "whisper_time_exact_match",
    "text_entity_exact_match",
    "whisper_entity_exact_match",
    "audio_evaluated",
)

PipelineRunner = Callable[..., ASRNLUPipelineResult]


def _bool(value: bool) -> str:
    return str(value).lower()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _expected_entities(command: dict[str, str]) -> dict[str, str]:
    return {
        field: command[f"expected_{field}"]
        for field in ("title", "date", "time")
        if command.get(f"expected_{field}", "")
    }


def evaluate_text_commands(commands: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for command in commands:
        expected = _expected_entities(command)
        result = parse_command(command["transcript"], command.get("reference_date") or None)
        predicted = result["entities"]
        rows.append(
            {
                "command_id": command["command_id"],
                "transcript": command["transcript"],
                "reference_date": command.get("reference_date", ""),
                "expected_intent": command["intent"],
                "predicted_intent": result["intent"],
                "intent_correct": _bool(result["intent"] == command["intent"]),
                "expected_title": expected.get("title", ""),
                "predicted_title": predicted.get("title", ""),
                "title_exact_match": _bool(
                    expected.get("title", "") == predicted.get("title", "")
                ),
                "expected_date": expected.get("date", ""),
                "predicted_date": predicted.get("date", ""),
                "date_exact_match": _bool(
                    expected.get("date", "") == predicted.get("date", "")
                ),
                "expected_time": expected.get("time", ""),
                "predicted_time": predicted.get("time", ""),
                "time_exact_match": _bool(
                    expected.get("time", "") == predicted.get("time", "")
                ),
                "entity_exact_match": _bool(predicted == expected),
                "missing_fields": json.dumps(
                    result["missing_fields"], ensure_ascii=False
                ),
            }
        )
    return rows


def _unevaluated_audio_row(
    command: dict[str, str],
    manifest: dict[str, str],
    *,
    error: str,
    lock_hash: str,
) -> dict[str, str]:
    expected = _expected_entities(command)
    return {
        "recording_id": manifest.get("recording_id", ""),
        "command_id": command["command_id"],
        "speaker_id": manifest.get("speaker_id", ""),
        "audio_path": manifest.get("audio_path", ""),
        "manifest_status": manifest.get("status", "missing_manifest"),
        "expected_transcript": command["transcript"],
        "whisper_transcript": "",
        "expected_intent": command["intent"],
        "predicted_intent": "",
        "intent_correct": "",
        "expected_title": expected.get("title", ""),
        "predicted_title": "",
        "title_exact_match": "",
        "expected_date": expected.get("date", ""),
        "predicted_date": "",
        "date_exact_match": "",
        "expected_time": expected.get("time", ""),
        "predicted_time": "",
        "time_exact_match": "",
        "entity_exact_match": "",
        "missing_fields": "[]",
        "asr_wer": "",
        "asr_cer": "",
        "latency_ms": "",
        "model": "",
        "language": "",
        "evaluated": "false",
        "asr_success": "false",
        "error": error,
        "test_config_sha256": lock_hash,
    }


def _pipeline_from_cached(
    cached: dict[str, str], command: dict[str, str]
) -> ASRNLUPipelineResult:
    transcript = cached["whisper_transcript"]
    nlu = parse_command(transcript, command.get("reference_date") or None)
    return {
        "success": cached.get("asr_success", "").lower() == "true",
        "transcript": transcript,
        "normalized_transcript": "",
        "model": cached.get("model", ""),
        "language": cached.get("language", "vi"),
        "latency_ms": float(cached.get("latency_ms") or 0),
        "intent": nlu["intent"],
        "entities": nlu["entities"],
        "missing_fields": nlu["missing_fields"],
        "can_execute": False,
        "can_write_database": False,
        "error": cached.get("error") or None,
    }


def evaluate_audio_commands(
    commands: list[dict[str, str]],
    manifests: list[dict[str, str]],
    *,
    config_path: Path,
    lock_hash: str,
    previous: dict[str, dict[str, str]] | None = None,
    pipeline_runner: PipelineRunner = run_asr_nlu_pipeline,
    checkpoint_path: Path | None = None,
) -> list[dict[str, str]]:
    manifest_by_command = {
        row["command_id"]: row for row in manifests if row.get("command_id")
    }
    previous = previous or {}
    output: list[dict[str, str]] = []
    for index, command in enumerate(commands, start=1):
        manifest = manifest_by_command.get(command["command_id"], {})
        path_value = manifest.get("audio_path", "")
        if manifest.get("status") != "recorded" or not path_value:
            output.append(
                _unevaluated_audio_row(
                    command, manifest, error="not_recorded", lock_hash=lock_hash
                )
            )
            continue
        if not Path(path_value).is_file():
            output.append(
                _unevaluated_audio_row(
                    command, manifest, error="missing_audio_file", lock_hash=lock_hash
                )
            )
            continue

        recording_id = manifest.get("recording_id", command["command_id"])
        cached = previous.get(recording_id)
        if (
            cached
            and cached.get("test_config_sha256") == lock_hash
            and cached.get("evaluated", "").lower() == "true"
            and cached.get("asr_success", "").lower() == "true"
        ):
            pipeline = _pipeline_from_cached(cached, command)
            print(f"[{index}/{len(commands)}] resume {recording_id}")
        else:
            try:
                pipeline = pipeline_runner(
                    path_value,
                    reference_date=command.get("reference_date") or None,
                    config_path=config_path,
                )
            except Exception as exc:  # preserve a complete, auditable test table
                output.append(
                    _unevaluated_audio_row(
                        command,
                        manifest,
                        error=f"pipeline_exception:{type(exc).__name__}:{exc}",
                        lock_hash=lock_hash,
                    )
                )
                if checkpoint_path:
                    _write_csv(checkpoint_path, AUDIO_FIELDS, output)
                print(f"[{index}/{len(commands)}] {recording_id}: exception={exc}")
                continue
            print(
                f"[{index}/{len(commands)}] {recording_id}: "
                f"success={pipeline['success']} intent={pipeline['intent']}"
            )

        expected = _expected_entities(command)
        predicted = pipeline["entities"]
        rates = calculate_error_rates(command["transcript"], pipeline["transcript"])
        output.append(
            {
                "recording_id": recording_id,
                "command_id": command["command_id"],
                "speaker_id": manifest.get("speaker_id", ""),
                "audio_path": path_value,
                "manifest_status": manifest.get("status", ""),
                "expected_transcript": command["transcript"],
                "whisper_transcript": pipeline["transcript"],
                "expected_intent": command["intent"],
                "predicted_intent": pipeline["intent"],
                "intent_correct": _bool(pipeline["intent"] == command["intent"]),
                "expected_title": expected.get("title", ""),
                "predicted_title": predicted.get("title", ""),
                "title_exact_match": _bool(
                    expected.get("title", "") == predicted.get("title", "")
                ),
                "expected_date": expected.get("date", ""),
                "predicted_date": predicted.get("date", ""),
                "date_exact_match": _bool(
                    expected.get("date", "") == predicted.get("date", "")
                ),
                "expected_time": expected.get("time", ""),
                "predicted_time": predicted.get("time", ""),
                "time_exact_match": _bool(
                    expected.get("time", "") == predicted.get("time", "")
                ),
                "entity_exact_match": _bool(predicted == expected),
                "missing_fields": json.dumps(
                    pipeline["missing_fields"], ensure_ascii=False
                ),
                "asr_wer": f"{rates['wer']:.6f}",
                "asr_cer": f"{rates['cer']:.6f}",
                "latency_ms": f"{pipeline['latency_ms']:.3f}",
                "model": pipeline["model"],
                "language": pipeline["language"],
                "evaluated": "true",
                "asr_success": _bool(pipeline["success"]),
                "error": pipeline["error"] or "",
                "test_config_sha256": lock_hash,
            }
        )
        if checkpoint_path:
            _write_csv(checkpoint_path, AUDIO_FIELDS, output)
    return output


def _accuracy(rows: list[dict[str, str]], field: str = "intent_correct") -> float | None:
    if not rows:
        return None
    return sum(row.get(field, "").lower() == "true" for row in rows) / len(rows)


def _intent_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    functional_labels = set(APPLICATION_LABELS) - {"OUT_OF_SCOPE"}
    functional = [row for row in rows if row["expected_intent"] in functional_labels]
    out_of_scope = [row for row in rows if row["expected_intent"] == "OUT_OF_SCOPE"]
    per_intent_counts: dict[str, dict[str, Any]] = {}
    for label in APPLICATION_LABELS:
        subset = [row for row in rows if row["expected_intent"] == label]
        correct = sum(row.get("intent_correct", "").lower() == "true" for row in subset)
        per_intent_counts[label] = {
            "correct": correct,
            "count": len(subset),
            "accuracy": correct / len(subset) if subset else None,
        }
    return {
        "overall_5_label_accuracy": _accuracy(rows),
        "functional_4_intent_accuracy": _accuracy(functional),
        "out_of_scope_rejection_accuracy": _accuracy(out_of_scope),
        "per_intent_accuracy": {
            label: values["accuracy"] for label, values in per_intent_counts.items()
        },
        "per_intent_counts": per_intent_counts,
    }


def build_oos_rows(
    commands: list[dict[str, str]],
    text_rows: list[dict[str, str]],
    audio_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    text_by_id = {row["command_id"]: row for row in text_rows}
    audio_by_id = {row["command_id"]: row for row in audio_rows}
    output: list[dict[str, str]] = []
    for command in commands:
        if command["intent"] != "OUT_OF_SCOPE":
            continue
        text = text_by_id[command["command_id"]]
        audio = audio_by_id[command["command_id"]]
        output.append(
            {
                "command_id": command["command_id"],
                "transcript": command["transcript"],
                "expected_intent": command["intent"],
                "text_predicted_intent": text["predicted_intent"],
                "text_correctly_rejected": text["intent_correct"],
                "audio_available": audio["evaluated"],
                "whisper_transcript": audio["whisper_transcript"],
                "whisper_predicted_intent": audio["predicted_intent"],
                "whisper_correctly_rejected": audio["intent_correct"],
                "error": audio["error"],
            }
        )
    return output


def build_entity_rows(
    commands: list[dict[str, str]],
    text_rows: list[dict[str, str]],
    audio_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    text_by_id = {row["command_id"]: row for row in text_rows}
    audio_by_id = {row["command_id"]: row for row in audio_rows}
    output: list[dict[str, str]] = []
    for command in commands:
        text = text_by_id[command["command_id"]]
        audio = audio_by_id[command["command_id"]]
        expected = _expected_entities(command)
        output.append(
            {
                "command_id": command["command_id"],
                "expected_intent": command["intent"],
                "entity_bearing_command": _bool(bool(expected)),
                "expected_title": text["expected_title"],
                "text_predicted_title": text["predicted_title"],
                "text_title_exact_match": text["title_exact_match"],
                "whisper_predicted_title": audio["predicted_title"],
                "whisper_title_exact_match": audio["title_exact_match"],
                "expected_date": text["expected_date"],
                "text_predicted_date": text["predicted_date"],
                "text_date_exact_match": text["date_exact_match"],
                "whisper_predicted_date": audio["predicted_date"],
                "whisper_date_exact_match": audio["date_exact_match"],
                "expected_time": text["expected_time"],
                "text_predicted_time": text["predicted_time"],
                "text_time_exact_match": text["time_exact_match"],
                "whisper_predicted_time": audio["predicted_time"],
                "whisper_time_exact_match": audio["time_exact_match"],
                "text_entity_exact_match": text["entity_exact_match"],
                "whisper_entity_exact_match": audio["entity_exact_match"],
                "audio_evaluated": audio["evaluated"],
            }
        )
    return output


def _entity_metrics(rows: list[dict[str, str]], prefix: str) -> dict[str, Any]:
    if prefix == "whisper":
        rows = [row for row in rows if row["audio_evaluated"] == "true"]
    entity_bearing = [row for row in rows if row["entity_bearing_command"] == "true"]
    per_field: dict[str, dict[str, Any]] = {}
    for field in ("date", "time", "title"):
        applicable = [row for row in rows if row[f"expected_{field}"]]
        per_field[field] = {
            "correct": sum(
                row[f"{prefix}_{field}_exact_match"] == "true" for row in applicable
            ),
            "count": len(applicable),
            "exact_match": _accuracy(applicable, f"{prefix}_{field}_exact_match"),
        }
    return {
        "evaluated_count": len(rows),
        "all_command_exact_match": _accuracy(rows, f"{prefix}_entity_exact_match"),
        "entity_bearing_count": len(entity_bearing),
        "entity_bearing_exact_match": _accuracy(
            entity_bearing, f"{prefix}_entity_exact_match"
        ),
        "per_field": per_field,
    }


def build_metrics(
    commands: list[dict[str, str]],
    text_rows: list[dict[str, str]],
    audio_rows: list[dict[str, str]],
    entity_rows: list[dict[str, str]],
    *,
    lock_path: Path,
    lock_hash: str,
) -> dict[str, Any]:
    successful_audio = [
        row
        for row in audio_rows
        if row["evaluated"] == "true" and row["asr_success"] == "true"
    ]
    measured = _intent_metrics(successful_audio)
    expected_by_intent = {
        label: sum(command["intent"] == label for command in commands)
        for label in APPLICATION_LABELS
    }
    correct_audio = sum(row["intent_correct"] == "true" for row in successful_audio)
    functional_labels = set(APPLICATION_LABELS) - {"OUT_OF_SCOPE"}
    correct_functional = sum(
        row["intent_correct"] == "true"
        for row in successful_audio
        if row["expected_intent"] in functional_labels
    )
    correct_oos = sum(
        row["intent_correct"] == "true"
        for row in successful_audio
        if row["expected_intent"] == "OUT_OF_SCOPE"
    )
    metrics: dict[str, Any] = {
        **measured,
        "metric_scope": "successful recorded command audio only",
        "complete": len(successful_audio) == len(commands),
        "expected_sample_count": len(commands),
        "audio_evaluated_count": sum(row["evaluated"] == "true" for row in audio_rows),
        "audio_successful_count": len(successful_audio),
        "audio_missing_count": sum(row["error"] == "not_recorded" for row in audio_rows),
        "audio_failure_count": sum(
            row["evaluated"] == "true" and row["asr_success"] != "true"
            for row in audio_rows
        ),
        "coverage": len(successful_audio) / len(commands) if commands else 0.0,
        "coverage_adjusted_accuracy_missing_not_counted_as_predictions": {
            "overall_5_label_correct_over_expected": correct_audio / len(commands)
            if commands
            else None,
            "functional_4_intent_correct_over_expected": correct_functional
            / sum(value for key, value in expected_by_intent.items() if key in functional_labels),
            "out_of_scope_correct_over_expected": correct_oos
            / expected_by_intent["OUT_OF_SCOPE"],
        },
        "ground_truth_text": _intent_metrics(text_rows),
        "entity_exact_match": {
            "ground_truth_text": _entity_metrics(entity_rows, "text"),
            "audio_whisper": _entity_metrics(entity_rows, "whisper"),
        },
        "test_config": str(lock_path),
        "test_config_sha256": lock_hash,
        "label_provenance_note": (
            "Application labels are project-defined. They are not claimed to be "
            "the original Speech-MASSIVE intent labels."
        ),
        "artifacts": {
            "intent_ground_truth": "reports/nlu/intent_test_ground_truth.csv",
            "intent_whisper": "reports/nlu/intent_test_whisper.csv",
            "out_of_scope": "reports/nlu/out_of_scope_test_results.csv",
            "entities": "reports/nlu/entity_test_results.csv",
        },
    }
    return metrics


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commands", type=Path, default=Path("data/metadata/command_test.csv")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/commands/command_audio_manifest.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--test-lock", type=Path, default=Path("reports/asr_nlu_test_config.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/nlu"))
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    verify_snapshot(args.test_lock)
    lock_hash = sha256_file(args.test_lock)
    commands = [row for row in _read_csv(args.commands) if row.get("split") == "test"]
    manifests = [row for row in _read_csv(args.manifest) if row.get("split") == "test"]
    if not commands:
        raise ValueError("No command test rows found")

    ground_path = args.output_dir / "intent_test_ground_truth.csv"
    whisper_path = args.output_dir / "intent_test_whisper.csv"
    metrics_path = args.output_dir / "intent_test_metrics.json"
    oos_path = args.output_dir / "out_of_scope_test_results.csv"
    entity_path = args.output_dir / "entity_test_results.csv"

    text_rows = evaluate_text_commands(commands)
    _write_csv(ground_path, TEXT_FIELDS, text_rows)

    previous: dict[str, dict[str, str]] = {}
    if args.resume and whisper_path.is_file():
        previous = {
            row["recording_id"]: row
            for row in _read_csv(whisper_path)
            if row.get("recording_id")
        }
    audio_rows = evaluate_audio_commands(
        commands,
        manifests,
        config_path=args.config,
        lock_hash=lock_hash,
        previous=previous,
        checkpoint_path=whisper_path,
    )
    _write_csv(whisper_path, AUDIO_FIELDS, audio_rows)

    oos_rows = build_oos_rows(commands, text_rows, audio_rows)
    entity_rows = build_entity_rows(commands, text_rows, audio_rows)
    _write_csv(oos_path, OOS_FIELDS, oos_rows)
    _write_csv(entity_path, ENTITY_FIELDS, entity_rows)
    metrics = build_metrics(
        commands,
        text_rows,
        audio_rows,
        entity_rows,
        lock_path=args.test_lock,
        lock_hash=lock_hash,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["audio_failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
