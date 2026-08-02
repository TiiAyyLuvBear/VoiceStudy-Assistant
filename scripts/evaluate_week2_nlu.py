"""Generate Week 2 text-level and audio-to-NLU validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from src.asr.metrics import calculate_error_rates
from src.nlu.command_parser import parse_command
from src.nlu.missing_fields import can_execute_command, can_write_database
from src.pipeline.asr_nlu import run_asr_nlu_pipeline


GROUND_TRUTH_FIELDS = (
    "command_id",
    "split",
    "transcript",
    "reference_date",
    "expected_intent",
    "predicted_intent",
    "intent_correct",
    "expected_entities",
    "predicted_entities",
    "missing_fields",
    "can_execute",
    "can_write_database",
)

OUT_OF_SCOPE_FIELDS = (
    "command_id",
    "transcript",
    "expected_intent",
    "predicted_intent",
    "correctly_rejected",
    "entities_empty",
    "missing_fields_empty",
)

ENTITY_FIELDS = (
    "command_id",
    "intent",
    "expected_title",
    "predicted_title",
    "title_match",
    "expected_date",
    "predicted_date",
    "date_match",
    "expected_time",
    "predicted_time",
    "time_match",
    "entity_exact_match",
)

WHISPER_FIELDS = (
    "recording_id",
    "command_id",
    "split",
    "manifest_status",
    "speaker_id",
    "audio_path",
    "expected_transcript",
    "whisper_transcript",
    "expected_intent",
    "predicted_intent",
    "intent_correct",
    "expected_title",
    "predicted_title",
    "expected_date",
    "predicted_date",
    "expected_time",
    "predicted_time",
    "entity_exact_match",
    "missing_fields",
    "can_execute",
    "can_write_database",
    "asr_wer",
    "asr_cer",
    "latency_ms",
    "model",
    "language",
    "asr_success",
    "evaluated",
    "error",
)


def _bool(value: bool) -> str:
    return str(value).lower()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _expected_entities(command: dict[str, str]) -> dict[str, str]:
    return {
        name: command[f"expected_{name}"]
        for name in ("title", "date", "time")
        if command.get(f"expected_{name}", "")
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_text_rows(
    commands: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Evaluate intent, OOS, and entities from reference transcripts."""

    ground_truth: list[dict[str, str]] = []
    out_of_scope: list[dict[str, str]] = []
    entities: list[dict[str, str]] = []
    for command in commands:
        expected = _expected_entities(command)
        result = parse_command(command["transcript"], command.get("reference_date") or None)
        intent_correct = result["intent"] == command["intent"]
        entity_exact = result["entities"] == expected
        executable = can_execute_command(result["intent"], result["missing_fields"])
        database_write = can_write_database(result["intent"], result["entities"])
        ground_truth.append(
            {
                "command_id": command["command_id"],
                "split": command.get("split", ""),
                "transcript": command["transcript"],
                "reference_date": command.get("reference_date", ""),
                "expected_intent": command["intent"],
                "predicted_intent": result["intent"],
                "intent_correct": _bool(intent_correct),
                "expected_entities": _json(expected),
                "predicted_entities": _json(result["entities"]),
                "missing_fields": _json(result["missing_fields"]),
                "can_execute": _bool(executable),
                "can_write_database": _bool(database_write),
            }
        )
        if command["intent"] == "OUT_OF_SCOPE":
            out_of_scope.append(
                {
                    "command_id": command["command_id"],
                    "transcript": command["transcript"],
                    "expected_intent": command["intent"],
                    "predicted_intent": result["intent"],
                    "correctly_rejected": _bool(intent_correct),
                    "entities_empty": _bool(not result["entities"]),
                    "missing_fields_empty": _bool(not result["missing_fields"]),
                }
            )
        entities.append(
            {
                "command_id": command["command_id"],
                "intent": command["intent"],
                "expected_title": expected.get("title", ""),
                "predicted_title": result["entities"].get("title", ""),
                "title_match": _bool(
                    expected.get("title", "") == result["entities"].get("title", "")
                ),
                "expected_date": expected.get("date", ""),
                "predicted_date": result["entities"].get("date", ""),
                "date_match": _bool(
                    expected.get("date", "") == result["entities"].get("date", "")
                ),
                "expected_time": expected.get("time", ""),
                "predicted_time": result["entities"].get("time", ""),
                "time_match": _bool(
                    expected.get("time", "") == result["entities"].get("time", "")
                ),
                "entity_exact_match": _bool(entity_exact),
            }
        )
    return ground_truth, out_of_scope, entities


def _pending_whisper_row(
    command: dict[str, str],
    manifest: dict[str, str],
) -> dict[str, str]:
    expected = _expected_entities(command)
    return {
        "recording_id": manifest.get("recording_id", ""),
        "command_id": command["command_id"],
        "split": command.get("split", ""),
        "manifest_status": manifest.get("status", "missing_manifest"),
        "speaker_id": manifest.get("speaker_id", ""),
        "audio_path": manifest.get("audio_path", ""),
        "expected_transcript": command["transcript"],
        "whisper_transcript": "",
        "expected_intent": command["intent"],
        "predicted_intent": "",
        "intent_correct": "",
        "expected_title": expected.get("title", ""),
        "predicted_title": "",
        "expected_date": expected.get("date", ""),
        "predicted_date": "",
        "expected_time": expected.get("time", ""),
        "predicted_time": "",
        "entity_exact_match": "",
        "missing_fields": "[]",
        "can_execute": "false",
        "can_write_database": "false",
        "asr_wer": "",
        "asr_cer": "",
        "latency_ms": "",
        "model": "",
        "language": "",
        "asr_success": "false",
        "evaluated": "false",
        "error": "not_recorded",
    }


def _pipeline_from_cache(
    cached: dict[str, str],
    command: dict[str, str],
) -> dict[str, Any]:
    transcript = cached["whisper_transcript"]
    nlu = parse_command(transcript, command.get("reference_date") or None)
    return {
        "success": cached.get("asr_success", "").lower() == "true",
        "transcript": transcript,
        "model": cached.get("model", ""),
        "language": cached.get("language", "vi"),
        "latency_ms": float(cached.get("latency_ms") or 0),
        "intent": nlu["intent"],
        "entities": nlu["entities"],
        "missing_fields": nlu["missing_fields"],
        "can_execute": can_execute_command(nlu["intent"], nlu["missing_fields"]),
        "can_write_database": can_write_database(nlu["intent"], nlu["entities"]),
        "error": cached.get("error") or None,
    }


def evaluate_audio_rows(
    commands: list[dict[str, str]],
    manifests: list[dict[str, str]],
    *,
    config_path: Path,
    previous: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Evaluate available command audio and preserve explicit pending rows."""

    manifest_by_command = {
        row["command_id"]: row for row in manifests if row.get("command_id")
    }
    output: list[dict[str, str]] = []
    for index, command in enumerate(commands, start=1):
        manifest = manifest_by_command.get(command["command_id"], {})
        path_value = manifest.get("audio_path", "")
        if manifest.get("status") != "recorded" or not path_value:
            output.append(_pending_whisper_row(command, manifest))
            continue

        recording_id = manifest.get("recording_id", command["command_id"])
        cached = previous.get(recording_id)
        if cached and cached.get("asr_success", "").lower() == "true":
            pipeline = _pipeline_from_cache(cached, command)
            print(f"[{index}/{len(commands)}] resume {recording_id}")
        else:
            pipeline = run_asr_nlu_pipeline(
                path_value,
                reference_date=command.get("reference_date") or None,
                config_path=config_path,
            )
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
                "split": command.get("split", ""),
                "manifest_status": manifest.get("status", ""),
                "speaker_id": manifest.get("speaker_id", ""),
                "audio_path": path_value,
                "expected_transcript": command["transcript"],
                "whisper_transcript": pipeline["transcript"],
                "expected_intent": command["intent"],
                "predicted_intent": pipeline["intent"],
                "intent_correct": _bool(pipeline["intent"] == command["intent"]),
                "expected_title": expected.get("title", ""),
                "predicted_title": predicted.get("title", ""),
                "expected_date": expected.get("date", ""),
                "predicted_date": predicted.get("date", ""),
                "expected_time": expected.get("time", ""),
                "predicted_time": predicted.get("time", ""),
                "entity_exact_match": _bool(predicted == expected),
                "missing_fields": _json(pipeline["missing_fields"]),
                "can_execute": _bool(pipeline["can_execute"]),
                "can_write_database": _bool(pipeline["can_write_database"]),
                "asr_wer": f"{rates['wer']:.6f}",
                "asr_cer": f"{rates['cer']:.6f}",
                "latency_ms": f"{pipeline['latency_ms']:.3f}",
                "model": pipeline["model"],
                "language": pipeline["language"],
                "asr_success": _bool(pipeline["success"]),
                "evaluated": "true",
                "error": pipeline["error"] or "",
            }
        )
    return output


def _ratio(rows: list[dict[str, str]], field: str) -> float:
    return (
        sum(row.get(field, "").lower() == "true" for row in rows) / len(rows)
        if rows
        else 0.0
    )


def _optional_ratio(
    rows: list[dict[str, str]], field: str
) -> float | None:
    return _ratio(rows, field) if rows else None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commands",
        type=Path,
        default=Path("data/metadata/command_validation.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/commands/command_audio_manifest.csv"),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/nlu"))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    commands = [
        row for row in _read_csv(args.commands) if row.get("split") == args.split
    ]
    manifests = [
        row for row in _read_csv(args.manifest) if row.get("split") == args.split
    ]
    ground_truth, out_of_scope, entities = evaluate_text_rows(commands)
    ground_path = args.output_dir / "intent_validation_ground_truth.csv"
    oos_path = args.output_dir / "out_of_scope_validation.csv"
    entity_path = args.output_dir / "entity_validation_results.csv"
    whisper_path = args.output_dir / "intent_validation_whisper.csv"
    summary_path = args.output_dir / "week2_validation_summary.json"
    _write_csv(ground_path, GROUND_TRUTH_FIELDS, ground_truth)
    _write_csv(oos_path, OUT_OF_SCOPE_FIELDS, out_of_scope)
    _write_csv(entity_path, ENTITY_FIELDS, entities)

    manifest_by_command = {row.get("command_id", ""): row for row in manifests}
    ready = [
        command
        for command in commands
        if manifest_by_command.get(command["command_id"], {}).get("status")
        == "recorded"
        and Path(
            manifest_by_command.get(command["command_id"], {}).get("audio_path", "")
        ).is_file()
    ]
    if len(ready) != len(commands) and not args.allow_incomplete:
        print(
            f"Command audio incomplete: {len(ready)}/{len(commands)} ready. "
            "Record the remaining audio or use --allow-incomplete for a partial run."
        )
        return 2

    previous: dict[str, dict[str, str]] = {}
    if args.resume and whisper_path.is_file():
        previous = {
            row["recording_id"]: row
            for row in _read_csv(whisper_path)
            if row.get("recording_id")
        }
    whisper = evaluate_audio_rows(
        commands,
        manifests,
        config_path=args.config,
        previous=previous,
    )
    _write_csv(whisper_path, WHISPER_FIELDS, whisper)
    evaluated = [row for row in whisper if row["evaluated"] == "true"]
    successful = [row for row in evaluated if row["asr_success"] == "true"]
    audio_oos = [row for row in successful if row["expected_intent"] == "OUT_OF_SCOPE"]
    summary = {
        "commands_file": str(args.commands),
        "manifest_file": str(args.manifest),
        "split": args.split,
        "text_sample_count": len(commands),
        "text_intent_accuracy": _ratio(ground_truth, "intent_correct"),
        "text_entity_exact_match": _ratio(entities, "entity_exact_match"),
        "out_of_scope_sample_count": len(out_of_scope),
        "out_of_scope_rejection_rate": _ratio(out_of_scope, "correctly_rejected"),
        "audio_expected_count": len(commands),
        "audio_ready_count": len(ready),
        "audio_evaluated_count": len(evaluated),
        "audio_successful_count": len(successful),
        "audio_failure_count": len(evaluated) - len(successful),
        "audio_intent_accuracy": _ratio(successful, "intent_correct"),
        "audio_entity_exact_match": _ratio(successful, "entity_exact_match"),
        "audio_out_of_scope_count": len(audio_oos),
        "audio_out_of_scope_rejection_rate": _optional_ratio(
            audio_oos, "intent_correct"
        ),
        "complete": len(ready) == len(commands),
        "intent_validation_ground_truth": str(ground_path),
        "intent_validation_whisper": str(whisper_path),
        "out_of_scope_validation": str(oos_path),
        "entity_validation_results": str(entity_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["audio_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
