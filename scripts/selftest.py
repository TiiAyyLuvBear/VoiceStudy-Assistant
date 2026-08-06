"""Run deterministic end-to-end VoiceStudy self-tests without audio models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.seed_database import seed_database
from src.pipeline.orchestrator import process_audio_request


def _pipeline(intent: str, entities: dict[str, str] | None = None) -> Callable:
    """Build deterministic ASR/NLU result for a named test scenario."""

    def run(*args, **kwargs) -> dict:
        return {
            "success": True,
            "transcript": "mock command",
            "normalized_transcript": "mock command",
            "model": "mock-asr",
            "language": "vi",
            "latency_ms": 0.0,
            "intent": intent,
            "entities": entities or {},
            "missing_fields": [],
            "can_execute": True,
            "can_write_database": intent == "ADD_SCHEDULE",
            "error": None,
        }

    return run


def _known_speaker(*args, **kwargs) -> dict:
    return {
        "success": True,
        "candidate_user_id": "demo-anh",
        "similarity": 0.99,
        "identified": True,
        "centroid_path": "mock-centroid.npy",
        "error": None,
    }


def _verified_speaker(*args, **kwargs) -> dict:
    return {
        "success": True,
        "candidate_user_id": "demo-anh",
        "similarity": 0.99,
        "verified": True,
        "centroid_path": "mock-centroid.npy",
        "error": None,
    }


def _actual_output(result: dict) -> dict:
    return {
        "policy": result["policy"],
        "speaker_id": result["speaker"].get("candidate_user_id"),
        "verified": result["speaker"].get("verified"),
        "response": result["response"],
        "error": result["error"],
    }


def _matches_expected(actual: dict, expected: dict) -> bool:
    return all(
        actual.get(key) == value if key != "response_contains"
        else value in actual["response"]
        for key, value in expected.items()
    )


def run_selftest(database_path: str | Path) -> list[dict]:
    """Seed mock data and return one structured log record per workflow."""

    seed_database(database_path)
    common = {"database_path": database_path, "identifier": _known_speaker,
              "verifier": _verified_speaker}
    cases = (
        ("public_time", "GET_TIME", {}, {"policy": "PUBLIC", "speaker_id": None,
         "error": None}, _pipeline("GET_TIME")),
        ("view_schedule", "VIEW_SCHEDULE", {}, {"policy": "SID",
         "speaker_id": "demo-anh", "response_contains": "Học Thống kê", "error": None},
         _pipeline("VIEW_SCHEDULE")),
        ("private_note", "VIEW_PRIVATE_NOTE", {}, {"policy": "SID_AND_SV",
         "speaker_id": "demo-anh", "verified": True,
         "response_contains": "Hoàn thành báo cáo", "error": None},
         _pipeline("VIEW_PRIVATE_NOTE")),
        (
            "add_schedule",
            "ADD_SCHEDULE",
            {"title": "Tự kiểm tra", "date": "2026-08-06", "time": "09:00"},
            {"policy": "SID", "speaker_id": "demo-anh",
             "response_contains": "Đã thêm lịch Tự kiểm tra", "error": None},
            _pipeline("ADD_SCHEDULE", {
                "title": "Tự kiểm tra", "date": "2026-08-06", "time": "09:00",
            }),
        ),
        ("out_of_scope", "OUT_OF_SCOPE", {}, {"policy": "REJECT",
         "speaker_id": None, "error": "OUT_OF_SCOPE"}, _pipeline("OUT_OF_SCOPE")),
    )
    logs = []
    for name, intent, entities, expected, runner in cases:
        result = process_audio_request(
            f"mock-{name}.wav", asr_nlu_runner=runner, **common,
        )
        actual = _actual_output(result)
        logs.append({
            "task_name": name,
            "input": {"audio_path": f"mock-{name}.wav", "intent": intent,
                      "entities": entities},
            "expected_output": expected,
            "actual_output": actual,
            "result": "pass" if _matches_expected(actual, expected) else "not pass",
        })
    return logs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path,
        default=Path("data/database/voicestudy-selftest.db"),
        help="Isolated SQLite file for mock records (default: %(default)s)",
    )
    args = parser.parse_args()
    logs = run_selftest(args.database)
    passed = all(log["result"] == "pass" for log in logs)
    print(json.dumps({"database": str(args.database), "passed": passed, "logs": logs},
                     ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
