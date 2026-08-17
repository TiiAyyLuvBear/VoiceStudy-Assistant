"""Run deterministic Week-3 system contracts without changing speaker-v2 data."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.user_repository import create_user
from src.pipeline.orchestrator import process_audio_request
from src.speaker.application import enroll_user, identify_application_user
from src.tasks.note_tasks import add_note
from src.tasks.schedule_tasks import add_schedule, get_schedules
from src.utils.files import sha256_file


OUTPUT_DIR = PROJECT_ROOT / "experiments/system"
SYSTEM_CASES = (
    ("SYS001", "PUBLIC", "GET_TIME", "time_normal", "PUBLIC", "NONE"),
    ("SYS002", "PUBLIC", "GET_TIME", "time_morning", "PUBLIC", "NONE"),
    ("SYS003", "PUBLIC", "GET_TIME", "time_evening", "PUBLIC", "NONE"),
    ("SYS004", "PUBLIC", "GET_TIME", "time_with_user_claim", "PUBLIC", "NONE"),
    ("SYS005", "SID", "VIEW_SCHEDULE", "view_all", "SID", "NONE"),
    ("SYS006", "SID", "VIEW_SCHEDULE", "view_by_date", "SID", "NONE"),
    ("SYS007", "SID", "VIEW_SCHEDULE", "view_claim_other", "SID", "NONE"),
    ("SYS008", "SID", "VIEW_SCHEDULE", "unknown_view", "SID", "UNKNOWN_SPEAKER"),
    ("SYS009", "SID", "VIEW_SCHEDULE", "no_enrollment", "SID", "NO_ENROLLMENT"),
    ("SYS010", "ERROR", "VIEW_SCHEDULE", "bad_audio_view", "SID", "ANY_ERROR"),
    ("SYS011", "SID_WRITE", "ADD_SCHEDULE", "add_valid", "SID", "NONE"),
    ("SYS012", "SID_WRITE", "ADD_SCHEDULE", "add_claim_other", "SID", "NONE"),
    ("SYS013", "SID_WRITE", "ADD_SCHEDULE", "missing_title", "SID", "MISSING_FIELDS"),
    ("SYS014", "SID_WRITE", "ADD_SCHEDULE", "missing_date", "SID", "MISSING_FIELDS"),
    ("SYS015", "SID_WRITE", "ADD_SCHEDULE", "missing_time", "SID", "MISSING_FIELDS"),
    ("SYS016", "SID_WRITE", "ADD_SCHEDULE", "unknown_add", "SID", "UNKNOWN_SPEAKER"),
    ("SYS017", "ERROR", "ADD_SCHEDULE", "empty_transcript_add", "SID", "EMPTY_TRANSCRIPT"),
    ("SYS018", "SID_SV", "VIEW_PRIVATE_NOTE", "private_valid", "SID_AND_SV", "NONE"),
    ("SYS019", "SID_SV", "VIEW_PRIVATE_NOTE", "private_claim_other", "SID_AND_SV", "NONE"),
    ("SYS020", "SID_SV", "VIEW_PRIVATE_NOTE", "impostor", "SID_AND_SV", "VERIFICATION_FAILED"),
    ("SYS021", "SID_SV", "VIEW_PRIVATE_NOTE", "unknown_private", "SID_AND_SV", "UNKNOWN_SPEAKER"),
    ("SYS022", "SID_SV", "VIEW_PRIVATE_NOTE", "sv_missing_centroid", "SID_AND_SV", "CENTROID_NOT_FOUND"),
    ("SYS023", "SID_SV", "VIEW_PRIVATE_NOTE", "sv_exception", "SID_AND_SV", "SV_ERROR"),
    ("SYS024", "SID_SV", "VIEW_PRIVATE_NOTE", "sid_exception", "SID_AND_SV", "SID_ERROR"),
    ("SYS025", "REJECT", "OUT_OF_SCOPE", "music", "REJECT", "OUT_OF_SCOPE"),
    ("SYS026", "REJECT", "OUT_OF_SCOPE", "weather", "REJECT", "OUT_OF_SCOPE"),
    ("SYS027", "REJECT", "OUT_OF_SCOPE", "phone", "REJECT", "OUT_OF_SCOPE"),
    ("SYS028", "ERROR", "OUT_OF_SCOPE", "empty_transcript_oos", "REJECT", "EMPTY_TRANSCRIPT"),
    ("SYS029", "ERROR", "OUT_OF_SCOPE", "asr_failure", "REJECT", "ASR_FAILED"),
    ("SYS030", "REJECT", "OUT_OF_SCOPE", "unsupported", "REJECT", "OUT_OF_SCOPE"),
)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pipeline(intent: str, scenario: str):
    entities: dict[str, str] = {}
    missing: list[str] = []
    transcript = f"system command {scenario}"
    if intent == "VIEW_SCHEDULE" and scenario == "view_by_date":
        entities["date"] = "2026-08-13"
    if scenario in {"time_with_user_claim", "view_claim_other", "add_claim_other", "private_claim_other"}:
        entities["user_id"] = "user_002"
    if intent == "ADD_SCHEDULE":
        entities.update({"title": "Lịch hệ thống", "date": "2026-08-14", "time": "10:30"})
        if scenario.startswith("missing_"):
            field = scenario.removeprefix("missing_")
            entities.pop(field)
            missing.append(field)
    if scenario.startswith("empty_transcript"):
        transcript = ""

    def run(*args, **kwargs):
        if scenario == "asr_failure":
            return {
                "success": False,
                "transcript": "",
                "normalized_transcript": "",
                "model": "deterministic-asr",
                "language": "vi",
                "latency_ms": 0.1,
                "intent": "OUT_OF_SCOPE",
                "entities": {},
                "missing_fields": [],
                "error": "ASR_FAILED",
            }
        return {
            "success": True,
            "transcript": transcript,
            "normalized_transcript": transcript,
            "model": "deterministic-asr",
            "language": "vi",
            "latency_ms": 0.1,
            "intent": intent,
            "entities": entities,
            "missing_fields": missing,
            "can_execute": not missing,
            "can_write_database": intent == "ADD_SCHEDULE" and not missing,
            "error": None,
        }

    return run


def _callbacks(scenario: str, calls: dict[str, int]):
    def identify(*args, **kwargs):
        calls["sid"] += 1
        if scenario == "sid_exception":
            raise RuntimeError("simulated SID exception")
        if scenario == "no_enrollment":
            return {"success": False, "identified": False, "error": "NO_ENROLLMENT"}
        known = not scenario.startswith("unknown_")
        return {
            "protocol": "APPLICATION_SID",
            "success": True,
            "candidate_user_id": "user_001" if known else None,
            "cosine_similarity": 0.91 if known else 0.2,
            "similarity": 0.91 if known else 0.2,
            "unknown_threshold": 0.68,
            "status": "KNOWN" if known else "UNKNOWN",
            "identified": known,
            "centroid_path": "contract-centroid.npy" if known else None,
            "latency_ms": 0.2,
            "error": None,
        }

    def verify(*args, **kwargs):
        calls["sv"] += 1
        if scenario == "sv_exception":
            raise RuntimeError("simulated SV exception")
        if scenario == "sv_missing_centroid":
            return {
                "success": False,
                "candidate_user_id": args[1],
                "verified": False,
                "error": "CENTROID_NOT_FOUND",
            }
        verified = scenario != "impostor"
        return {
            "protocol": "APPLICATION_SV",
            "success": True,
            "candidate_user_id": args[1],
            "similarity": 0.9 if verified else 0.1,
            "verification_threshold": 0.72,
            "verified": verified,
            "latency_ms": 0.1,
            "error": None,
        }

    return identify, verify


def _error_matches(actual: str | None, expected: str) -> bool:
    if expected == "NONE":
        return actual is None
    if expected == "ANY_ERROR":
        return bool(actual)
    return bool(actual) and (actual == expected or actual.startswith(expected))


def _seed(database: Path) -> None:
    create_user("user_001", "System User One", database_path=database)
    create_user("user_002", "System User Two", database_path=database)
    add_schedule("user_001", "Lịch user 1", "2026-08-13", "08:00", database_path=database)
    add_schedule("user_002", "Lịch user 2", "2026-08-13", "09:00", database_path=database)
    add_note("user_001", "Ghi chú user 1", database_path=database)
    add_note("user_002", "Ghi chú user 2", database_path=database)


def run_system_suite(output_dir: Path = OUTPUT_DIR) -> tuple[list[dict], dict]:
    case_rows = [
        {
            "test_case_id": case_id,
            "category": category,
            "intent": intent,
            "scenario": scenario,
            "expected_policy": policy,
            "expected_error": error,
            "execution_mode": "deterministic_integration_contract",
        }
        for case_id, category, intent, scenario, policy, error in SYSTEM_CASES
    ]
    results: list[dict] = []
    with TemporaryDirectory(prefix="voicestudy-system-tests-") as directory:
        root = Path(directory)
        for case_id, category, intent, scenario, policy, expected_error in SYSTEM_CASES:
            database = root / f"{case_id}.db"
            audio = root / f"{case_id}.wav"
            if scenario != "bad_audio_view":
                audio.write_bytes(b"deterministic-system-audio")
            _seed(database)
            before_user_1 = len(get_schedules("user_001", database_path=database))
            before_user_2 = len(get_schedules("user_002", database_path=database))
            calls = {"sid": 0, "sv": 0}
            identifier, verifier = _callbacks(scenario, calls)
            result = process_audio_request(
                audio,
                database_path=database,
                asr_nlu_runner=_pipeline(intent, scenario),
                identifier=identifier,
                verifier=verifier,
            )
            after_user_1 = len(get_schedules("user_001", database_path=database))
            after_user_2 = len(get_schedules("user_002", database_path=database))
            expected_sid = policy in {"SID", "SID_AND_SV"} and scenario not in {
                "bad_audio_view", "empty_transcript_add",
            }
            expected_sv = (
                policy == "SID_AND_SV"
                and scenario not in {"unknown_private", "sid_exception"}
            )
            access_ok = (
                result["policy"] == policy
                or scenario in {"bad_audio_view", "empty_transcript_add", "asr_failure"}
            )
            speaker_calls_ok = (
                calls["sid"] == int(expected_sid)
                and calls["sv"] == int(expected_sv)
            )
            database_ok = after_user_2 == before_user_2
            if scenario in {"add_valid", "add_claim_other"}:
                database_ok = database_ok and after_user_1 == before_user_1 + 1
            elif intent == "ADD_SCHEDULE":
                database_ok = database_ok and after_user_1 == before_user_1
            response_ok = "user 2" not in result["response"].lower()
            passed = all(
                (
                    _error_matches(result["error"], expected_error),
                    access_ok,
                    speaker_calls_ok,
                    database_ok,
                    response_ok,
                )
            )
            results.append(
                {
                    "test_case_id": case_id,
                    "category": category,
                    "scenario": scenario,
                    "intent": intent,
                    "actual_policy": result["policy"],
                    "actual_error": result["error"] or "",
                    "sid_calls": calls["sid"],
                    "sv_calls": calls["sv"],
                    "database_integrity_pass": str(database_ok).lower(),
                    "response_scope_pass": str(response_ok).lower(),
                    "latency_ms": f"{float(result['latency_ms']):.6f}",
                    "passed": str(passed).lower(),
                }
            )

    passed_count = sum(row["passed"] == "true" for row in results)
    latencies = [float(row["latency_ms"]) for row in results]
    metrics = {
        "execution_mode": "deterministic_integration_contract",
        "speaker_dataset_version": "v2",
        "test_case_count": len(results),
        "passed_count": passed_count,
        "failed_count": len(results) - passed_count,
        "task_success_rate": passed_count / len(results),
        "mean_system_latency_ms": statistics.fmean(latencies),
        "max_system_latency_ms": max(latencies),
        "note": (
            "Measures orchestration, access policy, database ownership and error "
            "contracts. ASR and speaker recognition accuracy are evaluated by "
            "their frozen component test artifacts."
        ),
    }
    latency_rows = [
        {
            "test_case_id": row["test_case_id"],
            "scenario": row["scenario"],
            "latency_ms": row["latency_ms"],
            "execution_mode": metrics["execution_mode"],
        }
        for row in results
    ]
    _write_csv(
        output_dir / "system_test_cases.csv",
        case_rows,
        list(case_rows[0]),
    )
    _write_csv(
        output_dir / "system_test_results.csv",
        results,
        list(results[0]),
    )
    _write_csv(
        output_dir / "system_latency_results.csv",
        latency_rows,
        list(latency_rows[0]),
    )
    workflow_outputs = {
        "public_test_results.csv": lambda row: row["category"] == "PUBLIC",
        "sid_personalization_results.csv": lambda row: (
            row["category"] == "SID"
        ),
        "add_schedule_results.csv": lambda row: row["category"] == "SID_WRITE",
        "private_access_results.csv": lambda row: row["category"] == "SID_SV",
        "unknown_access_results.csv": lambda row: row["scenario"].startswith(
            "unknown_"
        ),
        "impostor_results.csv": lambda row: row["scenario"] == "impostor",
        "out_of_scope_system_results.csv": lambda row: (
            row["intent"] == "OUT_OF_SCOPE"
        ),
    }
    for filename, predicate in workflow_outputs.items():
        selected_rows = [row for row in results if predicate(row)]
        _write_csv(output_dir / filename, selected_rows, list(results[0]))
    (output_dir / "task_success_rate.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results, metrics


class _DynamicExtractor:
    def extract(self, audio, *, sample_rate):
        vector = np.asarray(audio, dtype=np.float32).reshape(-1)
        vector = vector / np.linalg.norm(vector)
        return vector, vector.size, 0.1


def run_dynamic_enrollment_test(output_dir: Path = OUTPUT_DIR) -> dict:
    svm_path = PROJECT_ROOT / "models/experimental/v2/speaker_svm_linear.pkl"
    svm_before = sha256_file(svm_path)
    with TemporaryDirectory(prefix="voicestudy-dynamic-enrollment-") as directory:
        root = Path(directory)
        centroid_dir = root / "centroids"
        sid_threshold = root / "sid.json"
        sv_threshold = root / "sv.json"
        sid_threshold.write_text('{"threshold": 0.70}', encoding="utf-8")
        sv_threshold.write_text('{"threshold": 0.72}', encoding="utf-8")
        config = root / "config.yaml"
        config.write_text(
            "\n".join(
                (
                    "speaker:",
                    f"  application_centroid_dir: {centroid_dir.as_posix()}",
                    f"  application_sid_threshold_path: {sid_threshold.as_posix()}",
                    f"  application_verification_threshold_path: {sv_threshold.as_posix()}",
                )
            ),
            encoding="utf-8",
        )
        database = root / "dynamic.db"
        audio_paths = []
        for index in range(5):
            path = root / f"user004-enroll-{index}.wav"
            path.write_bytes(b"independent-enrollment-audio")
            audio_paths.append(path)
        query = root / "user004-heldout-query.wav"
        query.write_bytes(b"independent-heldout-query")

        def loader(path):
            if "heldout" in Path(path).name:
                return np.asarray([0.98, 0.2], dtype=np.float32), 16000
            return np.asarray([1.0, 0.0], dtype=np.float32), 16000

        enrollment = enroll_user(
            "user_004",
            "Dynamic User 004",
            audio_paths,
            database_path=database,
            config_path=config,
            extractor=_DynamicExtractor(),
            audio_loader=loader,
        )
        add_schedule(
            "user_004",
            "Lịch riêng user 004",
            "2026-08-15",
            "14:00",
            database_path=database,
        )
        sid = identify_application_user(
            query,
            database_path=database,
            config_path=config,
            extractor=_DynamicExtractor(),
            audio_loader=loader,
        )

        def identifier(audio_path, **kwargs):
            return identify_application_user(
                audio_path,
                database_path=database,
                config_path=config,
                extractor=_DynamicExtractor(),
                audio_loader=loader,
            )

        result = process_audio_request(
            query,
            database_path=database,
            config_path=config,
            asr_nlu_runner=_pipeline("VIEW_SCHEDULE", "dynamic_user_004"),
            identifier=identifier,
        )
        centroid_value = Path(enrollment.get("centroid_path") or "")
        centroid_created = (
            centroid_value
            if centroid_value.is_absolute()
            else config.parent / centroid_value
        ).is_file()
        svm_after = sha256_file(svm_path)
        passed = all(
            (
                enrollment["success"],
                enrollment["embedding_count"] == 5,
                centroid_created,
                sid["candidate_user_id"] == "user_004",
                sid["identified"],
                result["speaker"]["candidate_user_id"] == "user_004",
                "user 004" in result["response"].lower(),
                svm_before == svm_after,
            )
        )
        row = {
            "user_id": "user_004",
            "enrollment_audio_count": enrollment["embedding_count"],
            "heldout_query_independent": "true",
            "centroid_created": str(centroid_created).lower(),
            "identified_user_id": sid["candidate_user_id"],
            "cosine_similarity": f"{float(sid['similarity']):.6f}",
            "schedule_owner_returned": result["speaker"]["candidate_user_id"],
            "svm_sha256_before": svm_before,
            "svm_sha256_after": svm_after,
            "svm_retrained": str(svm_before != svm_after).lower(),
            "execution_mode": "deterministic_embedding_contract",
            "passed": str(passed).lower(),
        }
    _write_csv(
        output_dir / "dynamic_enrollment_test_results.csv",
        [row],
        list(row),
    )
    return row


def main() -> int:
    results, metrics = run_system_suite()
    dynamic = run_dynamic_enrollment_test()
    passed = metrics["failed_count"] == 0 and dynamic["passed"] == "true"
    print(
        json.dumps(
            {"system": metrics, "dynamic_enrollment": dynamic, "passed": passed},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
