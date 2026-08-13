from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_system_tests import (
    run_dynamic_enrollment_test,
    run_system_suite,
)


def test_system_suite_generates_30_passing_cases(tmp_path: Path) -> None:
    results, metrics = run_system_suite(tmp_path)
    assert len(results) == 30
    assert metrics["passed_count"] == 30
    assert metrics["task_success_rate"] == 1.0
    assert all(row["database_integrity_pass"] == "true" for row in results)

    with (tmp_path / "system_test_cases.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 30
    stored = json.loads(
        (tmp_path / "task_success_rate.json").read_text(encoding="utf-8")
    )
    assert stored["execution_mode"] == "deterministic_integration_contract"
    for filename in (
        "public_test_results.csv",
        "sid_personalization_results.csv",
        "add_schedule_results.csv",
        "private_access_results.csv",
        "unknown_access_results.csv",
        "impostor_results.csv",
        "out_of_scope_system_results.csv",
    ):
        assert (tmp_path / filename).is_file()


def test_dynamic_enrollment_does_not_retrain_svm(tmp_path: Path) -> None:
    result = run_dynamic_enrollment_test(tmp_path)
    assert result["passed"] == "true"
    assert result["user_id"] == "user_004"
    assert result["enrollment_audio_count"] == 5
    assert result["identified_user_id"] == "user_004"
    assert result["svm_retrained"] == "false"
