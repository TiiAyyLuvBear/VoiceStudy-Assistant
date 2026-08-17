from __future__ import annotations

import json
from pathlib import Path

from src.utils import sha256_file


TEST_ROOT = Path("experiments/v2/test")
CONFIG = Path("experiments/v2/speaker_test_config.json")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_speaker_v2_config_locks_reproducibility_inputs() -> None:
    config = _json(CONFIG)

    assert config["dataset_version"] == "v2"
    assert config["random_seed"] == 42
    assert sha256_file(config["split_manifest"]["path"]) == config[
        "split_manifest"
    ]["sha256"]
    assert sha256_file(config["ecapa"]["checkpoint"]) == config["ecapa"][
        "checkpoint_sha256"
    ]
    assert sha256_file(config["ecapa"]["hyperparameters"]) == config["ecapa"][
        "hyperparameters_sha256"
    ]
    assert sha256_file(config["svm_closed_set"]["model_path"]) == config[
        "svm_closed_set"
    ]["model_sha256"]
    assert config["svm_closed_set"]["selected_C"] == 0.1
    assert config["speaker_disjoint_sid"]["threshold_tuned_on_v2_test"] is False
    assert (
        config["speaker_disjoint_verification"]["threshold_tuned_on_v2_test"]
        is False
    )
    assert (
        config["speaker_disjoint_verification"][
            "test_eer_threshold_used_for_system"
        ]
        is False
    )


def test_speaker_v2_test_metrics_are_complete() -> None:
    svm = _json(TEST_ROOT / "svm_closed_set_metrics.json")
    cosine = _json(TEST_ROOT / "cosine_closed_set_metrics.json")
    sid = _json(TEST_ROOT / "speaker_disjoint_sid_test_metrics.json")
    fixed = _json(
        TEST_ROOT
        / "speaker_disjoint_verification_fixed_threshold_metrics.json"
    )
    curve = _json(
        TEST_ROOT / "speaker_disjoint_verification_curve_metrics.json"
    )

    assert svm["test_sample_count"] == cosine["test_sample_count"] == 128
    assert svm["unknown_threshold_applied"] is False
    assert cosine["unknown_threshold_applied"] is False
    assert svm["metrics"]["accuracy"] == 1.0
    assert sid["known_query"]["count"] == 200
    assert sid["unknown_query"]["count"] == 109
    assert sid["open_set"]["total_queries"] == 309
    assert sid["threshold_tuning_on_v2_test"] is False
    assert fixed["test_trial_count"] == 2472
    assert fixed["threshold_tuned_on_v2_test"] is False
    assert curve["eer_threshold_used_for_system"] is False
    assert curve["system_threshold"] == fixed["threshold"]


def test_asr_and_nlu_v2_results_have_full_coverage() -> None:
    validation = _json(Path("reports/asr/v2/asr_validation_metrics.json"))
    test = _json(Path("reports/asr/v2/asr_test_metrics.json"))
    nlu = _json(Path("reports/nlu/v2/intent_test_metrics.json"))

    assert validation["sample_count"] == validation["successful_count"] == 322
    assert validation["failure_count"] == 0
    assert test["sample_count"] == test["successful_count"] == 249
    assert test["failure_count"] == 0
    assert nlu["expected_sample_count"] == nlu["audio_successful_count"] == 30
    assert nlu["audio_failure_count"] == 0
    assert all("/v2/" in path for path in nlu["artifacts"].values())
