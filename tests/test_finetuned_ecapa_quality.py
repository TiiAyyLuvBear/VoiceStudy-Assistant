from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audio.preprocessing import preprocess_audio
from src.speaker.embedding import clear_embedding_extractor_cache, get_embedding_extractor
from src.speaker.evaluation import rates_at_threshold, verification_metrics
from src.utils.config import load_yaml_mapping, resolve_path


def _quality_config() -> tuple[dict, Path]:
    config, root = load_yaml_mapping("config.yaml")
    return config["speaker"]["quality_gate"], root


def test_finetuned_held_out_verification_quality_gate() -> None:
    quality, root = _quality_config()
    scores_path = resolve_path(quality["trial_scores_path"], root)
    metrics_path = resolve_path(quality["metrics_path"], root)
    scores = pd.read_csv(scores_path)
    reported = json.loads(metrics_path.read_text(encoding="utf-8"))
    labels = scores["label"].to_numpy(dtype=int)
    values = scores["score"].to_numpy(dtype=float)
    threshold = float(reported["test"]["threshold"])
    rates = rates_at_threshold(labels, values, threshold)
    curves = verification_metrics(labels, values)
    accepted = values >= threshold

    assert len(scores) == quality["expected_trials"]
    assert int(labels.sum()) == quality["expected_genuine_trials"]
    assert int((labels == 0).sum()) == quality["expected_impostor_trials"]
    assert int(np.count_nonzero(accepted != scores["accepted"].astype(bool))) == 0
    assert rates["far"] == pytest.approx(quality["expected_far"], abs=quality["metric_tolerance"])
    assert rates["frr"] == pytest.approx(quality["expected_frr"], abs=quality["metric_tolerance"])
    assert curves["eer"] == pytest.approx(quality["expected_eer"], abs=quality["curve_tolerance"])
    assert curves["min_dcf"] == pytest.approx(quality["expected_min_dcf"], abs=quality["curve_tolerance"])
    assert curves["eer"] <= quality["maximum_eer"]
    assert rates["far"] <= quality["maximum_far"]
    assert rates["frr"] <= quality["maximum_frr"]


@pytest.mark.skipif(
    os.environ.get("RUN_ECAPA_MODEL_TESTS") != "1",
    reason="Set RUN_ECAPA_MODEL_TESTS=1 for actual checkpoint inference",
)
def test_actual_finetuned_checkpoint_smoke() -> None:
    quality, root = _quality_config()
    audio_path = resolve_path(quality["smoke_audio_path"], root)
    audio, sample_rate = preprocess_audio(str(audio_path))
    clear_embedding_extractor_cache()
    extractor = get_embedding_extractor("config.yaml")
    embedding, dimension, latency_ms = extractor.extract(audio, sample_rate=sample_rate)

    assert dimension == 192
    assert embedding.shape == (192,)
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-6)
    assert latency_ms >= 0.0
