"""Small unit checks for the ASR v4 validation-selection protocol."""

from scripts.finalize_asr_v4 import _change
from scripts.finetune_asr_v4 import _is_better


def test_checkpoint_selection_prioritizes_wer_then_cer() -> None:
    best = {"wer": 0.20, "cer": 0.12}
    assert _is_better({"wer": 0.19, "cer": 0.50}, best)
    assert _is_better({"wer": 0.20, "cer": 0.11}, best)
    assert not _is_better({"wer": 0.20, "cer": 0.13}, best)


def test_metric_change_reports_relative_improvement() -> None:
    result = _change(0.20, 0.15)
    assert abs(result["percentage_point_change"] + 5.0) < 1e-12
    assert abs(result["relative_improvement"] - 0.25) < 1e-12
