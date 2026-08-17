from __future__ import annotations

import inspect

from src.speaker import evaluation


def test_validation_entry_points_exist_and_have_no_test_argument():
    for name in ("evaluate_closed_validation", "evaluate_verification_validation", "evaluate_open_validation"):
        function = getattr(evaluation, name)
        assert "test" not in inspect.signature(function).parameters


def test_validation_source_does_not_reference_test_protocols():
    source = inspect.getsource(evaluation.evaluate_closed_validation)
    assert "test" not in source.lower()
