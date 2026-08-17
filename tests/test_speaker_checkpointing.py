from __future__ import annotations

import json

import torch

from src.speaker.checkpointing import (
    CheckpointManager,
    balanced_score,
    checkpoint_payload,
    rank_metrics,
)


def test_task_ranking_and_tie_breaks_are_deterministic():
    assert rank_metrics("closed", {"macro_f1": .8, "accuracy": .9}, 4) < rank_metrics("closed", {"macro_f1": .8, "accuracy": .8}, 3)
    assert rank_metrics("verification", {"eer": .1, "min_dcf": .3}, 2) < rank_metrics("verification", {"eer": .2, "min_dcf": .1}, 1)
    assert rank_metrics("open", {"known_unknown_auroc": .9, "dir_at_far_1pct": .7}, 2) < rank_metrics("open", {"known_unknown_auroc": .9, "dir_at_far_1pct": .6}, 1)


def test_balanced_eligibility_and_fallback():
    metrics = {"closed_macro_f1": .8, "verification_eer": .1, "open_auroc": .9}
    assert balanced_score(metrics) == (.8 + .9 + .9) / 3
    assert rank_metrics("balanced", metrics, 2, frozen_closed_macro_f1=.81, max_closed_f1_drop=.005) is None
    assert rank_metrics("balanced", metrics, 2, frozen_closed_macro_f1=.8, max_closed_f1_drop=.005) is not None


def test_manager_overwrites_only_on_improvement_and_keeps_five_files(tmp_path):
    manager = CheckpointManager(tmp_path, max_checkpoints=5)
    payload = checkpoint_payload({"w": torch.tensor([1.])}, 1, "closed", {"macro_f1": .5}, {"x": 1})
    assert manager.save_if_improved("closed", payload, {"macro_f1": .5, "accuracy": .5}, 1)
    assert not manager.save_if_improved("closed", checkpoint_payload({"w": torch.tensor([2.])}, 2, "closed", {"macro_f1": .4}, {}), {"macro_f1": .4, "accuracy": .9}, 2)
    assert torch.equal(torch.load(tmp_path / "best_closed.pt", weights_only=True)["encoder"]["w"], torch.tensor([1.]))
    for task in ("closed", "verification", "open", "balanced", "latest"):
        (tmp_path / f"{task}.pt").write_bytes(b"x")
    manager.enforce_limit()
    assert len(list(tmp_path.glob("*.pt"))) <= 5
    registry = json.loads((tmp_path / "checkpoint_registry.json").read_text())
    assert registry["closed"]["epoch"] == 1


def test_payload_is_weight_only():
    payload = checkpoint_payload({"w": torch.tensor([1.])}, 1, "closed", {}, {})
    assert set(payload) == {"encoder", "classifier", "epoch", "task", "validation_metrics", "training_config"}
    assert "optimizer" not in payload and "predictions" not in payload
