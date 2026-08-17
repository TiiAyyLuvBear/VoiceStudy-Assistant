"""Validation-ranked, weight-only rolling ECAPA checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch


TASK_PATHS = {
    "closed": "best_closed.pt",
    "verification": "best_verification.pt",
    "open": "best_open.pt",
    "balanced": "best_balanced.pt",
    "latest": "latest.pt",
}


def balanced_score(metrics: Mapping[str, float]) -> float:
    return (float(metrics["closed_macro_f1"]) + (1.0 - float(metrics["verification_eer"])) + float(metrics["open_auroc"])) / 3.0


def rank_metrics(task: str, metrics: Mapping[str, float], epoch: int, *, frozen_closed_macro_f1: float | None = None, max_closed_f1_drop: float = .005):
    """Return comparable rank; lower wins. Earlier epoch wins exact ties."""
    if task == "closed":
        return (-float(metrics["macro_f1"]), -float(metrics["accuracy"]), int(epoch))
    if task == "verification":
        return (float(metrics["eer"]), float(metrics["min_dcf"]), int(epoch))
    if task == "open":
        return (-float(metrics["known_unknown_auroc"]), -float(metrics["dir_at_far_1pct"]), int(epoch))
    if task == "balanced":
        if frozen_closed_macro_f1 is not None and float(metrics["closed_macro_f1"]) < frozen_closed_macro_f1 - max_closed_f1_drop:
            return None
        return (-balanced_score(metrics), int(epoch))
    raise ValueError(f"Unknown checkpoint task: {task}")


def checkpoint_payload(encoder: Mapping[str, Any], epoch: int, task: str, validation_metrics: Mapping[str, Any], training_config: Mapping[str, Any], classifier: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "encoder": dict(encoder),
        "classifier": dict(classifier or {}),
        "epoch": int(epoch),
        "task": str(task),
        "validation_metrics": dict(validation_metrics),
        "training_config": dict(training_config),
    }


class CheckpointManager:
    def __init__(self, directory: str | Path, max_checkpoints: int = 5):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = int(max_checkpoints)
        self.registry_path = self.directory / "checkpoint_registry.json"
        self.registry = json.loads(self.registry_path.read_text()) if self.registry_path.exists() else {}

    def save_if_improved(self, task: str, payload: Mapping[str, Any], metrics: Mapping[str, float], epoch: int, *, reason: str = "validation improvement", frozen_closed_macro_f1: float | None = None, max_closed_f1_drop: float = .005) -> bool:
        rank = rank_metrics(task, metrics, epoch, frozen_closed_macro_f1=frozen_closed_macro_f1, max_closed_f1_drop=max_closed_f1_drop)
        if rank is None or (task in self.registry and tuple(self.registry[task]["rank"]) <= tuple(rank)):
            return False
        path = self.directory / TASK_PATHS[task]
        torch.save(dict(payload), path)
        self.registry[task] = {"task": task, "epoch": int(epoch), "validation_metrics": dict(metrics), "path": str(path), "selection_reason": reason, "rank": list(rank)}
        self._write_registry()
        self.enforce_limit()
        return True

    def record_latest(self, payload: Mapping[str, Any], epoch: int) -> None:
        torch.save(dict(payload), self.directory / TASK_PATHS["latest"])
        self.registry["latest"] = {"task": "latest", "epoch": int(epoch), "path": str(self.directory / TASK_PATHS["latest"]), "selection_reason": "latest epoch"}
        self._write_registry()
        self.enforce_limit()

    def enforce_limit(self) -> None:
        allowed = {TASK_PATHS[key] for key in TASK_PATHS}
        for path in self.directory.glob("*.pt"):
            if path.name not in allowed:
                path.unlink()
        paths = sorted(self.directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths[self.max_checkpoints:]:
            path.unlink()

    def _write_registry(self) -> None:
        self.registry_path.write_text(json.dumps(self.registry, indent=2, default=str) + "\n", encoding="utf-8")
