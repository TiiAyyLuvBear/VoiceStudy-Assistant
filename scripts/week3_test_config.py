"""Freeze and verify the reproducible ASR/NLU Week 3 test configuration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_OUTPUT = Path("reports/asr_nlu_test_config.json")
LOCKED_PATHS = (
    Path("config.yaml"),
    Path("data/metadata/asr_test.csv"),
    Path("data/metadata/command_test.csv"),
    Path("data/commands/command_audio_manifest.csv"),
    Path("src/asr/metrics.py"),
    Path("src/asr/text_normalizer.py"),
    Path("src/asr/whisper_model.py"),
    Path("src/nlu/command_parser.py"),
    Path("src/nlu/entity_extractor.py"),
    Path("src/nlu/intent_classifier.py"),
    Path("src/nlu/intent_schema.py"),
    Path("src/nlu/missing_fields.py"),
    Path("src/nlu/text_normalizer.py"),
    Path("src/pipeline/asr_nlu.py"),
    Path("scripts/evaluate_asr.py"),
    Path("scripts/evaluate_week3_nlu.py"),
    Path("scripts/week3_test_config.py"),
)
PACKAGE_NAMES = ("faster-whisper", "ctranslate2", "PyYAML")
APPLICATION_LABELS = (
    "GET_TIME",
    "VIEW_SCHEDULE",
    "ADD_SCHEDULE",
    "VIEW_PRIVATE_NOTE",
    "OUT_OF_SCOPE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_count(path: Path, *, split: str | None = None) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        return sum(split is None or row.get("split") == split for row in rows)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _model_cache_revision(config: dict[str, Any]) -> str | None:
    download_root = Path(config["asr"]["download_root"])
    model_size = str(config["asr"]["model_size"])
    reference = (
        download_root
        / f"models--Systran--faster-whisper-{model_size}"
        / "refs"
        / "main"
    )
    return reference.read_text(encoding="utf-8").strip() if reference.is_file() else None


def build_snapshot() -> dict[str, Any]:
    missing = [path.as_posix() for path in LOCKED_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing paths: {missing}")

    config_path = Path("config.yaml")
    with config_path.open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)

    command_manifest = Path("data/commands/command_audio_manifest.csv")
    with command_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        test_audio = [
            row for row in csv.DictReader(stream) if row.get("split") == "test"
        ]
    ready_audio = [
        row
        for row in test_audio
        if row.get("status") == "recorded"
        and row.get("audio_path")
        and Path(row["audio_path"]).is_file()
    ]

    dirty = _git_value("status", "--porcelain")
    return {
        "schema_version": 1,
        "purpose": "Week 3 held-out ASR and application command analysis",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": {
            "held_out_test": True,
            "rule_or_config_tuning_after_freeze_allowed": False,
            "rerun_to_improve_reported_results_allowed": False,
            "missing_command_audio_is_reported_not_imputed": True,
            "speech_massive_source_intent_is_application_ground_truth": False,
            "label_note": (
                "The five application labels are project-defined labels; source_intent "
                "in the Speech-MASSIVE-derived ASR split is informational only."
            ),
        },
        "application_labels": list(APPLICATION_LABELS),
        "reference_date": "2026-07-28",
        "asr": config["asr"],
        "model_cache_revision": _model_cache_revision(config),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "worktree_clean_before_snapshot": not bool(dirty),
            "worktree_status_before_snapshot": dirty.splitlines() if dirty else [],
        },
        "datasets": {
            "asr_test": {
                "path": "data/metadata/asr_test.csv",
                "row_count": _csv_count(Path("data/metadata/asr_test.csv")),
            },
            "command_test": {
                "path": "data/metadata/command_test.csv",
                "row_count": _csv_count(Path("data/metadata/command_test.csv")),
            },
            "command_audio_test": {
                "manifest": command_manifest.as_posix(),
                "expected_count": len(test_audio),
                "ready_count_at_freeze": len(ready_audio),
                "missing_count_at_freeze": len(test_audio) - len(ready_audio),
            },
        },
        "locked_files": {
            path.as_posix(): sha256_file(path) for path in LOCKED_PATHS
        },
    }


def write_snapshot(output: Path, *, force: bool = False) -> dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite frozen test configuration: {output}. "
            "Use --force only before any official test run."
        )
    snapshot = build_snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot


def verify_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen test configuration does not exist: {path}")
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for path_value, expected_hash in snapshot.get("locked_files", {}).items():
        source = Path(path_value)
        if not source.is_file():
            mismatches.append(f"missing:{path_value}")
            continue
        actual_hash = sha256_file(source)
        if actual_hash != expected_hash:
            mismatches.append(f"sha256:{path_value}")
    if mismatches:
        raise RuntimeError(
            "Frozen Week 3 configuration no longer matches the workspace: "
            + ", ".join(mismatches)
        )
    return snapshot


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.verify:
        snapshot = verify_snapshot(args.output)
        print(
            f"Verified {len(snapshot['locked_files'])} frozen files: {args.output}"
        )
        return 0

    snapshot = write_snapshot(args.output, force=args.force)
    print(
        f"Frozen {len(snapshot['locked_files'])} files to {args.output}; "
        f"command audio ready="
        f"{snapshot['datasets']['command_audio_test']['ready_count_at_freeze']}/"
        f"{snapshot['datasets']['command_audio_test']['expected_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
