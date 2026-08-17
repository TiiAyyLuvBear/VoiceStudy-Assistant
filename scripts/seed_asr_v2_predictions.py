"""Seed ASR v2 resume files with matching successful v1 predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]
V2_METADATA = ROOT / "data/processed/v2/metadata"
V1_REPORTS = ROOT / "reports/asr"
V2_REPORTS = ROOT / "reports/asr/v2"
PREDICTION_FIELDS = (
    "audio_id",
    "audio_path",
    "original_split",
    "reference_transcript",
    "hypothesis",
    "normalized_reference",
    "normalized_hypothesis",
    "wer",
    "cer",
    "latency_ms",
    "model",
    "language",
    "success",
    "error",
)


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _seed(split: str) -> dict[str, object]:
    split_path = V2_METADATA / f"asr_{split}.csv"
    source_path = V1_REPORTS / f"asr_{split}_predictions.csv"
    target_path = V2_REPORTS / f"asr_{split}_predictions.csv"
    source = {
        row["audio_id"]: row
        for row in _read(source_path)
        if row.get("success", "").lower() == "true"
    }
    existing = {
        row["audio_id"]: row
        for row in _read(target_path)
        if row.get("success", "").lower() == "true"
    }
    split_rows = _read(split_path)
    seeded: list[dict[str, str]] = []
    reused_from_v1 = 0
    reused_existing = 0
    for row in split_rows:
        audio_id = row["audio_id"]
        prediction = existing.get(audio_id)
        if prediction is not None:
            reused_existing += 1
        else:
            prediction = source.get(audio_id)
            if prediction is not None:
                reused_from_v1 += 1
        if prediction is None:
            continue
        if prediction.get("reference_transcript") != row["reference_transcript"]:
            raise ValueError(f"Reference mismatch for cached audio_id={audio_id}")
        seeded.append({field: prediction.get(field, "") for field in PREDICTION_FIELDS})

    V2_REPORTS.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(target_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(seeded)
    temporary.replace(target_path)
    return {
        "split": split,
        "split_path": split_path.relative_to(ROOT).as_posix(),
        "split_sha256": sha256_file(split_path),
        "source_prediction_path": source_path.relative_to(ROOT).as_posix(),
        "source_prediction_sha256": sha256_file(source_path),
        "target_prediction_path": target_path.relative_to(ROOT).as_posix(),
        "v2_sample_count": len(split_rows),
        "seeded_count": len(seeded),
        "reused_from_v1": reused_from_v1,
        "reused_existing_v2": reused_existing,
    }


def main() -> int:
    payload = {
        "policy": (
            "Only successful predictions for the same audio_id and exact reference "
            "transcript are reused. Full v2 metrics are recalculated by evaluate_asr."
        ),
        "model_and_normalization_changed": False,
        "splits": [_seed("validation"), _seed("test")],
    }
    provenance = V2_REPORTS / "prediction_resume_provenance.json"
    provenance.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
