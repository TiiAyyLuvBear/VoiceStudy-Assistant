"""Đánh giá intent accuracy và entity exact match trên command CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.nlu.command_parser import parse_command


def evaluate(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    intent_correct = 0
    entity_correct = 0
    mismatches: list[dict[str, object]] = []
    for row in rows:
        expected_entities = {
            name: row[f"expected_{name}"]
            for name in ("title", "date", "time")
            if row[f"expected_{name}"]
        }
        result = parse_command(row["transcript"], row.get("reference_date") or None)
        intent_matches = result["intent"] == row["intent"]
        entities_match = result["entities"] == expected_entities
        intent_correct += int(intent_matches)
        entity_correct += int(entities_match)
        if not intent_matches or not entities_match:
            mismatches.append(
                {
                    "command_id": row["command_id"],
                    "transcript": row["transcript"],
                    "expected_intent": row["intent"],
                    "predicted_intent": result["intent"],
                    "expected_entities": expected_entities,
                    "predicted_entities": result["entities"],
                }
            )

    total = len(rows)
    return {
        "dataset": str(path),
        "total": total,
        "intent_accuracy": intent_correct / total if total else 0.0,
        "entity_exact_match": entity_correct / total if total else 0.0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()

    reports = [evaluate(path) for path in args.datasets]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 1 if any(report["mismatch_count"] for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
