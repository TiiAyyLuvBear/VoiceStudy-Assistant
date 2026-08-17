"""Evaluate real application SID/SV on held-out command audio."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.speaker.application import identify_application_user, verify_speaker
from src.speaker.embedding import ECAPAEmbeddingExtractor
from src.audio.preprocessing import preprocess_audio


DEMO_DATA = PROJECT_ROOT / "experiments/system/demo_enrollment_data.csv"
SID_OUTPUT = PROJECT_ROOT / "experiments/system/application_sid_heldout_results.csv"
SV_OUTPUT = PROJECT_ROOT / "experiments/system/application_sv_results.csv"


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_demo_application_speaker() -> dict:
    with DEMO_DATA.open("r", encoding="utf-8-sig", newline="") as stream:
        heldout = [
            row for row in csv.DictReader(stream) if row["role"] == "HELDOUT_QUERY"
        ]
    # Finish Librosa lazy imports before SpeechBrain registers optional lazy
    # integrations (notably k2, which ECAPA inference does not require).
    preprocess_audio(PROJECT_ROOT / heldout[0]["audio_path"])
    extractor = ECAPAEmbeddingExtractor.from_config()
    sid_rows = []
    for row in heldout:
        result = identify_application_user(
            PROJECT_ROOT / row["audio_path"],
            extractor=extractor,
        )
        sid_rows.append(
            {
                "recording_id": row["recording_id"],
                "expected_user_id": row["user_id"],
                "candidate_user_id": result["candidate_user_id"] or "",
                "cosine_similarity": result["similarity"],
                "application_sid_threshold": result["unknown_threshold"],
                "status": result["status"],
                "correct": str(
                    result["candidate_user_id"] == row["user_id"]
                    and result["status"] == "KNOWN"
                ).lower(),
                "latency_ms": result["latency_ms"],
            }
        )

    users = sorted({row["user_id"] for row in heldout})
    first_query = {user_id: next(row for row in heldout if row["user_id"] == user_id) for user_id in users}
    sv_rows = []
    for index, user_id in enumerate(users):
        row = first_query[user_id]
        audio_path = PROJECT_ROOT / row["audio_path"]
        for trial_type, candidate_id, expected in (
            ("GENUINE", user_id, True),
            ("IMPOSTOR", users[(index + 1) % len(users)], False),
        ):
            result = verify_speaker(
                audio_path,
                candidate_id,
                extractor=extractor,
            )
            sv_rows.append(
                {
                    "recording_id": row["recording_id"],
                    "source_user_id": user_id,
                    "candidate_user_id": candidate_id,
                    "trial_type": trial_type,
                    "similarity": result["similarity"],
                    "application_verification_threshold": result[
                        "verification_threshold"
                    ],
                    "verified": str(result["verified"]).lower(),
                    "expected_verified": str(expected).lower(),
                    "correct": str(result["verified"] is expected).lower(),
                    "latency_ms": result["latency_ms"],
                }
            )
    _write(SID_OUTPUT, sid_rows)
    _write(SV_OUTPUT, sv_rows)
    summary = {
        "sid_count": len(sid_rows),
        "sid_correct": sum(row["correct"] == "true" for row in sid_rows),
        "sv_count": len(sv_rows),
        "sv_correct": sum(row["correct"] == "true" for row in sv_rows),
        "sid_threshold_source": "frozen experimental validation fallback",
        "sv_threshold_source": "application held-out validation",
        "threshold_tuned_on_v2_test": False,
    }
    return summary


def main() -> int:
    summary = evaluate_demo_application_speaker()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
