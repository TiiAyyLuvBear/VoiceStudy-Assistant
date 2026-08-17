"""Download and build compact VoxVietnam three-task protocols."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_voxvietnam_verification_dataset import build_from_huggingface


OUTPUT_ROOT = Path("data/datasets/voxvietnam_ecapa_three_task_v1")


def main() -> int:
    # Do not call load_dataset() without a split: that downloads train,
    # train_small, and test, although train_small is already a train subset.
    manifest = build_from_huggingface(
        output_root=OUTPUT_ROOT,
        training_source_split="train_small",
        test_source_split="test",
        train_speakers=230,
        validation_speakers=50,
        test_speakers=50,
        train_audio_per_speaker=30,
        evaluation_audio_per_speaker=15,
        enrollment_audio_per_speaker=5,
        negative_trials_per_query=5,
        closed_set_train_audio_per_speaker=20,
        closed_set_validation_audio_per_speaker=5,
        open_set_known_speakers=25,
        max_bytes=10 * 1024**3,
        seed=42,
    )
    print(
        json.dumps(
            {
                "output": OUTPUT_ROOT.as_posix(),
                "total_audio": manifest["total_audio"],
                "total_audio_gib": round(
                    manifest["total_audio_bytes"] / 1024**3,
                    3,
                ),
                "splits": manifest["splits"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
