"""Chạy kiểm tra thủ công Whisper trên một hoặc nhiều audio tiếng Việt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.asr.whisper_model import get_asr_model
from src.nlu.text_normalizer import normalize_text


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Transcribe Vietnamese sample audio with Whisper Small"
    )
    parser.add_argument("audio", nargs="+", help="One or more audio paths")
    parser.add_argument("--config", default="config.yaml", help="YAML config path")
    args = parser.parse_args()

    asr = get_asr_model(args.config)
    all_successful = True
    for value in args.audio:
        path = Path(value)
        result = asr.transcribe(path)
        payload = {
            "audio_path": str(path),
            **result,
            "normalized_transcript": (
                normalize_text(result["transcript"]) if result["success"] else ""
            ),
        }
        print(json.dumps(payload, ensure_ascii=False))
        all_successful = all_successful and result["success"]

    return 0 if all_successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
