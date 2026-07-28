"""Nhập một câu mới và chạy toàn bộ NLU baseline."""

from __future__ import annotations

import argparse
import json
import sys

from src.nlu.command_parser import parse_command
from src.nlu.text_normalizer import normalize_text


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="Vietnamese command")
    parser.add_argument("--reference-date", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    transcript = " ".join(args.text).strip()
    if not transcript:
        transcript = input("Nhập câu lệnh: ").strip()

    payload = {
        "transcript": transcript,
        "normalized_transcript": normalize_text(transcript),
        **parse_command(transcript, args.reference_date),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
