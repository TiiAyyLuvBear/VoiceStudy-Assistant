"""Merge the validation-selected ASR v3 LoRA adapter into Whisper Small."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.utils import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("models/cache/huggingface/openai-whisper-small"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("models/experimental/asr/v3/best_adapter"),
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("models/experimental/asr/v3/training_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/experimental/asr/v3/hf_merged"),
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite merged model: {args.output_dir}")
    training = json.loads(args.training_summary.read_text(encoding="utf-8"))
    if training.get("status") != "TRAINED_VALIDATION_SELECTED_NOT_TESTED":
        raise ValueError("Training summary is not in the pre-test selected state")
    if training.get("test_split_accessed") is not False:
        raise ValueError("Training summary indicates test access")

    processor = WhisperProcessor.from_pretrained(
        args.adapter,
        local_files_only=True,
        language="vi",
        task="transcribe",
    )
    base = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.generation_config.language = "vi"
    merged.generation_config.task = "transcribe"
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)

    summary = {
        "schema_version": 1,
        "status": "MERGED_NOT_LOCKED_NOT_TESTED",
        "base_model": args.base_model.as_posix(),
        "adapter": args.adapter.as_posix(),
        "training_summary": args.training_summary.as_posix(),
        "best_epoch": training["selection"]["best_epoch"],
        "test_split_accessed": False,
        "model_path": args.output_dir.as_posix(),
        "checksums": {
            "base_model": sha256_file(args.base_model / "model.safetensors"),
            "adapter": sha256_file(args.adapter / "adapter_model.safetensors"),
            "training_summary": sha256_file(args.training_summary),
            "merged_model": sha256_file(args.output_dir / "model.safetensors"),
        },
    }
    summary_path = args.output_dir.parent / "export_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
