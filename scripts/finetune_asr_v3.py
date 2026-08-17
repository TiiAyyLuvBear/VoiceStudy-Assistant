"""Fine-tune Whisper Small with decoder-only LoRA using ASR v3 train/validation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.optimization import get_linear_schedule_with_warmup

from src.asr.finetuning import WhisperBatchBuilder, batched, epoch_rows, load_finetune_rows
from src.utils import canonical_csv_sha256, sha256_file


def _validation_loss(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    builder: WhisperBatchBuilder,
    *,
    batch_size: int,
) -> float:
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    with torch.inference_mode():
        for chunk in batched(rows, batch_size):
            batch = builder(chunk)
            labels = batch["labels"]
            tokens = int(labels.ne(-100).sum().item())
            outputs = model(**batch)
            weighted_loss += float(outputs.loss.item()) * tokens
            token_count += tokens
    return weighted_loss / token_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("models/cache/huggingface/openai-whisper-small"),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/v3/metadata/asr_finetune_train.csv"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/processed/v3/metadata/asr_finetune_validation.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("models/experimental/asr/v3")
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--maximum-seconds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("epochs, batch-size and gradient-accumulation must be positive")
    if not args.base_model.is_dir():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    train_rows = load_finetune_rows(args.train)
    validation_rows = load_finetune_rows(args.validation)
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_validation:
        validation_rows = validation_rows[: args.limit_validation]

    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        local_files_only=True,
        language="vi",
        task="transcribe",
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    base_model.generation_config.language = "vi"
    base_model.generation_config.task = "transcribe"
    base_model.generation_config.forced_decoder_ids = None
    base_model.freeze_encoder()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    # PEFT targets matching encoder modules too; v3 intentionally trains only
    # decoder LoRA while keeping the expensive encoder completely frozen.
    model.get_encoder().requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if not trainable_count:
        raise RuntimeError("No trainable LoRA parameters were created")

    builder = WhisperBatchBuilder(
        processor,
        model,
        maximum_seconds=args.maximum_seconds,
    )
    optimizer = AdamW(trainable, lr=args.learning_rate, foreach=False)
    batches_per_epoch = (len(train_rows) + args.batch_size - 1) // args.batch_size
    update_steps_per_epoch = (
        batches_per_epoch + args.gradient_accumulation - 1
    ) // args.gradient_accumulation
    total_update_steps = update_steps_per_epoch * args.epochs
    if args.max_steps:
        total_update_steps = min(total_update_steps, args.max_steps)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    best_dir = args.output_root / "best_adapter"
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    global_step = 0
    micro_step = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)

    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_batches = 0
        for chunk in batched(
            epoch_rows(train_rows, seed=args.seed, epoch=epoch), args.batch_size
        ):
            batch = builder(chunk)
            outputs = model(**batch)
            loss = outputs.loss
            (loss / args.gradient_accumulation).backward()
            running_loss += float(loss.detach().item())
            running_batches += 1
            micro_step += 1
            should_update = (
                micro_step % args.gradient_accumulation == 0
                or running_batches == batches_per_epoch
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                print(
                    f"epoch={epoch} step={global_step}/{total_update_steps} "
                    f"loss={float(loss.detach().item()):.6f}",
                    flush=True,
                )
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        validation_loss = _validation_loss(
            model,
            validation_rows,
            builder,
            batch_size=args.batch_size,
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / max(1, running_batches),
            "validation_loss": validation_loss,
        }
        history.append(epoch_record)
        checkpoint_dir = args.output_root / f"checkpoint-epoch-{epoch}"
        builder.restore_full_window()
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        processor.save_pretrained(checkpoint_dir)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            model.save_pretrained(best_dir, safe_serialization=True)
            processor.save_pretrained(best_dir)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        if stop:
            break

    builder.restore_full_window()
    summary = {
        "schema_version": 1,
        "status": "TRAINED_VALIDATION_SELECTED_NOT_TESTED",
        "dataset_version": "asr-v3",
        "method": "decoder-only LoRA on openai/whisper-small",
        "test_split_accessed": False,
        "base_model": {
            "path": args.base_model.as_posix(),
            "model_safetensors_sha256": sha256_file(
                args.base_model / "model.safetensors"
            ),
        },
        "datasets": {
            "train": {
                "path": args.train.as_posix(),
                "row_count": len(train_rows),
                "canonical_csv_sha256": canonical_csv_sha256(args.train),
            },
            "validation": {
                "path": args.validation.as_posix(),
                "row_count": len(validation_rows),
                "canonical_csv_sha256": canonical_csv_sha256(args.validation),
            },
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": 0.05,
            "maximum_training_window_seconds": args.maximum_seconds,
            "seed": args.seed,
            "trainable_parameter_count": trainable_count,
            "max_steps": args.max_steps,
        },
        "selection": {
            "metric": "validation_loss",
            "lower_is_better": True,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_adapter_path": best_dir.as_posix(),
        },
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = args.output_root / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
