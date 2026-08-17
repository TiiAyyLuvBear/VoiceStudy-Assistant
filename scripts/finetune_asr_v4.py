"""Fine-tune Whisper Small with wider decoder LoRA on ASR v4."""

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
from src.asr.metrics import calculate_corpus_error_rates
from src.utils import canonical_csv_sha256, sha256_file


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def _validation_loss(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    builder: WhisperBatchBuilder,
    *,
    batch_size: int,
    use_amp: bool,
) -> float:
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    with torch.inference_mode():
        for chunk in batched(rows, batch_size):
            batch = builder(chunk)
            labels = batch["labels"]
            tokens = int(labels.ne(-100).sum().item())
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
            weighted_loss += float(outputs.loss.item()) * tokens
            token_count += tokens
    return weighted_loss / token_count


def _validation_error_rates(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    builder: WhisperBatchBuilder,
    processor: WhisperProcessor,
    *,
    batch_size: int,
    max_new_tokens: int,
    use_amp: bool,
) -> dict[str, float | int]:
    """Greedy validation decode used exclusively for checkpoint selection."""

    model.eval()
    model.config.use_cache = True
    references: list[str] = []
    hypotheses: list[str] = []
    with torch.inference_mode():
        for index, chunk in enumerate(batched(rows, batch_size), start=1):
            batch = builder(chunk)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                generated = model.generate(
                    input_features=batch["input_features"],
                    attention_mask=batch["attention_mask"],
                    language="vi",
                    task="transcribe",
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )
            hypotheses.extend(processor.batch_decode(generated, skip_special_tokens=True))
            references.extend(row["transcript"] for row in chunk)
            print(
                f"validation_decode batch={index} samples={len(references)}/{len(rows)}",
                flush=True,
            )
    model.config.use_cache = False
    return calculate_corpus_error_rates(references, hypotheses)


def _is_better(metrics: dict[str, float | int], best: dict[str, float | int] | None) -> bool:
    if best is None:
        return True
    return (float(metrics["wer"]), float(metrics["cer"])) < (
        float(best["wer"]),
        float(best["cer"]),
    )


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
        default=Path("data/processed/v4/metadata/asr_finetune_train.csv"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/processed/v4/metadata/asr_finetune_validation.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("models/experimental/asr/v4")
    )
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--maximum-seconds", type=int, default=16)
    parser.add_argument("--validation-max-new-tokens", type=int, default=128)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--mixed-precision", choices=("auto", "none", "fp16"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    args = parser.parse_args()

    if not 1 <= args.min_epochs <= args.max_epochs:
        parser.error("Require 1 <= min-epochs <= max-epochs")
    if min(
        args.batch_size,
        args.validation_batch_size,
        args.gradient_accumulation,
        args.early_stopping_patience,
    ) < 1:
        parser.error("Batch sizes, accumulation and patience must be positive")
    if args.lora_rank != 32:
        parser.error("ASR v4 protocol requires LoRA rank 32")
    if not args.base_model.is_dir():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")

    summary_path = args.output_root / "training_summary.json"
    if summary_path.exists() or (args.output_root / "best_adapter").exists():
        raise FileExistsError(f"Refusing to overwrite an existing v4 training: {args.output_root}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    use_amp = args.mixed_precision == "fp16" or (
        args.mixed_precision == "auto" and device.type == "cuda"
    )
    if use_amp and device.type != "cuda":
        raise ValueError("FP16 mixed precision requires CUDA")

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
        target_modules=list(TARGET_MODULES),
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    # PEFT finds matching modules in both halves. V4 deliberately remains
    # decoder-only so this wider adapter is practical on the current CPU host.
    model.get_encoder().requires_grad_(False)
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if not trainable_count:
        raise RuntimeError("No trainable LoRA parameters were created")

    builder = WhisperBatchBuilder(
        processor, model, maximum_seconds=args.maximum_seconds, device=device
    )
    optimizer = AdamW(trainable, lr=args.learning_rate, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batches_per_epoch = (len(train_rows) + args.batch_size - 1) // args.batch_size
    update_steps_per_epoch = (
        batches_per_epoch + args.gradient_accumulation - 1
    ) // args.gradient_accumulation
    total_update_steps = update_steps_per_epoch * args.max_epochs
    if args.max_steps:
        total_update_steps = min(total_update_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_update_steps * args.warmup_ratio),
        num_training_steps=total_update_steps,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    best_dir = args.output_root / "best_adapter"
    history: list[dict[str, object]] = []
    best_metrics: dict[str, float | int] | None = None
    best_epoch = 0
    non_improving_epochs = 0
    global_step = 0
    started = time.time()
    stop_reason = "maximum_epochs"
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        model.config.use_cache = False
        running_loss = 0.0
        running_batches = 0
        reached_max_steps = False
        for chunk in batched(
            epoch_rows(train_rows, seed=args.seed, epoch=epoch), args.batch_size
        ):
            batch = builder(chunk)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss / args.gradient_accumulation).backward()
            running_loss += float(loss.detach().item())
            running_batches += 1
            should_update = (
                running_batches % args.gradient_accumulation == 0
                or running_batches == batches_per_epoch
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                print(
                    f"epoch={epoch} step={global_step}/{total_update_steps} "
                    f"loss={float(loss.detach().item()):.6f}",
                    flush=True,
                )
                if args.max_steps and global_step >= args.max_steps:
                    reached_max_steps = True
                    break

        validation_loss = _validation_loss(
            model,
            validation_rows,
            builder,
            batch_size=args.validation_batch_size,
            use_amp=use_amp,
        )
        validation_metrics = _validation_error_rates(
            model,
            validation_rows,
            builder,
            processor,
            batch_size=args.validation_batch_size,
            max_new_tokens=args.validation_max_new_tokens,
            use_amp=use_amp,
        )
        improved = _is_better(validation_metrics, best_metrics)
        if improved:
            best_metrics = validation_metrics
            best_epoch = epoch
            non_improving_epochs = 0
        else:
            non_improving_epochs += 1

        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / max(1, running_batches),
            "validation_loss": validation_loss,
            "validation": validation_metrics,
            "improved": improved,
            "non_improving_epochs": non_improving_epochs,
        }
        history.append(epoch_record)
        checkpoint_dir = args.output_root / f"checkpoint-epoch-{epoch}"
        builder.restore_full_window()
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        processor.save_pretrained(checkpoint_dir)
        if improved:
            model.save_pretrained(best_dir, safe_serialization=True)
            processor.save_pretrained(best_dir)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

        if reached_max_steps:
            stop_reason = "max_steps"
            break
        if (
            epoch >= args.min_epochs
            and non_improving_epochs >= args.early_stopping_patience
        ):
            stop_reason = "early_stopping_validation_wer_cer"
            break

    builder.restore_full_window()
    assert best_metrics is not None
    summary = {
        "schema_version": 1,
        "status": "TRAINED_VALIDATION_SELECTED_NOT_TESTED",
        "dataset_version": "asr-v4",
        "method": "decoder-only wider LoRA on openai/whisper-small",
        "test_split_accessed": False,
        "base_model": {
            "path": args.base_model.as_posix(),
            "model_safetensors_sha256": sha256_file(args.base_model / "model.safetensors"),
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
            "minimum_epochs": args.min_epochs,
            "maximum_epochs": args.max_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "batch_size": args.batch_size,
            "validation_batch_size": args.validation_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(TARGET_MODULES),
            "encoder_frozen": True,
            "gradient_checkpointing": args.gradient_checkpointing,
            "device": device.type,
            "mixed_precision": "fp16" if use_amp else "none",
            "maximum_training_window_seconds": args.maximum_seconds,
            "validation_decode": {
                "num_beams": 1,
                "max_new_tokens": args.validation_max_new_tokens,
                "language": "vi",
                "task": "transcribe",
            },
            "seed": args.seed,
            "trainable_parameter_count": trainable_count,
            "max_steps": args.max_steps,
        },
        "selection": {
            "metric_order": ["validation_wer", "validation_cer"],
            "lower_is_better": True,
            "best_epoch": best_epoch,
            "best_validation": best_metrics,
            "best_adapter_path": best_dir.as_posix(),
        },
        "stop_reason": stop_reason,
        "epochs_completed": len(history),
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
