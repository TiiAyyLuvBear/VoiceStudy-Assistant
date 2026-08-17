# VoxVietnam ECAPA Three-Task Dataset and Kaggle Runbook

This project uses one compact package from the gated
`hustep-lab/VoxVietnam-Dataset` for closed-set identification, speaker
verification, and open-set identification. Audio is materialized once; task
ground truth is stored as CSV references under `protocols/`.

## Access and redistribution

1. Accept the dataset access conditions on Hugging Face.
2. Add a private Kaggle Secret named `HF_TOKEN`.
3. Keep the derived Kaggle Dataset private unless the gated terms explicitly
   permit redistribution.

Do not commit or paste the token into a notebook. VoxVietnam is licensed under
CC BY-NC 4.0 and its gated-access terms also apply.

## Protocol

- 230 closed-set/fine-tune speakers with 30 audio each: 20 classifier/AAM
  train, 5 LinearSVC validation, and 5 held-out closed-set test.
- 50 unseen validation speakers and 50 disjoint official-test speakers with 15
  audio each.
- Verification uses 5 enrollment audio per speaker, the remaining queries, and
  one positive plus five negative trials per query.
- Open set deterministically assigns 25 gallery-known and 25 unknown speakers
  within each validation/test group. Gallery audio and query audio are disjoint.
- Validation selects the ECAPA checkpoint, LinearSVC hyperparameters, claimed
  verification threshold, and open-set rejection threshold. Test does not tune
  any decision.

## Kaggle preparation

Run `notebooks/prepare-voxvietnam-on-kaggle.ipynb` with Internet enabled and
`HF_TOKEN` configured. It streams `train_small` and official `test`, writes
`/kaggle/working/voxvietnam_ecapa_three_task_v1`, checks all protocol
invariants, and enforces a 10 GiB hard output limit. It intentionally does not
create a ZIP because that would temporarily duplicate disk usage.

After a successful run, save a version and create a private Kaggle Dataset from
that output folder. Attach it to
`notebooks/finetune-ecapa-for-vietnamese-speaker-indentificat.ipynb`.

The equivalent local command is:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_voxvietnam_verification_dataset --output-root data/datasets/voxvietnam_ecapa_three_task_v1 --train-speakers 230 --validation-speakers 50 --test-speakers 50 --train-audio-per-speaker 30 --evaluation-audio-per-speaker 15 --enrollment-audio-per-speaker 5 --negative-trials-per-query 5 --closed-set-train-audio-per-speaker 20 --closed-set-validation-audio-per-speaker 5 --open-set-known-speakers 25 --max-gib 10 --seed 42
```

## Dataset layout

```text
voxvietnam_ecapa_three_task_v1/
├── train|validation|test/
│   ├── audio/<speaker>/*.wav
│   └── metadata.csv
├── protocols/
│   ├── closed_set/{classifier_train,validation_queries,test_queries}.csv
│   ├── verification/{validation_enrollment,validation_trials,test_enrollment,test_trials}.csv
│   └── open_set/{validation_gallery,validation_queries,test_gallery,test_queries}.csv
├── speaker_mapping.csv
├── manifest.json
└── README.md
```

`manifest.json` records counts, protocol CSV hashes, seed, byte usage, split
configuration, and leakage invariants. All audio is mono 16 kHz PCM16 WAV.

## Fine-tuning and artifact contract

Run the fine-tuning notebook from a fresh one-T4 Kaggle kernel. It restores the
validation-selected ECAPA checkpoint, caches each unique audio embedding once
per encoder, and evaluates frozen and fine-tuned ECAPA with identical
protocols. A successful run must leave:

```text
ecapa_voxvietnam_best.pt
ecapa_training_config.json
ecapa_training_history.csv
verification_metrics.json
verification_trial_scores.csv
closed_set_metrics.json
closed_set_predictions.csv
closed_set_confusion_matrix.csv
open_set_metrics.json
open_set_predictions.csv
three_task_summary.csv
```

Record output size, peak RAM/VRAM, runtime, best epoch, and the three-task
metrics before marking any training/evaluation checklist item complete.

## Current blocker

No real gated VoxVietnam audio or Kaggle GPU is available locally. The files
above are a verified workflow and artifact contract only; this repository does
not currently contain real Kaggle weights or metrics.
