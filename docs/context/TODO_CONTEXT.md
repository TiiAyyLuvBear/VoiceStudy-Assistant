# TODO Context

1. Run `notebooks/prepare-voxvietnam-on-kaggle.ipynb` against attached `/kaggle/input/voxvietnam-dataset` Parquet shards.
2. Inspect sample QC before starting full `train_small`/`test` QC; confirm duration and invalid-reason distributions are reasonable.
3. After full QC, confirm 230 `train_small` speakers with 30 valid audio, 50 remaining validation speakers with 15, and 50 official-test speakers with 15.
4. Save `voxvietnam_ecapa_three_task_v1` as a private Kaggle Dataset and attach it to the fine-tuning notebook.
5. Run standalone `notebooks/finetune-ecapa-for-vietnamese-speaker-indentificat.ipynb` from a fresh one-T4 kernel with processed dataset attached.
6. Confirm DataLoader/forward-backward probes and record peak RAM/VRAM, runtime, best epoch, and frozen-versus-fine-tuned metrics for all three tasks.
7. Verify the complete checkpoint/metrics/predictions artifact contract documented in `docs/voxvietnam_verification_dataset.md`.
8. Update `PROJECT_CHECKLIST.md` only after reproducible training and held-out evaluation artifacts exist.

Current blocker: local environment lacks attached Kaggle Parquet and Kaggle GPU; real QC/materialization/runtime verification requires Kaggle.

## VoxVietnam verification dataset

1. Attach the private `voxvietnam-dataset` containing `train_small-*`, `train-*`, and `test-*` Parquet shards.
2. Run `notebooks/prepare-voxvietnam-on-kaggle.ipynb`; full `train` is inventoried but not decoded.
3. Verify real-data speaker counts, output size, checksums, and generated trial balance.
4. Save Version and create a private Kaggle Dataset from `voxvietnam_ecapa_three_task_v1` output unless redistribution terms explicitly permit public hosting.
5. Run `notebooks/finetune-ecapa-for-vietnamese-speaker-indentificat.ipynb` on Kaggle with the private VoxVietnam subset attached.
6. Record frozen/fine-tuned closed-set accuracy/macro-F1, verification EER/minDCF/FAR/FRR/TAR, and open-set known accuracy/unknown rejection/AUROC/DIR.

Current blocker: real Kaggle QC and materialization have not been run yet.

## PhoWhisper-small runtime follow-up

1. Restart backend and confirm startup reports `vinai/PhoWhisper-small`.
2. Run command-audio evaluation with PhoWhisper-small; compare intent and entity
   accuracy against prior local LoRA v4 results.
3. If PhoWhisper-small wins, cache the model and set `asr.local_files_only: true`
   for offline demo runs.

Current local state: runtime config and wrapper support are updated. Real
PhoWhisper inference has not been run locally in this session.

## Three-task ECAPA evaluation

Implementation source of truth: `docs/ecapa_three_task_status_and_plan.md`.

Local Phases 1–3 are implemented and verified by unit/notebook contract tests.
Remaining work is Phase 4 only:

1. Run both regenerated notebooks on real gated VoxVietnam data and Kaggle GPU.
2. Preserve the generated checkpoint, metric JSON, prediction CSV, confusion matrix, and three-task summary artifacts.
3. Record actual results and resource usage in project documentation.

Do not fine-tune ECAPA separately for each task. Reuse one selected encoder checkpoint and task-specific decision layers.

## Next session: task-specific rolling checkpoints

Implementation plan: `docs/context/ECAPA_TASK_CHECKPOINT_PLAN.md`.

Use `gpt-5.6-luna`, medium reasoning, and `$caveman full`. Write tests first.
Implement validation-only rolling best checkpoints for closed, verification,
open, and balanced selection. Retain at most five weight-only `.pt` files.

Implementation complete locally; real Kaggle execution remains pending.

## Fine-tuned ECAPA runtime follow-up

Integration is complete and locally verified. Remaining operational steps:

1. Re-enroll every application user, including `user_001`, using five distinct
   recordings so centroid metadata matches `ecapa-voxvietnam-epoch-9`.
2. Run the Streamlit/end-to-end flow in the intended quiet demonstration room.
3. Confirm retry and non-voice fallback behavior for genuine rejection.
4. Keep the model limited to coursework/low-risk use until FRR and unseen-domain
   robustness improve.
