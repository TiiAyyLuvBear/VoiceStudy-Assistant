# ECAPA Task-Specific Rolling Checkpoint Plan

Status: ready for implementation in next session.

## Required Session Configuration

- Model: `gpt-5.6-luna`
- Reasoning: `medium`
- Communication skill: `$caveman full`
- Work inside `.venv`; write tests before implementation.

## Objective

Select and retain weight-only ECAPA checkpoints best matched to:

1. Closed-set speaker identification.
2. Speaker verification.
3. Open-set speaker identification.
4. Balanced three-task deployment.

Checkpoint selection must use validation data only. Test data runs once after
selection. Keep at most five checkpoint files so Kaggle `/kaggle/working`
remains well below its 20 GB limit.

## Current Evidence

Latest reported frozen-versus-fine-tuned results:

| Metric | Frozen | Fine-tuned |
|---|---:|---:|
| Closed-set accuracy | 0.897391 | 0.871304 |
| Closed-set macro-F1 | 0.891181 | 0.866045 |
| Verification EER | 0.1820 | 0.1396 |
| Verification minDCF | 0.6408 | 0.5328 |
| Open-set known-ID accuracy | 0.600 | 0.648 |
| Open-set unknown rejection | 0.860 | 0.900 |
| Open-set AUROC | 0.814160 | 0.830464 |

Fine-tuning improves verification and open-set metrics but reduces closed-set
performance. Current checkpoint selection prioritizes verification EER only.

## Files to Change

| File | Planned change |
|---|---|
| `src/speaker/checkpointing.py` | Rolling checkpoint comparison, overwrite, registry logic |
| `src/speaker/evaluation.py` | Validation-only task evaluators separated from held-out test evaluation |
| `scripts/rebuild_voxvietnam_finetune_notebook.py` | Training-loop integration and standalone embedding |
| `notebooks/finetune-ecapa-for-vietnamese-speaker-indentificat.ipynb` | Regenerated standalone Kaggle notebook |
| `tests/test_speaker_checkpointing.py` | Unit tests for ranking, ties, overwrite, file limit, leakage prevention |
| `tests/test_speaker_evaluation.py` | Validation-only evaluation tests |
| `tests/test_voxvietnam_finetune_notebook.py` | Static notebook ordering, standalone, and no-test-selection contract |

## Configuration to Add

Place all values in notebook section `Configuration — edit only this cell`:

```python
CHECKPOINT_EVAL_INTERVAL = 2
MAX_ROLLING_CHECKPOINTS = 5

CHECKPOINT_DIR = WORKING_DIR / "checkpoints"
BEST_CLOSED_PATH = CHECKPOINT_DIR / "best_closed.pt"
BEST_VERIFICATION_PATH = CHECKPOINT_DIR / "best_verification.pt"
BEST_OPEN_PATH = CHECKPOINT_DIR / "best_open.pt"
BEST_BALANCED_PATH = CHECKPOINT_DIR / "best_balanced.pt"
LATEST_PATH = CHECKPOINT_DIR / "latest.pt"

MAX_CLOSED_F1_DROP = 0.005
```

Do not redeclare these variables in later cells. Extend the AST-based notebook
configuration test when adding them.

## Tests First

Implement failing tests for:

1. Closed checkpoint: highest validation macro-F1; accuracy tie-break.
2. Verification checkpoint: lowest validation EER; minDCF tie-break.
3. Open checkpoint: highest validation AUROC; DIR@FAR=1% tie-break.
4. Balanced checkpoint: highest fixed balanced score while satisfying the
   closed-F1 constraint.
5. Stable checkpoint path is overwritten only on improvement.
6. Registry records task, epoch, metrics, path, and selection reason.
7. No more than five `.pt` files exist.
8. Optimizer, scaler, embeddings, and predictions are absent from checkpoint.
9. Epoch-selection code never reads test protocols or test metrics.
10. Notebook imports no external local script and every Python cell compiles.

## Validation/Test Separation

Add validation-only entry points:

```python
evaluate_closed_validation(...)
evaluate_verification_validation(...)
evaluate_open_validation(...)
```

The training loop may call only these functions. Do not call the current
held-out `evaluate_three_tasks()` path during checkpoint selection.

Recommended schedule:

```text
verification validation: every epoch
closed/open validation: every 2 epochs after warm-up
```

Closed validation requires fitting LinearSVC from current closed-train
embeddings and evaluating closed-validation embeddings. Open/verification use
validation enrollment/gallery/query protocols only.

## Selection Rules

Use deterministic tuples:

```text
closed:       maximize (macro-F1, accuracy, earlier epoch)
verification: minimize (EER, minDCF, earlier epoch)
open:         maximize (AUROC, DIR@FAR=1%, earlier epoch)
```

Balanced score:

```python
balanced_score = (
    closed_macro_f1
    + (1.0 - verification_eer)
    + open_auroc
) / 3.0
```

Balanced eligibility:

```python
closed_macro_f1 >= frozen_closed_macro_f1 - MAX_CLOSED_F1_DROP
```

If no epoch satisfies the constraint, record this explicitly and fall back to
the best closed checkpoint. Never silently relax the constraint.

## Checkpoint Payload

Save weights and compact metadata only:

```python
{
    "encoder": model.encoder.state_dict(),
    "classifier": model.classifier.state_dict(),
    "epoch": epoch,
    "task": task,
    "validation_metrics": metrics,
    "training_config": training_config,
}
```

Do not save optimizer, scheduler, AMP scaler, embeddings, trial scores,
predictions, or confusion matrices per epoch.

## Rolling Storage Contract

Stable filenames:

```text
checkpoints/best_closed.pt
checkpoints/best_verification.pt
checkpoints/best_open.pt
checkpoints/best_balanced.pt
checkpoints/latest.pt
```

Each current weight-only checkpoint is about 83.5 MB. Five files should remain
near 418 MB. Write metadata separately:

```text
checkpoints/checkpoint_registry.json
checkpoint_validation_history.csv
```

## Final Held-Out Evaluation

After training and all selection decisions:

1. Load `best_closed.pt`; run closed test only.
2. Load `best_verification.pt`; run verification test only.
3. Load `best_open.pt`; run open test only.
4. Load `best_balanced.pt`; run all three tests for shared-encoder comparison.

Write task-specific artifacts under distinct directories. Include selected
epoch and checkpoint filename in every metrics JSON. Test labels or scores must
not modify checkpoint selection.

## Verification Commands

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_voxvietnam_finetune_notebook.py
.\.venv\Scripts\python.exe -m pytest --rootdir=tests --confcutdir=tests --ignore=lib64 --ignore=.venv tests\test_speaker_checkpointing.py tests\test_speaker_evaluation.py tests\test_voxvietnam_finetune_notebook.py -q
```

Then run the relevant full suite. Record exact pass counts.

## Acceptance Criteria

- Tests written before implementation.
- At most five weight-only checkpoints.
- Storage estimate below 500 MB.
- Best epoch and validation metric recorded for each task.
- Balanced checkpoint enforces closed-F1 constraint.
- No test leakage.
- Final test executes only after checkpoint selection.
- Notebook remains standalone on Kaggle with processed dataset attached.
- All configuration remains in one documented configuration cell.
- Context and artifact documentation updated after verification.

## Next-Session Prompt

```text
Use $caveman full. Implement docs/context/ECAPA_TASK_CHECKPOINT_PLAN.md using
gpt-5.6-luna with medium reasoning. Work in .venv and write tests first. Keep
the Kaggle notebook standalone. Select checkpoints from validation data only,
run held-out test only after selection, and retain at most five weight-only
checkpoint files. Run targeted tests and relevant full suite before completion.
```
