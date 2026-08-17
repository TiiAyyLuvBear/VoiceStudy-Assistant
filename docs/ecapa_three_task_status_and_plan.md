# ECAPA Three-Task Status Report and Implementation Plan

Date: 2026-08-12

## 1. Scope

This report assesses the current Vietnamese ECAPA workflow against three system
tasks:

1. Closed-set speaker identification.
2. Speaker verification.
3. Open-set speaker identification.

The reviewed training notebook is
`notebooks/finetune-ecapa-for-vietnamese-speaker-indentificat.ipynb`. Dataset
preparation is implemented in `notebooks/prepare-voxvietnam-on-kaggle.ipynb`
and `scripts/build_voxvietnam_verification_dataset.py`.

## 2. Common Foundation Model

All three tasks should share one fine-tuned ECAPA-TDNN encoder:

```text
16 kHz waveform
-> SpeechBrain ECAPA-TDNN
-> L2-normalized 192-dimensional embedding
```

Current training starts from `speechbrain/spkrec-ecapa-voxceleb`, replaces the
VoxCeleb classifier with a classifier for selected VoxVietnam train speakers,
and optimizes AAM-Softmax. This is an appropriate supervised adaptation method.
The AAM classifier is a training head; deployment tasks consume encoder
embeddings through different decision layers.

## 3. Current Dataset Protocol

The corrected compact configuration uses:

- 230 train speakers with 30 audio files per speaker.
- 50 validation speakers with 15 audio files per speaker.
- 50 official-test speakers with 15 audio files per speaker.
- Validation/test speakers are unseen during fine-tuning and mutually disjoint.
- Five validation/test recordings per speaker form enrollment; the remaining
  recordings form verification queries.
- Verification trials contain one positive and five negative trials per query.
- Dataset output is limited to 10 GiB.

The real Hugging Face scan found only 283 `train_small` speakers with at least
30 recordings. Therefore, the earlier request for 300 train plus 50 validation
speakers was infeasible. The corrected requirement is 230 plus 50, leaving a
three-speaker eligibility buffer.

## 4. Completion Assessment

### 4.1 Fine-tuning foundation encoder — substantially complete

Implemented:

- VoxCeleb-pretrained ECAPA initialization.
- New VoxVietnam AAM-Softmax training head.
- Three-second random training crops.
- Classifier warm-up followed by encoder unfreezing.
- AdamW, separate encoder/classifier learning rates, gradient accumulation,
  mixed precision, gradient clipping, scheduler, early stopping, and checkpoint.
- Checkpoint selection using unseen-speaker validation EER with minDCF as the
  tie-break.
- Frozen-versus-fine-tuned encoder comparison.
- Saved training configuration, history, checkpoint, trial scores, and metrics.

Not yet verified:

- Complete Kaggle execution on real gated VoxVietnam audio.
- Peak CPU/GPU memory and runtime stability.
- Actual best epoch and validation/test metrics.
- Benefit of full-utterance or multi-crop evaluation.
- Benefit of noise, reverb, or speed augmentation.

### 4.2 Speaker verification — implemented, awaiting real run

Decision layer:

```text
enrollment recordings -> mean normalized centroid
query recording -> normalized embedding
cosine(query, centroid) >= validation threshold -> same speaker
```

Ground truth:

```csv
enrollment_speaker_id,query_audio_path,query_speaker_id,label
```

Implemented metrics:

- EER.
- Normalized minDCF with target prior 0.01.
- FAR, FRR, and TAR at a validation-selected threshold.
- TAR at FAR no greater than 1%.

Protocol controls:

- Fine-tune, validation, and test speaker sets are disjoint.
- Threshold and checkpoint are selected on validation only.
- Held-out test runs after checkpoint selection.

Status: implementation is suitable for speaker verification, but no real-run
result exists yet.

### 4.3 Closed-set speaker identification — implemented locally, awaiting real run

Required decision layer:

```text
fine-tuned ECAPA embedding -> multiclass LinearSVC -> known speaker_id
```

Implemented components:

- Audio-disjoint train/validation/test partitions sharing the same known
  speaker identities.
- SVM training and validation selection for `C` and optional `class_weight`.
- Test predictions and ground truth for known speakers.
- Accuracy, macro-F1, per-speaker recall, and confusion matrix.

The AAM training-head accuracy is not a substitute for this task evaluation.
The current validation/test speakers are unseen, so evaluating the train-class
AAM head on them would be invalid.

### 4.4 Open-set speaker identification — implemented locally, awaiting real run

Required decision layer:

```text
query embedding
-> cosine against every enrolled known-speaker centroid
-> highest-scoring identity
-> reject as UNKNOWN when maximum score is below validation threshold
```

Implemented components:

- Known-speaker gallery.
- Audio-disjoint known queries.
- Unknown queries from speakers absent from the gallery.
- Maximum-centroid scoring and rejection threshold selection.
- Known identification accuracy, unknown rejection rate, AUROC, FAR/FRR, and
  DIR at fixed FAR.

Current verification negative trials are not an open-set test. Every
validation/test speaker has an enrollment centroid, so no query speaker is
truly absent from the gallery.

## 5. Overall Status

| Component | Status | Evidence gap |
|---|---|---|
| Common ECAPA fine-tuning | Implemented | Real Kaggle run and metrics missing |
| Speaker verification | Implemented | Real held-out results missing |
| Closed-set identification | Implemented locally | Real held-out results missing |
| Open-set identification | Implemented locally | Real held-out results missing |

The regenerated notebook and reusable module now cover all three evaluation
tasks with one shared fine-tuned ECAPA checkpoint. Real gated-data execution,
saved weights, and held-out metrics remain outstanding.

## 6. Implementation Plan

### Phase 1 — Extend dataset protocols

Modify the VoxVietnam builder and preparation notebook to produce three task
protocols without duplicating audio unnecessarily.

Required metadata layout:

```text
protocols/
├── closed_set/
│   ├── classifier_train.csv
│   ├── validation_queries.csv
│   └── test_queries.csv
├── verification/
│   ├── validation_enrollment.csv
│   ├── validation_trials.csv
│   ├── test_enrollment.csv
│   └── test_trials.csv
└── open_set/
    ├── validation_gallery.csv
    ├── validation_queries.csv
    ├── test_gallery.csv
    └── test_queries.csv
```

Protocol requirements:

- Closed-set: same known speaker set across classifier train, validation, and
  test; audio paths and checksums disjoint.
- Verification: speakers disjoint from fine-tune train; enrollment/query audio
  disjoint; binary positive/negative trials.
- Open-set: known gallery speakers and known queries share identities; unknown
  queries come from speakers absent from gallery; validation and test protocol
  speakers are disjoint.
- No audio path/checksum overlaps incompatible roles.
- Manifest records counts, speaker sets, checksums, seed, ratios, and invariants.
- Remain within Kaggle storage budget.

Tests to deliver first:

- Deterministic protocol generation for fixed seed.
- Speaker and audio/checksum leakage rejection.
- Correct known/unknown membership.
- Every query and enrollment path resolves.
- Trial labels match speaker identities.
- Byte-budget failure removes partial output.

### Phase 2 — Add reusable evaluation functions

Prefer a testable Python module instead of placing all logic only in notebook
cells. Suggested module:

```text
src/speaker/evaluation.py
```

Functions should cover:

- L2 normalization and enrollment centroid construction.
- Cosine trial scoring.
- EER, minDCF, thresholded FAR/FRR/TAR.
- Closed-set LinearSVC selection using validation only.
- Open-set maximum-centroid prediction and rejection.
- Known/unknown AUROC and DIR at configured FAR.
- Machine-readable metric serialization.

Tests:

- Small synthetic embeddings with analytically predictable outputs.
- Threshold equality boundary.
- Missing centroid/query rejection.
- Degenerate labels and non-finite scores rejected.
- SVM hyperparameter selection never reads test labels.

### Phase 3 — Extend fine-tuning/evaluation notebook

Keep one ECAPA fine-tuning pass. After restoring the selected checkpoint:

1. Extract/cache embeddings once per unique audio path.
2. Run speaker verification evaluation.
3. Train/select LinearSVC on closed-set train/validation embeddings; evaluate
   closed-set test once.
4. Build gallery centroids, select open-set rejection threshold on validation,
   then evaluate held-out open-set test once.
5. Compare frozen and fine-tuned ECAPA under identical protocols.

Artifacts:

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

Notebook contract tests must verify:

- All Python cells compile.
- Test evaluation occurs after checkpoint restoration.
- Validation thresholds/hyperparameters are reused unchanged on test.
- All three task protocols and metrics are present.
- No obsolete closed-set assumption is applied to unseen verification speakers.

### Phase 4 — Run on Kaggle and record evidence

1. Run preparation notebook with gated `HF_TOKEN` and Internet enabled.
2. Save output as a private Kaggle Dataset.
3. Run fine-tuning/evaluation on one T4 from a fresh kernel.
4. Record dataset counts, output size, peak RAM/VRAM, runtime, best epoch, and
   all three task metrics.
5. Save checkpoint and result artifacts.
6. Update `PROJECT_CHECKLIST.md` only after artifacts and held-out results exist.

## 7. Definition of Done

Work is complete when:

- A single fine-tuned ECAPA checkpoint serves all three tasks.
- All three protocols have explicit, validated ground truth.
- No prohibited speaker/audio/checksum leakage exists.
- Hyperparameters and thresholds are selected only on validation.
- Test sets run only after model/protocol selection.
- Automated unit and notebook contract tests pass.
- Kaggle produces saved checkpoint, predictions, and metric artifacts.
- Report contains actual results rather than placeholders.

## 8. Immediate Next Action

Run the preparation notebook with gated `HF_TOKEN`, create the private Kaggle
Dataset, and run the fine-tuning/evaluation notebook from a fresh one-T4 kernel.
Do not mark checklist training/evaluation items complete until the real artifact
contract and held-out results have been saved.
