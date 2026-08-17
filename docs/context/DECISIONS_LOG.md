# Decisions Log

## 2026-08-12 — ECAPA fine-tuning evaluation

- Use AAM-Softmax with a new classifier for project speaker labels; never reuse the 7,205-class VoxCeleb classifier.
- Select checkpoints using validation macro-F1 and restore the best checkpoint before test evaluation.
- Report direct AAM-classifier results, but compare pretrained and fine-tuned embedding quality using the same LinearSVC protocol and full utterances.
- Use test data only after training and validation selection are complete.

## 2026-08-12 — Stable Kaggle training default

- Default notebook training to one T4, batch size 2, gradient accumulation 16, effective batch 32.
- Keep DataLoader workers at zero until memory probes prove multiprocessing stable.
- Read at most 3 seconds from each source file before resampling to prevent CPU RAM and decode spikes.
- Enable two-GPU DataParallel only after DataLoader and forward/backward probes pass.

## 2026-08-12 — VoxVietnam compact verification protocol

- Use `train_small`, not redundant full `train`, as source for compact fine-tuning data.
- Select 300 train speakers with 30 audio each and 50 validation speakers with 15 audio each.
- Select 50 unseen speakers from official `test`, with 15 audio each.
- Reserve 5 validation/test audio per speaker for enrollment and use remaining audio as queries.
- Generate one positive and five negative trials per query; enforce a 10 GiB output budget.
- Keep train, validation, and test speakers pairwise disjoint.

## 2026-08-12 — VoxVietnam notebook evaluation protocol

- This protocol supersedes the earlier closed-set macro-F1/LinearSVC evaluation when the notebook uses the speaker-disjoint VoxVietnam subset.
- Train the AAM-Softmax head only on `train` speaker identities.
- Do not report closed-set classifier accuracy on validation/test because those speakers are unseen.
- Rank checkpoints by validation EER, using validation minDCF as the tie-break.
- Derive the operating threshold from validation minDCF and reuse it unchanged for held-out test FAR/FRR/TAR.
- Evaluate both frozen and fine-tuned ECAPA on the held-out test only after fine-tuning model selection finishes.

## 2026-08-12 — VoxVietnam eligible-speaker correction

- Real Hugging Face `train_small` scan found only 283 speakers with at least 30 audio.
- Use 230 train and 50 validation speakers at 30 selected audio per train speaker, leaving 3 eligible speakers as selection buffer.
- This supersedes the earlier 300-train-speaker compact configuration; official test remains 50 unseen speakers.

## 2026-08-12 — One encoder, three task protocols

- Fine-tune one ECAPA-TDNN encoder and reuse its embeddings for all speaker tasks.
- Use LinearSVC for closed-set identification, claimed-centroid cosine scoring for verification, and maximum-centroid scoring plus rejection threshold for open-set identification.
- Maintain separate ground-truth protocols and validation-selected hyperparameters/thresholds for each task.
- Current notebook is complete only for verification; identification and open-set evaluation must be added before claiming three-task coverage.

## 2026-08-12 — Three-task protocol materialization and evaluation

- Materialize audio once under train/validation/test and store task roles as CSV references under `protocols/`.
- Split each 30-audio closed-set speaker into 20 classifier/AAM train, 5 LinearSVC validation, and 5 held-out test recordings; all three roles are path/checksum disjoint.
- Within each 50-speaker validation/test group, deterministically use 25 speakers as open-set gallery-known and 25 as unknown; unknown speakers are absent from the corresponding gallery.
- Fine-tune ECAPA only on the 20-audio closed-set classifier-train partition so closed-set validation/test recordings are not encoder-training inputs.
- Embed the tested `src/speaker/evaluation.py` implementation into the self-contained Kaggle notebook and cache each unique protocol audio once per frozen/fine-tuned encoder after checkpoint restoration.
- Treat notebook output filenames as an artifact contract only; do not claim weights or metrics until the gated Kaggle run completes.

## 2026-08-13 — Kaggle-input preprocessing before protocol freeze

- Use attached `/kaggle/input/voxvietnam-dataset` Parquet shards as preparation input; no `HF_TOKEN` is required for this workflow.
- Audit speaker metadata first, then sample QC, then full streaming QC before selecting speakers and recordings.
- Decode only `train_small` and official `test`; inventory but never combine or decode redundant full `train`.
- Standardize selected audio to mono 16 kHz PCM16, trim boundary silence, enforce 2–10 second processed duration, and reject exact processed-content duplicates.
- Use a 12 GiB derived-audio budget and keep at least 4 GiB `/kaggle/working` headroom within the reported 20 GiB output allocation.
- Persist augmentation only as training-time configuration; never write augmented validation/test audio.

## 2026-08-14 — Separate train and validation eligibility

- Require 30 valid recordings only for 230 fine-tuning/closed-set speakers.
- Select 50 validation speakers from remaining `train_small` identities with at least 15 valid recordings; validation does not require 30.
- Keep official-test requirement at 50 speakers with at least 15 valid recordings.

## 2026-08-15 — Separate pre/post three-task evaluation

- Evaluate frozen ECAPA and cache its embeddings before any fine-tuning step.
- Evaluate restored fine-tuned ECAPA in a later dedicated cell.
- Keep all three task orchestration in `src/speaker/evaluation.py`; notebook imports it instead of embedding a copy.
- Store detailed artifacts in model-specific `evaluation/frozen` and `evaluation/finetuned` directories, with one combined `three_task_summary.csv`.

## 2026-08-15 — Standalone Kaggle fine-tuning notebook

- Supersedes runtime-import part of prior decision: Kaggle notebook embeds a generated snapshot of `src/speaker/evaluation.py`.
- Keep local module as tested source of truth; never hand-edit embedded evaluator cell.
- Runtime requires processed VoxVietnam Kaggle Dataset, but no repository source or external Python script.

## 2026-08-15 — Single notebook configuration cell

- All user-adjustable dataset, audio, hardware/batch, training, and output settings live in section 2 `Configuration — edit only this cell`.
- Execution cells consume settings and must not redeclare them.
- Derived values such as `BATCH_SIZE`, `GRAD_ACCUM`, paths, device, and AMP mode are computed once in the same cell.

## 2026-08-16 — Validation-only rolling checkpoint selection

- Keep stable `best_closed.pt`, `best_verification.pt`, `best_open.pt`, `best_balanced.pt`, and `latest.pt` paths under one checkpoint directory.
- Store weight-only payloads plus compact metadata; registry and validation history remain separate.
- Rank tasks by deterministic validation tuples; balanced selection requires frozen closed macro-F1 minus 0.005 constraint and falls back to best closed when no epoch qualifies.
- Notebook embeds checkpointing and evaluation code, so Kaggle run needs only notebook and processed dataset.

## 2026-08-16 — Fine-tuned ECAPA application runtime

- Use the epoch-9 VoxVietnam encoder for shared enrollment, application SID,
  and SV; do not load its 230-speaker training classifier at runtime.
- Verify checkpoint SHA-256 and strictly load `checkpoint["encoder"]` into the
  SpeechBrain ECAPA embedding model before serving requests.
- Keep deployment settings and quality gates in `config.yaml`.
- Use validation-selected thresholds: SV `0.4322190229975736` and open-set SID
  `0.3781695766069066`; never calibrate from test scores.
- Bind centroids to a model version through sibling metadata and fail closed on
  missing/mismatched metadata. Baseline users require re-enrollment.

## 2026-08-17 — Streamlit watcher with SpeechBrain

- Disable Streamlit file watching because module inspection activates
  SpeechBrain lazy optional integrations that ECAPA does not use.
- Do not install k2, flair, transformers, or numba solely to silence watcher tracebacks.

## 2026-08-17 — Structured backend request logging

- Use Python standard `logging` with JSON lines, console output, and rotating file.
- Pair request start/end through one request ID; record duration and SID/SV decisions.
- Keep transcripts disabled by default and never log audio or private-note content.

## 2026-08-17 — FastAPI backend boundary

- Run FastAPI and Streamlit as separate processes; Streamlit calls HTTP only for
  pipeline, speaker enrollment, and user management.
- Use one Uvicorn worker so cached CPU models are not duplicated in memory.
- Accept WAV multipart uploads in request-scoped temporary directories with a
  configurable 25 MiB per-file limit.
- Keep backend network/runtime settings centralized in `config.yaml`.

## 2026-08-17 — Backend startup readiness output

- Eagerly preload ECAPA and Whisper before FastAPI accepts requests; fail startup
  by default if either configured model cannot load.
- Print startup and request records one field per terminal line.
- Preserve JSONL rotating request files for parsing and later analysis.

## 2026-08-17 — Runtime audio resampling without librosa

- Use `scipy.signal.resample_poly` and NumPy silence trimming in backend audio
  preprocessing.
- Do not suppress SpeechBrain deprecation warnings or install optional `k2` for
  ECAPA. Librosa lazy stack inspection caused both misleading paths.

## 2026-08-17 — Authenticate before intent processing

- Run application SID and SV before ASR/NLU for every real audio request.
- Stop before transcription when SID or SV fails.
- Preserve speaker results if later ASR fails; intent controls allowed actions,
  not whether authentication runs.

## 2026-08-17 — Protocol-specific speaker response fields

- Treat top-level `speaker.identified` as SID decision.
- Nested `speaker.verification` exposes only SV fields. Do not include generic
  enrollment fields or a second `identified` value.

## 2026-08-17 — Repository artifact hygiene

- Keep `.env`, runtime SQLite databases, downloaded datasets, logs, model caches,
  and per-user centroids outside Git.
- Version deployable ECAPA `.pt` checkpoints with Git LFS.
- Keep source tests, docs, scripts, evaluation metrics, and reproducible notebook
  builders under normal version control.
