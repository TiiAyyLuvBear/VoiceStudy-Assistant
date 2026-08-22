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

## 2026-08-17 — Deploy local ASR v4 artifact explicitly

- Treat `asr.model_path` as actual faster-whisper model source when configured;
  `model_name` remains reporting metadata only.
- Resolve relative model paths from `config.yaml` and fail startup when required
  CTranslate2 files are missing.
- Deploy Whisper Small LoRA v4 from `models/experimental/asr/v4/ctranslate2`.
- Keep CPU int8 default; allow `cuda` and `auto` for later hardware deployment.
---

# Decision: Prompt-guided enrollment and secret phrase verification

Date: 2026-08-20

Decision:
- New application enrollment requires 5 guided WAV samples and a secret phrase with at least 3 words.
- Secret phrases are stored as salted PBKDF2 hashes, never plaintext.
- Private-note access requires SID, matching spoken secret phrase after a marker such as `mật khẩu`, then SV.
- Audio quality and embedding consistency gates are configurable under `speaker.enrollment_quality`; strict gates are enabled in the main `config.yaml`.

Rationale:
- Prompt-guided samples improve speaker embedding robustness across realistic commands.
- Hash-only secret storage reduces damage if SQLite leaks.
- Secret phrase adds a knowledge factor for private tasks without retraining speaker models.
- Configurable quality gates keep tests deterministic while enforcing stricter behavior in the real app config.

Consequences:
- Clients calling `/api/v1/enroll` must send `secret_phrase` and should send the official 5 `enrollment_prompts`.
- Existing enrolled users without secret phrase columns fail private-note access with `SECRET_PHRASE_NOT_CONFIGURED` until re-enrolled or migrated.
- Transcript must include a supported marker, e.g. `mở ghi chú riêng tư mật khẩu hoa sen xanh`.
---

# Decision: User-scoped schedule and note management endpoints

Date: 2026-08-20

Decision:
- Manage schedules and notes under `/api/v1/users/{user_id}/schedules` and `/api/v1/users/{user_id}/notes`.
- Delete operations require both `user_id` and item ID in SQL.
- Streamlit User Management page is the first UI surface for CRUD management.

Rationale:
- Notes and schedules are user-owned data; URL structure and SQL filters should make ownership explicit.
- Owner-scoped deletes prevent deleting another user's record by guessing `schedule_id` or `note_id`.
- Streamlit already hosts the operational user management workflow.

Consequences:
- React frontend still lacks schedule/note management UI unless added later.
- Current CRUD supports create/list/delete; edit/update is not implemented.
---

# Decision: Fixed command template catalog and separate secret-audio verification

Date: 2026-08-20

Decision:
- Maintain a fixed command template catalog in `src/nlu/command_catalog.py`.
- Run ASR post-processing by fuzzy-snapping noisy transcripts to catalog phrases only when the command has no free slots.
- Expose catalog through `GET /api/v1/commands`.
- For protected intents, React records command audio first, then asks for a second audio sample containing the registered secret phrase.
- `POST /api/v1/process` accepts optional `secret_audio`; orchestrator verifies this raw phrase before SV and private data access.

Rationale:
- Fixed command scripts reduce ASR variance and make downstream intent/entity parsing more stable.
- Slot templates keep system commands general; task data such as schedule title or note content remains in the user utterance.
- Separate secret phrase audio makes verification explicit instead of forcing users to append the secret to the command.
- Same-transcript marker compatibility remains useful for direct API/manual tests.

Consequences:
- Adding a new supported user command now requires updating command catalog, NLU parsing/entity logic, access policy, and frontend display/tests.
- React protected-flow state keeps the original command audio in memory until verification completes or user records a new command.

---

# Decision: Add private-note write intent

Date: 2026-08-20

Decision:
- Add `ADD_PRIVATE_NOTE` as a protected intent requiring SID, spoken secret phrase, and SV before database write.
- Extract note `content` from commands such as `Thêm ghi chú riêng tư <nội dung>`.
- Treat generic non-private notes such as `Thêm ghi chú về môn toán` as `OUT_OF_SCOPE` for the voice pipeline.

Rationale:
- The old catalog could view private notes but could not create them by voice.
- Writing private notes is a private task and should use the same verification policy as viewing private notes.

Consequences:
- `can_write_database` now covers `ADD_SCHEDULE` and `ADD_PRIVATE_NOTE`.
- Slot commands are shown in UI but are not snap-normalized to fixed phrases, preserving user content.

---

# Decision: Backend Vietnamese TTS for assistant output

Date: 2026-08-20

Decision:
- React assistant output must request speech from backend `POST /api/v1/tts`.
- Backend returns MP3 bytes with `audio/mpeg` and no-store cache policy.
- Browser `window.speechSynthesis` is no longer used for the React assistant output path.
- Current implementation uses existing `src.tts.text_to_speech.synthesize_vietnamese`, backed by `gTTS(lang="vi")`.

Rationale:
- Browser voices vary by OS/browser and often lack natural Vietnamese support.
- Backend TTS gives one API contract for later model replacement without changing frontend workflow.

Consequences:
- Runtime TTS depends on the configured backend synthesizer; current `gTTS` implementation needs network access.
- If `gTTS` fails, API returns HTTP 503 and frontend keeps text visible with a warning.

---

# Decision: Trial PhoWhisper-small runtime ASR

Date: 2026-08-20

Decision:
- Switch runtime ASR config to `backend: transformers` and `model_name: vinai/PhoWhisper-small`.
- Keep `model_size: small`, Vietnamese transcribe task, CPU device, and beam size 10.
- Use `models/cache/phowhisper` as the Hugging Face cache and allow first-run download with `local_files_only: false`.
- Preserve faster-whisper/CTranslate2 support for the older local LoRA v4 artifact.

Rationale:
- PhoWhisper-small is a Vietnamese ASR model exposed through Transformers/PyTorch.
- The existing faster-whisper loader cannot directly load this Hugging Face Transformers checkpoint.
- Keeping the same ASR wrapper contract avoids changing orchestration, authentication, and NLU code.

Consequences:
- Startup may download and load `vinai/PhoWhisper-small` before accepting requests when `backend.preload_models` is true.
- Transformers ASR currently uses `soundfile`/SciPy decoding and resampling; WAV/FLAC are the safest runtime inputs.
- To run fully offline after caching, set `asr.local_files_only: true`.

---

# Decision: Enrollment requires secret audio plus typed transcript

Date: 2026-08-20

Decision:
- React and Streamlit enrollment require a typed secret phrase transcript and a separate WAV/audio recording of the same phrase.
- Backend `/api/v1/enroll` requires multipart `secret_audio`, transcribes it with the configured ASR backend, and compares normalized ASR transcript against the typed transcript before hashing/storing.
- If ASR fails or the transcript does not match, enrollment returns `success=false` with `SECRET_PHRASE_ASR_FAILED` or `SECRET_PHRASE_TRANSCRIPT_MISMATCH`; the enroller does not run.
- Secret phrase storage remains salted PBKDF2 hash only; raw enrollment secret audio is temporary and not stored.

Rationale:
- Enrollment should verify that the user's spoken secret phrase is recognizable by the same ASR stack used later for protected actions.
- Typed transcript gives a human-confirmed target; spoken audio confirms ASR can recover it.
- Using the configured ASR means PhoWhisper-small is used for this check when `config.yaml` selects `backend: transformers`.

Consequences:
- All `/api/v1/enroll` clients must send `secret_phrase`, `secret_phrase_transcript`, `secret_audio`, five enrollment WAV samples, and the official prompts.
- Existing API tests and Streamlit refresh enrollment flows must inject or collect secret audio.

---

# Decision: Enrollment consistency can pass by centroid coherence

Date: 2026-08-22

Decision:
- Speaker enrollment consistency is valid when either strict pairwise cosine thresholds pass or accepted embeddings remain coherent around the shared centroid.
- Per-sample outlier rejection by `min_centroid_similarity` remains active before final centroid creation.
- `speaker.enrollment_quality.min_mean_centroid_similarity` is explicit in `config.yaml` with default `0.70`.

Rationale:
- Guided enrollment records different prompts, so same-speaker short utterances can have low pairwise cosine even in the same environment.
- Centroid coherence better matches the artifact being stored: one normalized user centroid built from accepted samples.

Consequences:
- `embedding_consistency` now reports `valid_by_pairwise`, `valid_by_centroid`, `mean_centroid_similarity`, and `min_centroid_similarity`.
- If enrollment still returns `EMBEDDING_CONSISTENCY_FAILED`, inspect `embedding_consistency.accepted` to see which gate failed.
