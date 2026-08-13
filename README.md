# VoiceStudy Assistant

Vietnamese voice assistant with Whisper Small, rule-based NLU, frozen
ECAPA-TDNN embeddings, cosine application speaker identification and
verification, SQLite ownership controls, TTS fallback, and a Streamlit UI.

The active speaker evaluation data is frozen at `data/processed/v2/`.
Application enrollment is separate from the experimental Linear SVM:

- Linear SVM: experimental closed-set evaluation only.
- Application users: five recordings per user, one cosine centroid per user.
- Application SID: cosine gallery plus `application_sid_threshold`.
- Private-note SV: cosine score against the SID candidate plus
  `application_verification_threshold`.

No command audio is used to train the speaker model.

## Requirements

- Python 3.11 or 3.12
- Windows, Linux, or macOS
- CPU inference is supported; the first ASR/ECAPA run can be slow

Install:

```bash
python -m pip install -r requirements.txt
```

Model caches are read from `models/cache/`. Whisper can download its configured
checkpoint on first use. ECAPA uses
`speechbrain/spkrec-ecapa-voxceleb` in evaluation mode with frozen parameters.

## Configuration

All paths are project-relative in `config.yaml`. Important application paths:

```yaml
speaker:
  application_centroid_dir: models/application/user_embeddings
  application_sid_threshold_path: models/application/application_sid_threshold.json
  application_verification_threshold_path: models/application/application_verification_threshold.json
```

The application SID threshold currently inherits the frozen experimental
validation threshold because independent non-enrolled application validation
audio is unavailable. Its JSON records this domain-mismatch limitation.
The application SV threshold is calibrated only from held-out validation
command audio and is not tuned on speaker-v2 test data.

## Database and demo users

Create or seed SQLite:

```bash
python scripts/seed_database.py
```

Prepare real centroids for `user_001`, `user_002`, and `user_003`:

```bash
python scripts/prepare_demo_enrollments.py
python scripts/calibrate_application_verification.py
python scripts/build_application_sid_config.py
```

Each demo user receives exactly five validation command recordings for
enrollment. Five additional recordings per user remain held out. The mapping is
saved to `experiments/system/demo_enrollment_data.csv`.

Evaluate held-out application SID and genuine/impostor SV:

```bash
python scripts/evaluate_demo_application_speaker.py
```

## Run the Streamlit application

```bash
streamlit run app/main.py
```

Pages:

- **Voice Assistant**: record/upload WAV, then show ASR, NLU, SID/SV,
  thresholds, latency, database response, and best-effort Vietnamese TTS.
- **Speaker Enrollment**: record or upload exactly five distinct WAV files.
- **User Management**: view users, replace five-file enrollment, or delete a
  user together with owned SQLite rows and the managed application centroid.

Access rules:

| Intent | Policy | Execution |
|---|---|---|
| `GET_TIME` | Public | Never calls SID or SV |
| `VIEW_SCHEDULE` | SID | Uses only the SID candidate as database owner |
| `ADD_SCHEDULE` | SID | Requires title/date/time before writing |
| `VIEW_PRIVATE_NOTE` | SID + SV | Reads notes only after successful SV |
| `OUT_OF_SCOPE` | Reject | Calls neither speaker APIs nor database |

Client input and transcripts cannot supply the owner `user_id`; schedule and
note access always uses the candidate returned by application SID.

## Tests

Run all tests:

```bash
python -m pytest -q
```

Run the integration/API subset:

```bash
python -m pytest -q tests/test_application_api.py tests/test_integration.py
```

Generate the 30 deterministic system contracts, task-success artifact, latency
artifact, and dynamic `user_004` enrollment test:

```bash
python scripts/run_system_tests.py
```

This runner uses isolated SQLite files and deterministic ASR/speaker adapters
to measure orchestration, authorization, ownership, and error contracts. It
also verifies that dynamic enrollment does not change the SHA-256 checksum of
the frozen v2 Linear SVM. Component accuracy remains in the real ASR/speaker
evaluation artifacts.

## Main outputs

- `models/application/user_embeddings/user_001.npy` through `user_003.npy`
- `models/application/application_sid_threshold.json`
- `models/application/application_verification_threshold.json`
- `models/application/application_sid_config.json`
- `experiments/system/system_test_cases.csv`
- `experiments/system/system_test_results.csv`
- `experiments/system/task_success_rate.json`
- `experiments/system/system_latency_results.csv`
- `experiments/system/dynamic_enrollment_test_results.csv`
- `experiments/system/application_sid_heldout_results.csv`
- `experiments/system/application_sv_results.csv`

Do not modify frozen v1/v2 metadata or reuse test results to tune thresholds.
Any future speaker dataset development must create a new version.
