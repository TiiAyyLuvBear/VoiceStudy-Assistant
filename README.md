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

```powershell
cd Final\VoiceStudy-Assistant
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\seed_database.py
```

Run backend and frontend in separate terminals.

Backend terminal:

```powershell
.\.venv\Scripts\python.exe -m backend.main
```

Frontend terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\main.py
```

FastAPI documentation: `http://127.0.0.1:8000/docs`.

Backend startup eagerly loads ECAPA and Whisper by default. Terminal prints one
field per line, including model name, epoch, device, and load status. Disable or
make startup non-strict through `backend.preload_models` and
`backend.strict_model_startup` in `config.yaml`.

`Voice Assistant` accepts WAV recording/upload. Until ASR, NLU, and Speaker modules arrive, enter a mock transcript to test flows. Use phrases such as `Bây giờ là mấy giờ?`, `Xem lịch của tôi`, and `Mở ghi chú riêng tư`.

Every real audio request performs speaker identification and verification
before ASR/NLU. Request logs therefore contain SID/SV decisions regardless of
recognized intent. Failed authentication stops transcription and application
actions.

## Backend request logs

Audio pipeline requests produce privacy-safe JSON records in both the FastAPI
terminal and `logs/requests.log`. Uvicorn also prints HTTP method, path, status,
and client address. Follow the rotating file live:

```powershell
Get-Content .\logs\requests.log -Wait -Tail 20
```

Settings live under `logging.requests` in `config.yaml`. Transcript logging is
disabled by default. Raw audio and private-note contents are never logged.
Terminal records print one field per line; rotating file records remain JSONL.

## Database

SQLite file defaults to `data/database/voicestudy.db`. Schema has users, schedules, and notes; all schedule/note reads are filtered by owner. Seed script is idempotent and does not erase records.

## Verify

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
