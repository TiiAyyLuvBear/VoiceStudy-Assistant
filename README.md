# VoiceStudy Assistant

Vietnamese voice assistant with PhoWhisper-small ASR, rule-based NLU, frozen
ECAPA-TDNN embeddings, cosine application speaker identification and
verification, SQLite ownership controls, backend Vietnamese TTS, and Streamlit
plus React UIs.

The active speaker evaluation data is frozen at `data/processed/v2/`.
Application enrollment is separate from the experimental Linear SVM:

- Linear SVM: experimental closed-set evaluation only.
- Application users: five recordings per user, one cosine centroid per user.
- Application SID: cosine gallery plus `application_sid_threshold`.
- Private-note SV: cosine score against the SID candidate plus
  `application_verification_threshold`.
- Private-note access also requires a spoken secret phrase marker, e.g.
  `mật khẩu hoa sen xanh`, matching the hash captured at enrollment.
- React frontend uses a two-step protected flow: command audio first, then
  registered secret phrase audio when prompted. FastAPI accepts optional
  `secret_audio` on `POST /api/v1/process`.

No command audio is used to train the speaker model.

## Requirements

- Python 3.10, 3.11, or 3.12
- Node.js 20 or newer for the React frontend
- Windows, Linux, or macOS
- CPU inference is supported; first ASR/ECAPA load can be slow

## Install

From the repository root:

```powershell
cd Final\VoiceStudy-Assistant
```

Create a Python virtual environment if `.venv` does not exist:

```powershell
python -m venv .venv
```

Install Python packages and seed local data:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\seed_database.py
```

Install React frontend packages:

```powershell
cd frontend
npm install
cd ..
```

## Start the System

Run backend and frontend in separate terminals.

Backend terminal:

```powershell
cd Final\VoiceStudy-Assistant
.\.venv\Scripts\python.exe -m backend.main
```

Backend URL: `http://127.0.0.1:8000`

FastAPI documentation: `http://127.0.0.1:8000/docs`

React frontend terminal:

```powershell
cd Final\VoiceStudy-Assistant\frontend
npm run dev -- --host 127.0.0.1
```

Open the Vite URL printed in the terminal, usually `http://localhost:5173/`.
If port `5173` is busy, Vite prints the next available port.

Optional Streamlit UI:

```powershell
cd Final\VoiceStudy-Assistant
.\.venv\Scripts\python.exe -m streamlit run app\main.py
```

Backend startup eagerly loads ECAPA and PhoWhisper by default. Terminal prints
one field per line, including model name, epoch, device, and load status.
Disable or make startup non-strict through `backend.preload_models` and
`backend.strict_model_startup` in `config.yaml`. First PhoWhisper run downloads
`vinai/PhoWhisper-small` into `models/cache/phowhisper` unless the model already
exists there or `asr.local_files_only` is enabled.

## Test Commands

Run deterministic backend/system contract tests:

```powershell
.\.venv\Scripts\python.exe scripts\run_system_tests.py
```

This runner uses isolated SQLite files and deterministic ASR/speaker adapters
to measure orchestration, authorization, ownership, and error contracts. It
also verifies that dynamic enrollment does not change the SHA-256 checksum of
the frozen v2 Linear SVM. Component accuracy remains in the real ASR/speaker
evaluation artifacts.

Run Python unit tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run React tests and production build:

```powershell
cd frontend
npm run test:run
npm run build
```

`Voice Assistant` accepts WAV recording/upload. Use the command catalog from
`GET /api/v1/commands`: time, view schedule, add schedule with free
`<tiêu đề>/<ngày>/<giờ>` slots, add note/private note with free `<nội dung>`,
and view latest private note. Protected private-note intents require the
registered secret phrase as a second audio sample in the React flow.
ASR post-processing details live in `docs/asr_postprocessing.md`.

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

SQLite file defaults to `data/database/voicestudy.db`. Schema has users, schedules, and notes; all schedule/note reads are filtered by owner. User secret phrases are stored as salted hashes, never plaintext. Seed script is idempotent and does not erase records.

User data management endpoints:

- `GET /api/v1/commands`
- `POST /api/v1/process` with required `audio` and optional `secret_audio`
- `GET /api/v1/users/{user_id}/schedules`
- `POST /api/v1/users/{user_id}/schedules`
- `DELETE /api/v1/users/{user_id}/schedules/{schedule_id}`
- `GET /api/v1/users/{user_id}/notes`
- `POST /api/v1/users/{user_id}/notes`
- `DELETE /api/v1/users/{user_id}/notes/{note_id}`

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
