# VoiceStudy-Assistant

Week 1 skeleton for Vietnamese voice-study assistant. Includes SQLite data layer, Streamlit pages, deterministic ASR/NLU/Speaker mocks, access policy, seed data, and optional Vietnamese TTS.

## Run

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
pytest
```
