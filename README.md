# VoiceStudy-Assistant

Week 1 skeleton for Vietnamese voice-study assistant. Includes SQLite data layer, Streamlit pages, deterministic ASR/NLU/Speaker mocks, access policy, seed data, and optional Vietnamese TTS.

## Run

```bash
cd Final/VoiceStudy-Assistant
python -m pip install -r requirements.txt
python scripts/seed_database.py
streamlit run app/main.py
```

`Voice Assistant` accepts WAV recording/upload. Until ASR, NLU, and Speaker modules arrive, enter a mock transcript to test flows. Use phrases such as `Bây giờ là mấy giờ?`, `Xem lịch của tôi`, and `Mở ghi chú riêng tư`.

## Database

SQLite file defaults to `data/database/voicestudy.db`. Schema has users, schedules, and notes; all schedule/note reads are filtered by owner. Seed script is idempotent and does not erase records.

## Verify

```bash
pytest
```
