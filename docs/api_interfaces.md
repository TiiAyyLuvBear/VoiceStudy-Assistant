# VoiceStudy Assistant — API Interfaces

Tài liệu này là hợp đồng dữ liệu giữa ASR, NLU, Speaker Recognition,
Access Policy và Orchestrator. Các module không được tự đổi tên field nếu chưa
thống nhất với các thành viên còn lại.

## 0. FastAPI backend boundary

Streamlit is now an HTTP client. It does not import pipeline, speaker, or
database services directly. Start the API with `python -m backend.main` and
inspect OpenAPI at `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service and deployed-model status |
| `POST` | `/api/v1/process` | Multipart WAV through ASR/NLU/SID/SV pipeline |
| `POST` | `/api/v1/enroll` | User fields plus exactly five multipart WAV files |
| `GET` | `/api/v1/users` | List application users |
| `DELETE` | `/api/v1/users/{user_id}` | Delete one application user |

Uploads are written only to request-scoped temporary directories and removed
after processing. Default per-file limit is 25 MiB. Backend host, port, URL,
timeout, and upload limit live under `backend` in `config.yaml`.

On startup, backend preloads the configured ECAPA and Whisper models and prints
one field per line: model identity, checkpoint epoch, device, compute type, and
loaded state. Strict startup prevents serving when a configured model fails to
load. Pipeline request console records also use one field per line, while
`logs/requests.log` remains JSONL for machine processing.

## 1. ASR

### Python interface

```python
from src.asr.whisper_model import transcribe_audio

result = transcribe_audio("data/commands/audio/example.wav")
```

### Thành công

```json
{
  "transcript": "thêm lịch học máy lúc 8 giờ sáng mai",
  "model": "whisper-small",
  "language": "vi",
  "latency_ms": 1200.0,
  "success": true,
  "error": null
}
```

### Thất bại

ASR không ném lỗi ra Orchestrator đối với lỗi đầu vào hoặc lỗi inference.

```json
{
  "transcript": "",
  "model": "whisper-small",
  "language": "vi",
  "latency_ms": 3.4,
  "success": false,
  "error": "Audio file does not exist: data/missing.wav"
}
```

Quy ước:

- `transcript`: văn bản thô từ Whisper, chưa qua text normalization.
- `model`: luôn là `whisper-small` với cấu hình tuần 1.
- `language`: luôn là `vi`; hệ thống không tự dò ngôn ngữ.
- `latency_ms`: thời gian xử lý lời gọi, tính bằng millisecond.
- `success`: `true` chỉ khi inference hoàn tất và transcript không rỗng.
- `error`: `null` khi thành công, chuỗi thông báo khi thất bại.

Model được nạp lazy và được tái sử dụng giữa các lần gọi. Lần chạy đầu tiên có
thể tải model vào `models/cache/whisper`.

## 2. Text normalization

```python
from src.nlu.text_normalizer import normalize_text

normalized = normalize_text("Thêm lịch Học Máy lúc 8h sáng mai!")
```

Kết quả:

```text
thêm lịch học máy lúc 8 giờ sáng mai
```

Normalizer chuyển chữ thường, chuẩn hóa Unicode và khoảng trắng, bỏ dấu câu
không cần thiết, giữ dấu tiếng Việt, đồng thời chuẩn hóa cách viết giờ.

## 3. NLU

### Input

NLU nhận transcript đã được normalize:

```python
result = parse_command(
    transcript="thêm lịch học máy lúc 8 giờ sáng mai",
    reference_date="2026-07-28"
)
```

### Output

```json
{
  "intent": "ADD_SCHEDULE",
  "entities": {
    "title": "học máy",
    "date": "2026-07-29",
    "time": "08:00"
  },
  "missing_fields": []
}
```

Intent hợp lệ:

- `GET_TIME`
- `VIEW_SCHEDULE`
- `ADD_SCHEDULE`
- `VIEW_PRIVATE_NOTE`
- `OUT_OF_SCOPE`

Nếu câu lệnh không khớp rõ ràng:

```json
{
  "intent": "OUT_OF_SCOPE",
  "entities": {},
  "missing_fields": []
}
```

### Week 2 execution gates

The stable end-to-end API is available without database side effects:

```python
from src.pipeline import run_asr_nlu_pipeline

result = run_asr_nlu_pipeline(
    "data/commands/audio/validation/example.wav",
    reference_date="2026-07-28",
)
```

The result includes `missing_fields`, `can_execute`, and
`can_write_database`. An `ADD_SCHEDULE` command may write to the database
only when `can_write_database` is `true`. `OUT_OF_SCOPE`, ASR failures, and
commands with missing required entities are always blocked.

## 4. Speaker Recognition

### Runtime model contract

Enrollment, application speaker identification, and speaker verification use
the same fine-tuned ECAPA-TDNN encoder. Runtime settings live under `speaker`
in `config.yaml`, including the checkpoint path and SHA-256, encoder state key,
model version, embedding dimension, enrollment count, and verification
threshold.

The runtime loader initializes `speechbrain/spkrec-ecapa-voxceleb`, validates
the configured checkpoint, and strictly loads only `checkpoint["encoder"]`
into `classifier.mods.embedding_model`. The 230-speaker training classifier is
not used for application inference. One frozen evaluation-mode extractor is
cached and shared by enrollment, SID, and SV.

Enrollment centroids have a sibling `.meta.json` file containing the model
version and embedding dimension. A missing or mismatched metadata file returns
`CENTROID_MODEL_MISMATCH`; users enrolled with the former baseline model must
be enrolled again.

Speaker verification accepts a claim only when:

```text
cosine_similarity >= 0.4322190229975736
```

This threshold was selected from validation minDCF. Test-derived thresholds
must not replace it.

Kết quả nhận diện/xác thực mà Orchestrator sử dụng:

```json
{
  "candidate_user_id": "user001",
  "speaker_id": "spk0001",
  "similarity": 0.82,
  "identified": true,
  "verified": true,
  "latency_ms": 95.2
}
```

Application requests run SID and SV before ASR/NLU, so speaker fields are
available even when transcription fails or intent is public/out of scope.
Intent controls the permitted action only; it no longer controls whether
speaker verification runs.

## 5. Access Policy

| Intent | Quyền truy cập | Luồng xử lý |
|---|---|---|
| `GET_TIME` | Public action after authentication | SID, SV, rồi thực hiện |
| `VIEW_SCHEDULE` | SID | Speaker Identification rồi truy vấn lịch |
| `ADD_SCHEDULE` | SID | Speaker Identification rồi thêm lịch |
| `VIEW_PRIVATE_NOTE` | SID + SV | Identification, Verification rồi truy vấn ghi chú |
| `OUT_OF_SCOPE` | Reject after authentication | SID, SV, rồi từ chối; không truy vấn dữ liệu |

## 6. Orchestrator response

Each `process_audio_request()` call writes paired `request_started` and
`request_finished` JSON events through Python `logging`. Unhandled failures
write `request_failed`. Events include request ID, duration, intent, policy,
speaker decisions/similarities, and error state. Transcript inclusion is
controlled by `logging.requests.include_transcript` and defaults to `false`.
Audio content and private database content are not logged.

```json
{
  "success": true,
  "transcript": "cho tôi xem lịch ngày mai",
  "normalized_transcript": "cho tôi xem lịch ngày mai",
  "intent": "VIEW_SCHEDULE",
  "entities": {
    "date": "2026-07-29"
  },
  "speaker": {
    "candidate_user_id": "user001",
    "similarity": 0.82,
    "identified": true,
    "verified": true,
    "verification": {
      "similarity": 0.74,
      "verified": true
    }
  },
  "response_text": "Xin chào Lộc. Ngày mai bạn có lịch học Máy học lúc 8 giờ.",
  "latency_ms": 1450.5,
  "error": null
}
```

Quy tắc lỗi:

- SID thất bại: dừng trước ASR/NLU và không truy vấn database.
- SV thất bại: dừng trước ASR/NLU và từ chối yêu cầu.
- ASR thất bại sau xác thực: giữ kết quả SID/SV rồi trả lỗi ASR.
- `OUT_OF_SCOPE`: đã xác thực speaker nhưng không truy vấn database.
