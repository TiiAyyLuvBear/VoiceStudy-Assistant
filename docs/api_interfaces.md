# VoiceStudy Assistant — API Interfaces

Tài liệu này là hợp đồng dữ liệu giữa ASR, NLU, Speaker Recognition,
Access Policy và Orchestrator. Các module không được tự đổi tên field nếu chưa
thống nhất với các thành viên còn lại.

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

`verified` chỉ có ý nghĩa với intent yêu cầu Speaker Verification.

## 5. Access Policy

| Intent | Quyền truy cập | Luồng xử lý |
|---|---|---|
| `GET_TIME` | Public | Thực hiện trực tiếp |
| `VIEW_SCHEDULE` | SID | Speaker Identification rồi truy vấn lịch |
| `ADD_SCHEDULE` | SID | Speaker Identification rồi thêm lịch |
| `VIEW_PRIVATE_NOTE` | SID + SV | Identification, Verification rồi truy vấn ghi chú |
| `OUT_OF_SCOPE` | Reject | Từ chối, không truy vấn dữ liệu |

## 6. Orchestrator response

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
    "verified": null
  },
  "response_text": "Xin chào Lộc. Ngày mai bạn có lịch học Máy học lúc 8 giờ.",
  "latency_ms": 1450.5,
  "error": null
}
```

Quy tắc lỗi:

- ASR thất bại: dừng pipeline và trả lỗi ASR.
- `OUT_OF_SCOPE`: không gọi Speaker Recognition và database.
- SID thất bại cho chức năng cá nhân: không truy vấn dữ liệu cá nhân.
- SV thất bại cho ghi chú riêng tư: từ chối truy cập.

## 7. Week 3 application speaker contracts

The Streamlit application and orchestrator use `src.speaker.application`.
Experimental Linear SVM identification remains separate.

```python
enroll_user(user_id, name, five_audio_paths)
identify_application_user(audio_path)
verify_speaker(audio_path, candidate_user_id)
```

Application SID returns both the canonical Week-3 fields and backward-compatible
aliases:

```json
{
  "protocol": "APPLICATION_SID",
  "candidate_user_id": "user_003",
  "cosine_similarity": 0.76,
  "similarity": 0.76,
  "unknown_threshold": 0.51307271,
  "status": "KNOWN",
  "identified": true,
  "latency_ms": 25.0,
  "success": true,
  "error": null
}
```

An unknown result has `status="UNKNOWN"`, `identified=false`, and hides
`candidate_user_id`. Speaker verification includes
`verification_threshold`, `verified`, and `latency_ms`.

The orchestrator response uses `response` for display text and contains total
`latency_ms` plus `stage_latency_ms`. Database ownership is always the
`candidate_user_id` returned by application SID. A client/transcript
`user_id` is ignored.
