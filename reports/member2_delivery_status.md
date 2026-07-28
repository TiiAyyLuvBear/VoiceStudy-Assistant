# Thành viên 2 — delivery status

## Hoàn thành

- Whisper Small multilingual chạy CPU INT8, language `vi`.
- ASR API contract, lazy model loading, error handling và latency.
- Vietnamese text normalization.
- 64 command development, 30 validation, 30 test; không trùng sau normalization.
- Năm intent và rule-based classifier có `OUT_OF_SCOPE`.
- Date/time/title entity extraction và unified command parser.
- NLU baseline report và test set được chạy sau khi rule đóng băng.
- WER/CER/latency evaluation pipeline có checkpoint/resume.
- Command audio manifest 60 prompt, recording utility và audio/leakage validator.
- Checksum freeze manifest cho command datasets v1.
- API interfaces, runbook và unit tests.

## Chờ dữ liệu hoặc thao tác con người

1. `asr_validation.csv` và `asr_test.csv`: chờ `data_inventory.csv` chứa audio
   Speech-MASSIVE official validation/test từ Thành viên 1. Script sinh split đã sẵn sàng.
2. Command audio: manifest hiện có 60 hàng `pending`; cần thành viên thật đọc câu
   vào microphone. Không sinh TTS giả vì task yêu cầu metadata speaker và audio demo thật.
3. WER/CER official: chỉ chạy sau khi hai ASR split có audio thật.

Không mục nào ở trạng thái chờ được thay bằng dữ liệu giả trong báo cáo chính thức.
