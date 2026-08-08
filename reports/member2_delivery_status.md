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
- ASR validation 100/100 và ASR test 125/125, có WER/CER/latency.
- Command audio đủ 60/60 và đã qua kiểm định WAV/checksum/leakage.
- Command validation và command test đã chạy đủ 30/30 audio bằng Whisper.
- Intent, OUT_OF_SCOPE và entity metrics đã được cập nhật trên bộ test đầy đủ.

## Trạng thái còn lại

Không còn dữ liệu ASR hoặc command audio bị thiếu. Hạn chế chất lượng chính là
VIEW_PRIVATE_NOTE qua Whisper đạt 0/5 dù transcript chuẩn đạt 5/5; kết quả này
được giữ nguyên, không chỉnh rule theo test.