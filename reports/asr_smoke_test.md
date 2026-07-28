# Whisper Small CPU smoke test

- Ngày chạy: 2026-07-28
- Backend: `faster-whisper 1.2.1`
- Model: multilingual Whisper Small
- Device/compute: CPU, INT8
- Language/task: `vi`, `transcribe`
- Samples: ba file WAV từ `dangvansam/viet-asr` (Apache-2.0), xem
  `data/samples/asr_smoke/SOURCES.md`

| Audio | Raw transcript | Normalized transcript | Latency (ms) | Success |
|---|---|---|---:|---|
| `vi_01.wav` | Mời các đồng chí và các bạn nghe chương trình phát hành Quân đội Nhân Dân. | mời các đồng chí và các bạn nghe chương trình phát hành quân đội nhân dân | 37977.238 | true |
| `vi_02.wav` | với bộ trưởng, bộ thông tin và truyền thông Nguyễn Mạnh Hùng. | với bộ trưởng bộ thông tin và truyền thông nguyễn mạnh hùng | 11657.892 | true |
| `vi_03.wav` | 875, 35647 | 875 35647 | 6636.258 | true |

Đây chỉ là smoke test để xác nhận model, output contract và normalizer hoạt
động end-to-end. Không dùng các số liệu này làm WER/CER chính thức vì repository
nguồn không cung cấp reference transcript đi kèm các file đã chọn. Latency lần
đầu cao hơn do khởi tạo model; đánh giá chính thức cần warm-up trước khi đo.
