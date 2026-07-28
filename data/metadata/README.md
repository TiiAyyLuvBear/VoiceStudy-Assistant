# Metadata status

## Command datasets — frozen v1

- `command_development.csv`: 64 câu, dùng phát triển rule.
- `command_validation.csv`: 30 câu, gồm 5 câu cho mỗi intent chức năng và
  10 câu `OUT_OF_SCOPE`.
- `command_test.csv`: 30 câu, phân bổ giống validation và không dùng để chỉnh rule.
- `reference_date=2026-07-28` cố định cách diễn giải “hôm nay”, “ngày mai”.

Kiểm tra trùng câu sau normalization:

```powershell
python -m scripts.check_command_duplicates
```

## ASR datasets — waiting for Member 1 inventory

Không tạo hàng giả cho `asr_validation.csv` và `asr_test.csv`. Khi nhận
`data_inventory.csv` có đường dẫn audio, transcript, original split và cờ valid,
chạy:

```powershell
python -m scripts.build_asr_splits --inventory data_inventory.csv
```

Script chọn cố định 100 audio từ original validation và 125 audio từ official
test với seed 42; đồng thời loại transcript rỗng, audio invalid, đường dẫn trùng
và kiểm tra không overlap giữa hai tập.

Checksum đóng băng command data nằm trong `member2_split_manifest.json`. Hướng
dẫn chạy toàn bộ pipeline xem `docs/member2_runbook.md`.
