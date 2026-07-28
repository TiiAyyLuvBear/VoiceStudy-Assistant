# Member 2 runbook — ASR and command NLU

## NLU

Nhập một câu bất kỳ:

```powershell
python -m scripts.parse_command_text --reference-date 2026-07-28
```

Kiểm tra trùng và tái lập metric:

```powershell
python -m scripts.check_command_duplicates
python -m scripts.evaluate_nlu data/metadata/command_validation.csv
python -m scripts.evaluate_nlu data/metadata/command_test.csv
```

## ASR official evaluation

Khi nhận `data_inventory.csv` từ Thành viên 1:

```powershell
python -m scripts.build_asr_splits --inventory data_inventory.csv
python -m scripts.evaluate_asr data/metadata/asr_validation.csv
```

Chỉ sau khi đã chốt model/normalization trên validation mới chạy:

```powershell
python -m scripts.evaluate_asr data/metadata/asr_test.csv
```

Mỗi lần inference được checkpoint vào `reports/asr/*_predictions.csv`; nếu bị
ngắt có thể chạy lại và script sẽ resume. Summary JSON gồm WER, CER, mean/p95
latency và số file ASR lỗi.

## Data freeze

Kiểm tra checksum chưa thay đổi:

```powershell
python -m scripts.freeze_member2_data
```

Không dùng `--force` nếu chưa review lý do dữ liệu frozen bị thay đổi.
