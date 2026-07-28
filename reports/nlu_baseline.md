# Rule-based NLU baseline — frozen v1

- Ngày đóng băng: 2026-07-28
- Phương pháp: rule-based intent classification + regular-expression entity extraction
- Intent: `GET_TIME`, `VIEW_SCHEDULE`, `ADD_SCHEDULE`,
  `VIEW_PRIVATE_NOTE`, `OUT_OF_SCOPE`
- Entity: `title`, `date`, `time`
- Reference date cố định cho câu tương đối: `2026-07-28`

## Dataset

| Split | Tổng câu | GET_TIME | VIEW_SCHEDULE | ADD_SCHEDULE | VIEW_PRIVATE_NOTE | OUT_OF_SCOPE |
|---|---:|---:|---:|---:|---:|---:|
| Development | 64 | 12 | 12 | 12 | 12 | 16 |
| Validation | 30 | 5 | 5 | 5 | 5 | 10 |
| Test | 30 | 5 | 5 | 5 | 5 | 10 |

Duplicate checker xác nhận không có transcript trùng trong hoặc giữa ba split
sau khi normalization.

## Kết quả

| Split | Intent accuracy | Entity exact match | Mismatch |
|---|---:|---:|---:|
| Development | 100% | 100% | 0 |
| Validation | 100% | 100% | 0 |
| Test | 100% | 100% | 0 |

Rule được chỉnh bằng development/validation rồi đóng băng trước khi chạy tập
test mới. Kết quả này đánh giá trên bộ câu lệnh riêng có phạm vi hẹp, không đại
diện cho khả năng hiểu tiếng Việt tổng quát.

## Lệnh tái lập

```powershell
python -m scripts.check_command_duplicates
python -m scripts.evaluate_nlu data/metadata/command_validation.csv
python -m scripts.evaluate_nlu data/metadata/command_test.csv
```
