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

## Kiểm thử toàn bộ phần Thành viên 2 hiện có

Các lệnh bên dưới chạy từ thư mục gốc của repository. Trên PowerShell, thiết
lập UTF-8 trước để tiếng Việt không bị hiển thị thành `\u...`:

```powershell
Set-Location 'C:\HỌC THỐNG KÊ\VoiceStudy-Assistant'
chcp 65001
$env:PYTHONUTF8 = "1"
```

### 1. Kiểm tra môi trường và cài dependency

```powershell
python --version
python -m pip install -r requirements.txt
```

Python hiện dùng để phát triển là 3.11. Whisper Small chạy CPU với INT8.

Kiểm tra nhanh cấu hình ASR:

```powershell
python -c "from src.asr.whisper_model import load_whisper_config; c=load_whisper_config(); print({'model': c.model_name, 'size': c.model_size, 'device': c.device, 'compute_type': c.compute_type, 'language': c.language})"
```

Kết quả cần có `whisper-small`, `small`, `cpu`, `int8` và `vi`.

### 2. Liệt kê và chạy toàn bộ unit test

Chỉ liệt kê các test mà không chạy:

```powershell
python -m pytest --collect-only -q
```

Chạy gọn toàn bộ test:

```powershell
python -m pytest -q
```

Kết quả hiện tại mong đợi:

```text
45 passed
```

Chạy và xem tên từng test:

```powershell
python -m pytest -v
```

### 3. Chạy test theo từng module

Whisper ASR wrapper:

```powershell
python -m pytest tests/test_whisper_model.py -v
```

Text normalization:

```powershell
python -m pytest tests/test_text_normalizer.py -v
```

Intent classifier — file bắt buộc của nhiệm vụ:

```powershell
python -m pytest tests/test_intent_classifier.py -v
```

Command parser — file bắt buộc của nhiệm vụ:

```powershell
python -m pytest tests/test_command_parser.py -v
```

Chạy riêng hai file test bắt buộc:

```powershell
python -m pytest `
  tests/test_intent_classifier.py `
  tests/test_command_parser.py `
  -v
```

Các test hỗ trợ/làm thêm:

```powershell
python -m pytest tests/test_asr_metrics.py -v
python -m pytest tests/test_command_audio_tools.py -v
python -m pytest tests/test_asr_split_builder.py -v
```

### 4. Kiểm tra cú pháp toàn bộ source và script

```powershell
python -m compileall -q src scripts tests
```

Không có output nghĩa là không phát hiện lỗi compile.

### 5. Chạy Whisper thật trên audio tiếng Việt

Chạy một audio để kiểm tra nhanh:

```powershell
python -m scripts.check_asr_samples `
  data/samples/asr_smoke/vi_01.wav
```

Chạy cả ba audio smoke test:

```powershell
python -m scripts.check_asr_samples `
  data/samples/asr_smoke/vi_01.wav `
  data/samples/asr_smoke/vi_02.wav `
  data/samples/asr_smoke/vi_03.wav
```

Mỗi output thành công cần có:

```json
{
  "model": "whisper-small",
  "language": "vi",
  "success": true,
  "error": null
}
```

Whisper Small chạy CPU có thể mất khoảng 10–40 giây mỗi file. Báo cáo smoke
test đã lưu tại `reports/asr_smoke_test.md`.

Kiểm tra xử lý file không tồn tại; lệnh này dự kiến trả `success=false` và exit
code khác 0:

```powershell
python -m scripts.check_asr_samples data/missing.wav
```

### 6. Thử Text Normalizer bằng câu mới

```powershell
python -c "from src.nlu.text_normalizer import normalize_text; text=input('Nhập câu: '); print('Kết quả:', normalize_text(text))"
```

Ví dụ nhập:

```text
Thêm lịch Học Máy lúc 8h30 sáng mai!!!
```

Kết quả mong đợi:

```text
thêm lịch học máy lúc 8 giờ 30 sáng mai
```

### 7. Thử Intent Classifier bằng câu mới

```powershell
python -c "from src.nlu.intent_classifier import classify_intent; print(classify_intent(input('Nhập câu: ')))"
```

Ví dụ:

```text
Mở nhạc cho tôi
```

Kết quả mong đợi là `OUT_OF_SCOPE`.

### 8. Thử Command Parser bằng câu mới

Nhập tương tác:

```powershell
python -m scripts.parse_command_text --reference-date 2026-07-28
```

Hoặc truyền câu trực tiếp:

```powershell
python -m scripts.parse_command_text `
  "Thêm lịch học xử lý ngôn ngữ lúc 8h30 sáng mai" `
  --reference-date 2026-07-28
```

Kết quả mong đợi:

```json
{
  "intent": "ADD_SCHEDULE",
  "entities": {
    "title": "học xử lý ngôn ngữ",
    "date": "2026-07-29",
    "time": "08:30"
  },
  "missing_fields": []
}
```

Thử câu ngoài phạm vi:

```powershell
python -m scripts.parse_command_text "Mở nhạc cho tôi"
```

Kết quả cần có `intent=OUT_OF_SCOPE`, `entities={}` và `missing_fields=[]`.

### 9. Kiểm tra số lượng command datasets

```powershell
(Import-Csv data/metadata/command_development.csv).Count
(Import-Csv data/metadata/command_validation.csv).Count
(Import-Csv data/metadata/command_test.csv).Count
```

Kết quả lần lượt phải là:

```text
64
30
30
```

Kiểm tra phân bố intent của từng split:

```powershell
Import-Csv data/metadata/command_development.csv |
  Group-Object intent |
  Select-Object Name,Count

Import-Csv data/metadata/command_validation.csv |
  Group-Object intent |
  Select-Object Name,Count

Import-Csv data/metadata/command_test.csv |
  Group-Object intent |
  Select-Object Name,Count
```

### 10. Kiểm tra command không trùng

```powershell
python -m scripts.check_command_duplicates
```

Kết quả mong đợi:

```json
{
  "duplicate_count": 0,
  "duplicates": []
}
```

### 11. Đánh giá NLU trên ba split

Development:

```powershell
python -m scripts.evaluate_nlu data/metadata/command_development.csv
```

Validation:

```powershell
python -m scripts.evaluate_nlu data/metadata/command_validation.csv
```

Test đã đóng băng:

```powershell
python -m scripts.evaluate_nlu data/metadata/command_test.csv
```

Kết quả hiện tại của mỗi split cần có:

```json
{
  "intent_accuracy": 1.0,
  "entity_exact_match": 1.0,
  "mismatch_count": 0
}
```

Không chỉnh rule hoặc normalization dựa trên kết quả test.

### 12. Kiểm tra command audio manifest

Xem tổng số hàng và trạng thái:

```powershell
$commandAudio = Import-Csv data/commands/command_audio_manifest.csv
$commandAudio.Count
$commandAudio |
  Group-Object status |
  Select-Object Name,Count
```

Trước khi thu, kết quả hiện tại là 60 hàng `pending`.

Xem 10 câu tiếp theo cần thu:

```powershell
Import-Csv data/commands/command_audio_manifest.csv |
  Where-Object status -eq "pending" |
  Select-Object -First 10 recording_id,split,expected_transcript,intent
```

### 13. Kiểm tra microphone và thu thử command audio

Liệt kê thiết bị input:

```powershell
python -m scripts.record_command_audio --list-devices
```

Lệnh dưới đây sẽ thật sự bật microphone, ghi file và thay đổi manifest. Chỉ chạy
khi sẵn sàng thu:

```powershell
python -m scripts.record_command_audio `
  --speaker-id cmdspk01 `
  --split validation `
  --count 1 `
  --duration 6 `
  --device 1
```

Sau khi thu, nghe lại file vừa tạo trong `data/commands/audio/validation/` và
kiểm tra manifest đã chuyển hàng tương ứng sang `recorded`.

### 14. Validate command audio

```powershell
python -m scripts.validate_command_audio
```

Report được ghi vào:

```text
reports/command_audio_validation.csv
```

Khi chưa thu đủ 60 audio, validator sẽ báo `not_recorded` và trả exit code khác
0; đây là trạng thái dự kiến. Khi hoàn tất, cần đạt 60/60 `recorded`, không có
file lỗi, checksum trùng hoặc `speaker_training_leakage`.

### 15. Kiểm tra dữ liệu command đã đóng băng

```powershell
python -m scripts.freeze_member2_data
```

Kết quả mong đợi khi dữ liệu chưa thay đổi:

```text
Frozen command checksums unchanged
```

Không dùng `--force` nếu chưa review và thống nhất thay đổi với nhóm.

### 16. Tạo ASR validation/test khi nhận dữ liệu Thành viên 1

Điều kiện đầu vào:

- có `data/metadata/data_inventory.csv`;
- các file trong cột `audio_path` thực sự tồn tại;
- có `original_split`, transcript và cờ audio valid.

Tạo hai split:

```powershell
python -m scripts.build_asr_splits `
  --inventory data/metadata/data_inventory.csv
```

Kiểm tra file và số lượng:

```powershell
Get-Item data/processed/v2/metadata/asr_validation.csv
Get-Item data/processed/v2/metadata/asr_test.csv
(Import-Csv data/processed/v2/metadata/asr_validation.csv).Count
(Import-Csv data/processed/v2/metadata/asr_test.csv).Count
```

Mặc định lấy toàn bộ bản ghi hợp lệ: 322 validation và 249 test. Split
`UNUSED` không được đưa vào. Bộ 100/125 trong `data/metadata/` là mốc v1
đã khóa và không bị ghi đè.

### 17. Chạy ASR official evaluation

Đo thử năm file validation trước:

```powershell
Measure-Command {
  python -m scripts.evaluate_asr `
    data/processed/v2/metadata/asr_validation.csv `
    --limit 5 `
    --output-dir reports/asr/v2
}
```

Chạy đủ validation; năm file đã thành công sẽ được resume:

```powershell
python -m scripts.evaluate_asr `
  data/processed/v2/metadata/asr_validation.csv `
  --output-dir reports/asr/v2
```

Sau khi đã chốt model và normalization mới chạy test:

```powershell
python -m scripts.evaluate_asr `
  data/processed/v2/metadata/asr_test.csv `
  --output-dir reports/asr/v2
```

Kiểm tra output:

```powershell
Get-ChildItem reports/asr/v2 -File |
  Select-Object Name,Length,LastWriteTime
```

Prediction CSV và summary JSON phải có WER, CER, mean/p95 latency và số file ASR
lỗi. Nếu inference bị ngắt, chạy lại đúng lệnh để resume checkpoint.

### 18. Bộ lệnh kiểm tra nhanh trước khi bàn giao

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
python -m scripts.check_command_duplicates
python -m scripts.evaluate_nlu data/metadata/command_validation.csv
python -m scripts.freeze_member2_data
python -m scripts.check_asr_samples data/samples/asr_smoke/vi_01.wav
```

Trước khi có dữ liệu Thành viên 1 và audio thu thật, bộ kiểm tra nhanh cần đạt:

- 45 unit test pass;
- không có lỗi compile;
- `duplicate_count=0`;
- validation intent/entity đạt kết quả đã ghi trong baseline;
- checksum command datasets không thay đổi;
- Whisper trả `success=true` trên audio smoke test.
