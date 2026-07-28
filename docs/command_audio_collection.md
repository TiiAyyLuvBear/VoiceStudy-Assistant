# Command audio collection guide

Command audio dùng để kiểm tra end-to-end cho validation/test. Các file này
không được đưa vào Enrollment, SVM train, Speaker Identification hoặc Speaker
Verification benchmark.

## 1. Cài dependency và xem microphone

```powershell
python -m pip install -r requirements.txt
python -m scripts.record_command_audio --list-devices
```

## 2. Phân speaker

Dùng ID riêng cho người thu, ví dụ `cmdspk01`, `cmdspk02`, `cmdspk03`. Không
dùng `spk0001`… của Speech-MASSIVE để tránh hiểu nhầm hai nguồn dữ liệu.

Manifest đã có 60 prompt tại `data/commands/command_audio_manifest.csv`: 30
validation và 30 test. Mỗi command cần ít nhất một file audio.

Ví dụ một thành viên thu 10 câu validation:

```powershell
python -m scripts.record_command_audio `
  --speaker-id cmdspk01 `
  --split validation `
  --count 10 `
  --duration 6
```

Thành viên tiếp theo dùng speaker ID khác; script tự chọn các hàng `pending`
tiếp theo và lưu WAV 16 kHz mono vào đúng thư mục split.

Thu test sau khi quy trình validation đã ổn định:

```powershell
python -m scripts.record_command_audio `
  --speaker-id cmdspk02 `
  --split test `
  --count 10 `
  --duration 6
```

## 3. Kiểm tra chất lượng và leakage

```powershell
python -m scripts.validate_command_audio
```

Report được ghi vào `reports/command_audio_validation.csv`. Một hàng hợp lệ cần:

- tồn tại và đọc được;
- WAV PCM16, mono, 16 kHz;
- thời lượng 0.5–15 giây;
- không quá nhỏ;
- không trùng checksum;
- không xuất hiện trong bất kỳ Speaker split nào.

Nên nghe thủ công ngẫu nhiên ít nhất 3 file mỗi speaker trước khi dùng demo.

## 4. Nếu đã có file thu từ điện thoại

Chuyển file về WAV PCM16 mono 16 kHz, đặt vào `data/commands/audio/validation`
hoặc `test`, sau đó cập nhật đúng `speaker_id`, `audio_path`, metadata và
`status=recorded` trong manifest. Chạy validator ngay sau khi cập nhật.
