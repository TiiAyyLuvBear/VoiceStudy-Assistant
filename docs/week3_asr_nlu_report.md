# Tuần 3 — Đánh giá ASR và Command Analysis

Phụ trách: **Trần Hữu Lộc**

## 1. Cam kết test và cấu hình khóa

Cấu hình được khóa lúc `2026-08-08T17:09:51+07:00` trong
`reports/asr_nlu_test_config.json`, trước khi chạy lại command test hoàn chỉnh.
Snapshot lưu SHA-256 của cấu hình, dữ liệu test, ASR/NLU normalizer, Whisper
wrapper, intent rules, entity rules và evaluator. Snapshot đã được kiểm tra lại
ngay trước command inference. Kết quả ASR test được giữ nguyên vì dữ liệu ASR,
cấu hình và code ASR/NLU không đổi. Lượt command audio hoàn chỉnh được chạy
lại bằng `--no-resume` để không dùng transcript của audio cũ.

Quy tắc của lượt test:

- không chỉnh Whisper config, normalization, intent rule hoặc entity rule theo
  kết quả test;
- không chạy lại để lựa chọn một kết quả đẹp hơn;
- audio command thiếu được ghi `not_recorded`, không thay thế hoặc nội suy;
- năm nhãn ứng dụng là nhãn do dự án định nghĩa, không được khẳng định là nhãn
  intent gốc của Speech-MASSIVE;
- `source_intent` trong `asr_test.csv` chỉ là metadata tham khảo và không được
  dùng làm ground truth cho năm nhãn ứng dụng.

Whisper được khóa ở `faster-whisper-small`, revision cache
`536b0662742c02347bc0e980a01041f333bce120`, tiếng Việt, CPU, int8, beam size
10, VAD bật và `condition_on_previous_text=false`.

## 2. Dữ liệu test

| Phần | Số mẫu | Trạng thái |
|---|---:|---|
| ASR test | 125 audio | Đủ 125/125, tất cả file tồn tại |
| Command test bằng transcript chuẩn | 30 câu | Đủ 30/30 |
| Command test bằng audio | 30 câu | Đủ 30/30, tất cả file đã được kiểm định |
| OUT_OF_SCOPE | 10 câu | Đủ audio 10/10 |

ASR test là project test được trích từ dữ liệu Speech-MASSIVE tiếng Việt.
Command test là bộ câu lệnh riêng của ứng dụng gồm bốn functional intents và
`OUT_OF_SCOPE`.

## 3. Kết quả ASR

Whisper chạy mới từ đầu trên toàn bộ 125 audio, không resume prediction cũ.

| Chỉ số | Kết quả |
|---|---:|
| Successful / total | 125 / 125 |
| Failure | 0 |
| WER | 0.225352 (22.535%) |
| CER | 0.147067 (14.707%) |
| Word edits / reference words | 240 / 1,065 |
| Character edits / reference characters | 524 / 3,563 |
| Mean latency | 6,242.272 ms |
| P95 latency | 11,227.857 ms |
| Median latency (phân tích bổ sung) | 4,980.276 ms |

Mẫu đầu tiên có latency 82,300.2 ms do gồm chi phí warm-up/model load. Mean
latency khi bỏ riêng mẫu warm-up là 5,628.902 ms, nhưng đây chỉ là phân tích bổ
sung; kết quả chính thức phía trên vẫn giữ nguyên đủ 125 mẫu.

Có 34/125 mẫu đạt WER bằng 0. Hai lỗi lớn nhất cần ghi nhận là audio ID `13157`
và `11496`: Whisper đều nhận thành câu kêu gọi đăng ký kênh dài, trong khi
reference lần lượt rất ngắn. Đây là dấu hiệu cần kiểm tra thủ công khả năng
audio/reference không khớp hoặc audio chứa nội dung ngoài transcript; không sửa
hoặc loại hai mẫu khỏi kết quả test.

## 4. Intent test

### 4.1 Transcript chuẩn

| Chỉ số | Kết quả |
|---|---:|
| Overall 5-label accuracy | 30/30 = 100% |
| Functional 4-intent accuracy | 20/20 = 100% |
| OUT_OF_SCOPE rejection | 10/10 = 100% |

### 4.2 Audio → Whisper → Intent

Các accuracy sau tính trên đủ 30 audio; cả 30 mẫu đều ASR thành công.

| Intent | Đúng / đã đánh giá | Accuracy |
|---|---:|---:|
| GET_TIME | 4 / 5 | 80.00% |
| VIEW_SCHEDULE | 4 / 5 | 80.00% |
| ADD_SCHEDULE | 4 / 5 | 80.00% |
| VIEW_PRIVATE_NOTE | 0 / 5 | 0.00% |
| OUT_OF_SCOPE | 10 / 10 | 100.00% |
| **Overall 5 labels** | **22 / 30** | **73.33%** |
| **Bốn functional intents** | **12 / 20** | **60.00%** |

Coverage audio là 30/30 = 100%; không còn mẫu thiếu hoặc failure.

Tám lỗi intent quan sát được:

- `TST0004`: “mấy giờ” bị Whisper nhận thành “mấy lần”, GET_TIME thành
  OUT_OF_SCOPE;
- `TST0009`: “buổi gì” thành “bối gì”, VIEW_SCHEDULE thành OUT_OF_SCOPE;
- `TST0013`: “lập” thành “lặp”, ADD_SCHEDULE thành OUT_OF_SCOPE;
- `TST0016` đến `TST0020`: các từ khóa `ghi chú`, `note`, `riêng tư` hoặc
  `bảo mật` bị nhận sai, làm cả năm mẫu VIEW_PRIVATE_NOTE thành OUT_OF_SCOPE.

## 5. OUT_OF_SCOPE test

`out_of_scope_test_results.csv` có đủ 10 câu ngoài phạm vi. Text chuẩn và nhánh
audio đều từ chối đúng 10/10 (100%).

## 6. Entity test

| Nguồn | All-command exact match | Entity-bearing exact match | Date | Time | Title |
|---|---:|---:|---:|---:|---:|
| Transcript chuẩn | 30/30 = 100% | 10/10 = 100% | 10/10 = 100% | 5/5 = 100% | 5/5 = 100% |
| Audio → Whisper | 24/30 = 80% | 4/10 = 40% | 6/10 = 60% | 3/5 = 60% | 1/5 = 20% |

`All-command exact match` có nhiều câu không cần entity nên có thể làm kết quả
trông cao hơn. Chỉ số phản ánh đúng các câu cần entity là entity-bearing exact
match: 4/10 = 40% trên đủ audio test.

## 7. Hạn chế và kết luận

- ASR test, Intent/Entity trên transcript chuẩn và command audio test đã hoàn
  thành đầy đủ.
- Manifest có đủ 60/60 command audio (30 validation, 30 test); báo cáo kiểm
  định không phát hiện lỗi định dạng, mức tín hiệu, trùng nội dung hoặc leakage.
- Rule-based NLU đạt 100% trên transcript chuẩn nhưng nhạy với lỗi từ khóa của
  Whisper, đặc biệt là VIEW_PRIVATE_NOTE.
- Kết quả OUT_OF_SCOPE audio đạt 10/10; VIEW_PRIVATE_NOTE audio đạt 0/5 và là
  hạn chế chính còn lại.
- Không có rule hoặc cấu hình nào được chỉnh sau khi xem test result.

## 8. Artifact bàn giao

- `reports/asr_nlu_test_config.json`
- `reports/asr/asr_test_predictions.csv`
- `reports/asr/asr_test_metrics.json`
- `reports/nlu/intent_test_ground_truth.csv`
- `reports/nlu/intent_test_whisper.csv`
- `reports/nlu/intent_test_metrics.json`
- `reports/nlu/out_of_scope_test_results.csv`
- `reports/nlu/entity_test_results.csv`

