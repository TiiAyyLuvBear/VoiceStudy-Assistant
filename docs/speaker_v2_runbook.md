# Speaker v2

Speaker v2 được tạo từ `data/metadata/data_inventory.csv` với seed 42.
Toàn bộ audio trong ASR v2 bị loại trước khi chọn speaker và audio.
Speaker v1 không bị sửa.

## Phân bổ

### SVM closed-set

- 9 speaker.
- Mỗi speaker có 5 enrollment. Train/validation/test được phân bổ gần
  70%/15%/15% theo lượng audio khả dụng của từng người.
- Tổng: 45 enrollment, 600 train, 129 validation và 128 test.
- Theo speaker: 64–72 train, 13–16 validation và 14–15 test.
- Mọi audio SVM v1 của 9 speaker được giữ nguyên vai trò cũ. Audio test hoặc
  validation v1 không bị chuyển vào train v2.
- Speaker `5fb84feecf0626000b30cbd3` không được chọn vì chỉ có 65 audio hợp lệ,
  thấp hơn 75 audio tối thiểu để tham gia đầy đủ bốn split.

### Speaker-disjoint test

- 8 enrolled speaker.
- Mỗi người có 5 enrollment và 25 query.
- 8 unknown speaker, tổng cộng 109 unknown query.
- Phân bố unknown theo speaker: 20, 20, 20, 18, 18, 9, 3 và 1.
- Enrolled và unknown không trùng speaker; toàn bộ các split không trùng audio.

Không tạo cosine validation mới vì số speaker độc lập còn lại không đủ để vừa
đạt 8+8 speaker cho validation, vừa đạt 8+8 speaker cho test. Không được dùng
cosine test v2 để chọn threshold. Khi đánh giá model hiện tại, giữ threshold đã
khóa từ validation v1.

## Tạo lại split

```powershell
python -B -m scripts.build_speaker_v2_splits
```

Builder cập nhật component `speaker` trong
`data/processed/v2/split_manifest.json` và bảo toàn component `asr`.

## Trích xuất embedding v2

```powershell
python -B -m scripts.extract_all_embeddings `
  --metadata-dir data/processed/v2/metadata `
  --audio-root data/audio `
  --embedding-dir data/embeddings/v2 `
  --output data/processed/v2/embedding_metadata.csv `
  --protocol-file svm_closed_set_enrollment.csv `
  --protocol-file svm_closed_set_train.csv `
  --protocol-file svm_closed_set_validation.csv `
  --protocol-file svm_closed_set_test.csv `
  --protocol-file cosine_test_enrollment.csv `
  --protocol-file cosine_test_query.csv `
  --protocol-file cosine_test_unknown.csv
```

Model và kết quả v2 phải ghi vào `models/experimental/v2/` và
`experiments/v2/`; không ghi đè artifact v1.
