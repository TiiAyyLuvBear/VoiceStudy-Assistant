# Shared-code rules

- Shared utilities belong in `src/utils/`.
- Before adding a helper, search `src/utils/` and reuse an existing utility.
- Import shared utilities with `from src.utils...`; do not copy or redefine them
  inside feature modules.
- Keep utilities dependency-light and domain-neutral. Speaker-, ASR-, or
  database-specific logic remains in its own domain package.
- Add reusable helper exports to `src/utils/__init__.py` and tests when behavior
  changes.



FROZEN không có nghĩa là không được phát triển tiếp. Nó có nghĩa là không sửa bộ v1, để kết quả cũ luôn tái lập được. Khi lấy thêm dữ liệu, hãy tạo v2.

  Có hai trường hợp:

  ### 1. Chỉ muốn đánh giá model hiện tại tốt không

  - Giữ nguyên model đã train bằng v1.
  - Lấy dữ liệu chưa từng xuất hiện trong train/validation/test v1.
  - Tạo data/processed/v2/metadata/.
  - Tạo test set lớn hơn, ví dụ:
      - 20–30 mẫu cho mỗi người đã biết.
      - Nhiều người lạ hơn.

  - Không train hoặc chỉnh threshold bằng test v2.
  - Chạy model hiện tại trên test v2 đúng một lần.

  Kết quả này cho biết model v1 có tổng quát tốt không. Nếu xem lỗi rồi sửa model, bộ test v2 đã gián tiếp được dùng để phát triển và nên xem nó như
  validation; lần đánh giá cuối cần một holdout mới.

  ### 2. Muốn tăng dữ liệu và train lại

  Tạo toàn bộ bộ dữ liệu v2:

  data/processed/
  ├── v1/                  # giữ nguyên, FROZEN
  └── v2/
      ├── metadata/
      │   ├── svm_closed_set_train.csv
      │   ├── svm_closed_set_validation.csv
      │   ├── svm_closed_set_test.csv
      │   ├── cosine_test_enrollment.csv
      │   ├── cosine_test_query.csv
      │   └── cosine_test_unknown.csv
      └── split_manifest.json

  Quy trình:

  1. Chọn dữ liệu v2.
  2. Tách train/validation/test trước.
  3. Train chỉ bằng train.
  4. Chọn C và threshold bằng validation.
  5. Khóa cấu hình và model.
  6. Chỉ sau đó mới chạy test.
  7. Ghi checksum rồi đặt v2 thành FROZEN.

  Model mới cũng nên lưu riêng, chẳng hạn:

  models/experimental/v1/
  models/experimental/v2/

  ### Những điều không được làm

  - Không thêm dòng vào các CSV của v1.
  - Không chuyển mẫu từ test v1 sang train.
  - Không dùng kết quả test để chọn threshold.
  - Không ghi đè model v1 bằng model mới.
  - Không gọi bộ dữ liệu mới là v1 vì kết quả sẽ không còn tái lập được.

  ### Dùng dữ liệu chưa phân bổ thế nào?

  Hiện có thể dùng các mẫu chưa phân bổ để tạo v2:

  - 797 mẫu còn lại của 10 người SVM.
  - 418 mẫu UNUSED từ 16 người nói gốc, phù hợp để tăng số người lạ.
  - Nên ưu tiên mẫu khác nội dung và điều kiện thu. Nếu không có thông tin phiên/thiết bị thì vẫn nên bổ sung một bộ ghi âm thực tế riêng.

  Tóm lại: giữ nguyên v1 làm mốc cũ, tạo v2 với dữ liệu nhiều hơn. FROZEN bảo vệ tính trung thực của thí nghiệm, không ngăn việc tạo phiên bản dữ liệu tốt
  hơn.