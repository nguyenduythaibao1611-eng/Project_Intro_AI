# ĐẶC TẢ KỸ THUẬT & PHÂN CÔNG NHIỆM VỤ TUẦN 4: DEEP LEARNING (PHOBERT & BI-LSTM) VÀ ỨNG DỤNG WEB DEMO

**Dự án:** Phân tích Cảm xúc Đánh giá Khách hàng Shopee (Vietnamese E-commerce Sentiment Analysis)  
**Học phần:** Artificial Intelligence Capstone 2025 — Viện VNUK, Đại học Đà Nẵng  
**Giảng viên hướng dẫn:** TS. Trần Thế Vũ (`vu.tran@vnuk.edu.vn`)  
**Quản lý dự án & Kiến trúc hệ thống:** Nguyễn Duy Thái Bảo  
**Nhánh hợp nhất bắt buộc (Target Branch):** `develop`  

---

## 1. TỔNG QUAN MỤC TIÊU & CHỈ TIÊU KỸ THUẬT (KPIs) TUẦN 4

Bước sang Tuần 4, toàn bộ đội ngũ chuyển đổi từ giai đoạn học máy cơ sở (Baseline) sang giai đoạn **Học sâu (Deep Learning) & Mô hình ngôn ngữ lớn tiếng Việt (PhoBERT Transformer)** kết hợp phát triển **Ứng dụng Web Live Demo (Streamlit)** để chuẩn bị cho buổi bảo vệ đồ án cuối kỳ ở Tuần 5.

### Chỉ tiêu kỹ thuật bắt buộc (Hard Constraints & KPIs):
1. **Chỉ số Macro F1-score:** Mô hình PhoBERT phải vượt mốc trần hiệu năng Baseline (65.99%), đạt **Macro F1 >= 75.0%** trên tập kiểm thử độc lập 5.292 mẫu test.
2. **Đột phá nhận diện lớp thiểu số (Trung tính - NEU):** Độ chính xác của lớp NEU phải được cải thiện từ 41.06% (Baseline) lên **>= 55.0%** nhờ kỹ thuật Weighted Cross-Entropy Loss (w_c = N / (K * N_c)).
3. **Khắc phục 3 điểm mù túi từ (Blindspots):** Chứng minh trên thực nghiệm PhoBERT phân loại chính xác các ca lỗi phân cực nghiêm trọng (|y_true - y_pred| = 2) gồm: Ngữ thái mỉa mai, Khen ngầm, và Ngữ cảnh dài đảo chiều ("tuy nhiên, nhưng mà").
4. **Ứng dụng Web Demo Streamlit (`app.py`):** Hoạt động mượt mà với độ trễ suy diễn (Inference Latency) < 0.2 giây mỗi câu, có chế độ đối chứng trực quan song song (Baseline vs PhoBERT).
5. **Độ ổn định mã nguồn (CI/CD Rule):** 100% mã nguồn phải vượt qua toàn bộ hệ thống Unit Tests tự động trước khi được phép merge vào nhánh `develop`.

---

## 2. QUY TẮC PHỐI HỢP & BẢO VỆ MÃ NGUỒN (GIT WORKFLOW & FROZEN FILES)

Để đảm bảo các thành viên làm việc độc lập mà không gây xung đột mã nguồn (merge conflicts) hoặc làm hỏng các module đã nghiệm thu ở Tuần 3, toàn bộ thành viên phải tuân thủ nghiêm ngặt các quy tắc sau:

### Quy trình Git Flow bắt buộc:
1. Mọi thành viên bắt đầu công việc bằng cách kéo mã nguồn mới nhất từ nhánh `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/w4-<ten-chuc-nang>-<ten-thanh-vien>
   ```
2. Tuyệt đối **KHÔNG ĐƯỢC COMMIT TRỰC TIẾP** vào nhánh `main` hoặc `develop`.
3. Sau khi hoàn thành và vượt qua toàn bộ Unit Tests cục bộ (`pytest tests/`), đẩy nhánh lên remote và tạo Pull Request (PR) trỏ vào `develop` để Leader Nguyễn Duy Thái Bảo phê duyệt và merge.

### Danh mục tệp ĐÓNG BĂNG TOÀN DỰ ÁN (Project-Wide Frozen Files - DO NOT TOUCH):
- `data/processed/clean_shopee_reviews.csv` (Tập dữ liệu sạch chuẩn 26.458 dòng đã chốt ở Tuần 2)
- `models/tfidf_vectorizer.pkl`, `models/logistic_regression.pkl`, `models/linear_svc.pkl` (Mốc chuẩn Baseline)
- `tests/test_dataset.py` (Bộ kiểm thử nạp dữ liệu PyTorch Tuần 3)

---

## 3. ĐẶC TẢ NHIỆM VỤ CHI TIẾT TỪNG THÀNH VIÊN

```
               [ĐỢT 1: THỨ 2 - THỨ 4]
┌──────────────────────────────────────────────────┐
│ Từ Văn Huy Hoàng: Train Bi-LSTM & PhoBERT GPU    │ ──> models/best_phobert.pt & src/inference.py
└──────────────────────────────────────────────────┘           │
┌──────────────────────────────────────────────────┐           │
│ Bùi Đình Hiếu: Tối ưu Tokenizer & Tiền xử lý     │ ──> src/preprocess.py (Tối ưu)
└──────────────────────────────────────────────────┘           │
                                                               ▼
               [ĐỢT 2: THỨ 5 - THỨ 6]
┌──────────────────────────────────────────────────┐
│ Đặng Đức Minh: Lập trình Web Demo Streamlit      │ <── Nhận Model từ Hoàng + Data từ Hiếu
└──────────────────────────────────────────────────┘           │
┌──────────────────────────────────────────────────┐           │
│ Châu Tuấn Kiệt: Đánh giá Benchmark & 15 Presets  │ <── Nhận kết quả dự đoán từ Hoàng
└──────────────────────────────────────────────────┘           │
                                                               ▼
               [ĐỢT 3: THỨ 7 - CHỦ NHẬT]
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Nguyễn Duy Thái Bảo: Code Review, Merge develop, Ráp Slide & Tổng duyệt Demo      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### PHẦN 1: TỪ VĂN HUY HOÀNG (Senior ML / Deep Learning Engineer)

* **Vai trò:** Phụ trách toàn bộ phần huấn luyện mô hình học sâu (Bi-LSTM) và tinh chỉnh mô hình ngôn ngữ lớn (PhoBERT v2) trên GPU, đóng gói mô hình cho Web Demo.
* **Nhánh Git phụ trách:** `feature/w4-deep-learning-hoang`
* **Deadline nộp code:** **23:59 Thứ 4**

#### Các tệp ĐƯỢC PHÉP tạo mới hoặc chỉnh sửa (Files to Edit/Create):
* `notebooks/03_train_bilstm.ipynb` (Notebook huấn luyện và log loss Bi-LSTM)
* `notebooks/04_train_phobert.ipynb` (Notebook fine-tuning PhoBERT v2 trên GPU Colab/Kaggle)
* `src/models.py` (Hoàn thiện lớp `PhoBERTClassifier` kế thừa `nn.Module`)
* `src/train.py` (Script huấn luyện chuẩn hóa có checkpointing)
* `src/inference.py` (Module suy diễn độc lập phục vụ Web Demo)
* `models/best_bilstm.pt` (File trọng số Bi-LSTM tối ưu)
* `models/best_phobert.pt` (File trọng số PhoBERT tối ưu)
* `tests/test_inference.py` (Unit test kiểm tra hàm dự đoán)

#### Các tệp TUYỆT ĐỐI KHÔNG ĐƯỢC CHẠM VÀO (FROZEN - DO NOT TOUCH):
* `src/preprocess.py` (Do Hiếu phụ trách)
* `app.py` (Do Minh phụ trách)
* `src/evaluate.py` (Do Kiệt phụ trách)
* `models/*.pkl` (Tệp mô hình Baseline cũ)

#### Đặc tả kỹ thuật chi tiết:
1. **Kiến trúc PhoBERT:** Sử dụng `vinai/phobert-base-v2` từ HuggingFace Transformers, bổ sung tầng phân loại tuần tự (Dropout 0.3 -> Linear(768, 256) -> ReLU -> Linear(256, 3)).
2. **Xử lý mất cân bằng nhãn (RQ3):** Bắt buộc tính toán vector trọng số lớp trên tập Train:
   w_c = N / (K * N_c)
   với N=18.520 (số mẫu train), K=3 lớp, và N_c là số mẫu tương ứng. Truyền `weight=weights.to(device)` vào `nn.CrossEntropyLoss()`.
3. **Chiến lược tối ưu:** Sử dụng `AdamW`, learning rate = 2e-5 cho PhoBERT backbone, 1e-3 cho classification head, áp dụng linear warmup scheduler trong 10% số bước đầu tiên, huấn luyện từ 4 đến 5 epochs với batch size 32.
4. **Hàm suy diễn `src/inference.py`:**
   ```python
   def predict_sentiment(text: str, model_type: str = "phobert") -> dict:
       """
       Đầu vào: Chuỗi văn bản thô từ người dùng.
       Đầu ra: Dictionary chuẩn định dạng:
       {
           "text_clean": str,
           "predicted_label": "POS" | "NEU" | "NEG",
           "probabilities": {"POS": float, "NEU": float, "NEG": float},
           "confidence": float,
           "inference_time_ms": float
       }
       """
   ```

#### Tiêu chuẩn nghiệm thu (Definition of Done - DoD):
- [ ] Huấn luyện thành công PhoBERT trên GPU, đạt Macro F1 trên tập test độc lập >= 75.0%.
- [ ] Export thành công file `models/best_phobert.pt` kèm `config.json`.
- [ ] Module `src/inference.py` chạy độc lập, suy diễn kết quả chính xác dưới 100ms.
- [ ] Vượt qua 100% tests trong `tests/test_inference.py`.

---

### PHẦN 2: BÙI ĐÌNH HIẾU (Data Engineer / Preprocessing Specialist)

* **Vai trò:** Tối ưu hóa hiệu năng đường ống tiền xử lý tiếng Việt, xây dựng module Tokenizer cho PhoBERT, và bổ sung tính năng theo dõi thay đổi teencode phục vụ giao diện Web Demo.
* **Nhánh Git phụ trách:** `feature/w4-preprocessing-optimization-hieu`
* **Deadline nộp code:** **23:59 Thứ 4**

#### Các tệp ĐƯỢC PHÉP tạo mới hoặc chỉnh sửa (Files to Edit/Create):
* `src/preprocess.py` (Tối ưu hóa regex, bổ sung từ điển teencode Shopee mới)
* `src/tokenizer_utils.py` (Module tiền xử lý và encoding chuỗi tương thích PhoBERT)
* `tests/test_preprocess.py` (Bổ sung test case kiểm tra tốc độ và các trường hợp biên)

#### Các tệp TUYỆT ĐỐI KHÔNG ĐƯỢC CHẠM VÀO (FROZEN - DO NOT TOUCH):
* `src/models.py`, `src/train.py`, `src/dataset.py` (Do Hoàng phụ trách)
* `app.py` (Do Minh phụ trách)
* `data/processed/clean_shopee_reviews.csv` (Dữ liệu gốc đóng băng)

#### Đặc tả kỹ thuật chi tiết:
1. **Tối ưu tốc độ tiền xử lý:** Biên dịch sẵn toàn bộ biểu thức chính quy (`re.compile()`) và sử dụng cấu trúc bảng băm (hashmap) cho từ điển teencode để thời gian làm sạch 1 câu review giảm xuống dưới 2 mili-giây.
2. **Hàm bóc tách biến đổi Teencode cho Giao diện Web:**
   Viết bổ sung vào `src/preprocess.py` hàm:
   ```python
   def clean_text_with_diff(raw_text: str) -> tuple[str, list[dict]]:
       """
       Đầu vào: Chuỗi bình luận thô chứa teencode.
       Đầu ra: 
         - clean_text: Chuỗi văn bản đã chuẩn hóa hoàn toàn.
         - changes: Danh sách các từ đã được can thiệp:
           [{"original": "sp", "normalized": "sản_phẩm", "type": "teencode"},
            {"original": "đẹpppp", "normalized": "đẹp", "type": "elongation"}]
       """
   ```
3. **Module Tokenizer PhoBERT (`src/tokenizer_utils.py`):** Tích hợp `AutoTokenizer.from_pretrained("vinai/phobert-base-v2")`, bổ sung hàm chuyển đổi câu tiếng Việt thành tensor `input_ids` và `attention_mask` chuẩn độ dài 64 token.

#### Tiêu chuẩn nghiệm thu (Definition of Done - DoD):
- [ ] Hàm `clean_text_with_diff()` hoạt động chính xác, không gây lỗi khi gặp chuỗi rỗng, số điện thoại dài, emoji lạ hoặc ký tự đặc biệt.
- [ ] Thời gian xử lý trung bình đạt < 2ms/câu.
- [ ] Vượt qua toàn bộ Unit Tests trong `tests/test_preprocess.py`.

---

### PHẦN 3: ĐẶNG ĐỨC MINH (AI/ML Application Engineer)

* **Vai trò:** Lập trình toàn bộ ứng dụng **Web Live Demo** bằng thư viện **Streamlit** (`app.py`), thiết kế giao diện tương tác chuyên nghiệp phục vụ buổi bảo vệ đồ án.
* **Nhánh Git phụ trách:** `feature/w4-streamlit-web-minh`
* **Deadline nộp code:** **23:59 Thứ 6**

#### Các tệp ĐƯỢC PHÉP tạo mới hoặc chỉnh sửa (Files to Edit/Create):
* `app.py` (Mã nguồn chính của ứng dụng Streamlit)
* `requirements.txt` (Khai báo thêm `streamlit>=1.35.0`, `plotly>=5.20.0`)
* `static/style.css` (Tùy biến CSS giao diện phong cách tối giản màu trắng / xanh ngọc)
* `tests/test_app_integration.py` (Kiểm thử tải mô hình và luồng giao diện cơ bản)

#### Các tệp TUYỆT ĐỐI KHÔNG ĐƯỢC CHẠM VÀO (FROZEN - DO NOT TOUCH):
* `src/preprocess.py` (Chỉ gọi `import`, không sửa code)
* `src/models.py`, `src/inference.py` (Chỉ nạp trọng số và gọi hàm dự đoán)
* `data/processed/*` (Không ghi đè dữ liệu)

#### Đặc tả kỹ thuật chi tiết:
Ứng dụng Web `app.py` phải được tổ chức thành 3 thẻ chức năng chính (Tabs):
1. **Tab 1: Phân tích Thời gian thực (Live Sentiment Analyzer):**
   * Ô nhập văn bản đa dòng (Text Area) cho phép người dùng dán bình luận Shopee bất kỳ.
   * Khu vực 15 nút bấm mẫu (Preset Buttons) chia theo 3 nhóm do Kiệt cung cấp (Mỉa mai, Khen ngầm, Teencode). Khi bấm vào, văn bản tự động điền vào ô nhập liệu.
   * Hộp thông tin trực quan đường ống (Pipeline Inspector): Hiển thị chuỗi ban đầu vs chuỗi sau khi làm sạch bởi module của Hiếu, gạch chân các từ teencode đã sửa.
   * Thẻ kết quả phân loại: Nhãn POS (Xanh lá), NEU (Vàng hổ phách), NEG (Đỏ thẫm) kèm biểu đồ thanh ngang hiển thị % độ tin cậy của cả 3 lớp.
2. **Tab 2: Chế độ So sánh Đối chứng (Side-by-Side: Baseline vs PhoBERT):**
   * Chia màn hình thành 2 cột:
     * Cột trái: Kết quả dự đoán của mô hình Baseline (LinearSVC).
     * Cột phải: Kết quả dự đoán của mô hình PhoBERT v2.
   * Làm nổi bật trường hợp Baseline đoán sai nhưng PhoBERT sửa sai thành công (ví dụ: câu mỉa mai).
3. **Tab 3: Phân tích Tệp Hàng loạt (Batch CSV Analysis):**
   * Cho phép kéo-thả tệp `.csv` chứa cột bình luận.
   * Xử lý hàng loạt và vẽ biểu đồ tròn (Pie chart) tỷ lệ hài lòng của khách hàng, cho phép tải kết quả về máy.

#### Tiêu chuẩn nghiệm thu (Definition of Done - DoD):
- [ ] Chạy thành công bằng lệnh `streamlit run app.py` trên môi trường nội bộ không có bất kỳ warning/lỗi nào.
- [ ] Tích hợp mượt mà với `src/inference.py` của Hoàng và `src/preprocess.py` của Hiếu.
- [ ] Giao diện trực quan, thẩm mỹ cao, tốc độ phản hồi mượt mà dưới 0.2s.

---

### PHẦN 4: CHÂU TUẤN KIỆT (Evaluation & QA Engineer)

* **Vai trò:** Thực hiện thẩm định định lượng chuyên sâu cho PhoBERT đối chứng với Bi-LSTM và Baseline, biên soạn bộ dữ liệu 15 câu kiểm thử thách thức cho Web Demo.
* **Nhánh Git phụ trách:** `feature/w4-evaluation-benchmark-kiet`
* **Deadline nộp code:** **23:59 Thứ 6**

#### Các tệp ĐƯỢC PHÉP tạo mới hoặc chỉnh sửa (Files to Edit/Create):
* `src/evaluate.py` (Mở rộng hàm đánh giá đa mô hình)
* `docs/benchmark_comparison.csv` (Bảng tổng hợp chỉ số đối chứng 3 thế hệ mô hình)
* `docs/phobert_confusion_matrix.png` (Ma trận nhầm lẫn chuẩn hóa 300 DPI của PhoBERT)
* `docs/demo_challenge_cases.json` (Bộ 15 câu kiểm thử đặc biệt cho Web Demo)
* `tests/test_evaluate.py` (Unit tests cho hệ thống tính toán chỉ số)

#### Các tệp TUYỆT ĐỐI KHÔNG ĐƯỢC CHẠM VÀO (FROZEN - DO NOT TOUCH):
* `src/preprocess.py`, `src/models.py`, `src/train.py`, `app.py`

#### Đặc tả kỹ thuật chi tiết:
1. **Bảng Benchmark so sánh 3 mô hình (`docs/benchmark_comparison.csv`):** Đánh giá trên tập kiểm thử độc lập 5.292 mẫu test:
   * **Cột chỉ số:** Model Name, Accuracy, Macro Precision, Macro Recall, Macro F1, F1-POS, F1-NEU, F1-NEG, Inference Latency (ms).
   * **3 Mô hình đối chứng:** Baseline (LinearSVC) vs Deep Learning (Bi-LSTM) vs Transformer (PhoBERT v2).
2. **Ma trận nhầm lẫn PhoBERT (`docs/phobert_confusion_matrix.png`):** Xuất ảnh độ phân giải cao 300 DPI, chuẩn hóa theo tỷ lệ phần trăm hàng thực tế, kiểm chứng độ chính xác lớp NEU đạt >= 55%.
3. **Bộ dữ liệu 15 câu thách thức (`docs/demo_challenge_cases.json`):**
   Biên soạn danh sách 15 câu review thực tế Shopee chia làm 3 chủ đề:
   * **Nhóm 1 (5 câu Mỉa mai - Sarcasm):** Khen hình thức nhưng chê bản chất.
   * **Nhóm 2 (5 câu Khen ngầm - Implicit Praise):** Dùng từ phủ định hoặc từ tiêu cực để khen.
   * **Nhóm 3 (5 câu Teencode & Từ lóng nặng):** Chứa nhiều biến thể phức tạp.
   Cấu trúc JSON:
   ```json
   {
     "id": 1,
     "category": "sarcasm",
     "text": "Shop phục vụ tận tình ghê, nhắn tin 1 ngày không thèm trả lời",
     "true_label": "NEG",
     "baseline_pred": "POS",
     "phobert_pred": "NEG",
     "explanation": "Baseline bị đánh lừa bởi từ 'tận tình'. PhoBERT bắt được ngữ cảnh phủ định 'không thèm trả lời' ở vế sau."
   }
   ```

#### Tiêu chuẩn nghiệm thu (Definition of Done - DoD):
- [ ] Hoàn thành bảng số liệu đối chứng khoa học và ma trận nhầm lẫn 300 DPI.
- [ ] Bàn giao file `docs/demo_challenge_cases.json` cho Minh nạp vào Web Demo đúng hạn Thứ 6.
- [ ] Vượt qua 100% tests trong `tests/test_evaluate.py`.

---

### PHẦN 5: NGUYỄN DUY THÁI BẢO (Leader / Project Manager & Integrator)

* **Vai trò:** Quản trị tổng thể dự án, phê duyệt Code Review trên GitHub, giải quyết xung đột mã nguồn khi merge vào `develop`, tích hợp toàn hệ thống và điều phối chuẩn bị nghiệm thu.
* **Thời gian thực hiện:** **Thứ 7 – Chủ Nhật**

#### Trách nhiệm kỹ thuật & Quản trị:
1. **Quản lý Pull Request & CI Pipeline:**
   * Kiểm tra code quality và đảm bảo không thành viên nào vi phạm quy tắc chạm vào các tệp đóng băng (Frozen Files).
   * Chạy kiểm thử tự động toàn diện: `pytest tests/ -v`. Tất cả các bài test (preprocess, dataset, models, inference, evaluate) phải PASS 100%.
   * Hợp nhất (Merge) lần lượt 4 nhánh feature vào nhánh `develop`.
2. **Tổng duyệt Web Live Demo:**
   * Cài đặt môi trường sạch, chạy thử nghiệm `streamlit run app.py` trên máy tính thuyết trình, kiểm tra tính ổn định của 15 nút test mẫu.
   * Chuẩn bị phương án dự phòng (quay video màn hình Demo 1 phút với độ phân giải Full HD).
3. **Cập nhật Tài liệu Nghiệm thu:**
   * Cập nhật Slide Canva Tuần 4 và Báo cáo tổng kết đồ án Capstone 2025.
   * Lên kịch bản phân công thuyết trình và bảo vệ đồ án cho Tuần 5.
