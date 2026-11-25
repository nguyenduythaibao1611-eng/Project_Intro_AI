# BỘ 4 GITHUB ISSUES CHI TIẾT CHO CÁC THÀNH VIÊN - TUẦN 4

---

## ISSUE #1: [Feature][DL] Huấn luyện Bi-LSTM và Tinh chỉnh (Fine-tuning) PhoBERT v2 trên GPU

* **Assignee:** `@hoang` (Từ Văn Huy Hoàng)
* **Labels:** `enhancement`, `deep-learning`, `gpu`, `model-training`, `high-priority`
* **Milestone:** Tuần 4 - Deep Learning & Web Demo
* **Branch:** `feature/w4-deep-learning-hoang`
* **Target Branch for PR:** `develop`
* **Deadline:** 23:59 Thứ 4

### 1. Bối cảnh & Mục tiêu
Ở Tuần 3, nhóm đã thiết lập mốc trần hiệu năng Baseline (LinearSVC đạt Macro F1 = 65.99%, Accuracy = 75.90%) và dựng sẵn khung kiến trúc PyTorch trong `src/dataset.py` và `src/models.py`. Ở Tuần 4, nhiệm vụ của bạn là đưa các mô hình Deep Learning vào huấn luyện thực tế trên GPU (Google Colab / Kaggle), áp dụng hàm mất mát có trọng số (Weighted Cross-Entropy Loss) để giải quyết bài toán mất cân bằng nhãn (RQ3), và phá vỡ mốc trần Baseline với chỉ tiêu **Macro F1 >= 75.0%**.

### 2. Phân vùng tệp mã nguồn (File Boundaries)

#### ✅ Các tệp ĐƯỢC PHÉP tạo mới hoặc chỉnh sửa:
* `notebooks/03_train_bilstm.ipynb` (Notebook huấn luyện Bi-LSTM)
* `notebooks/04_train_phobert.ipynb` (Notebook fine-tuning PhoBERT v2 trên GPU)
* `src/models.py` (Hoàn thiện lớp `PhoBERTClassifier` và `BiLSTMClassifier`)
* `src/train.py` (Script huấn luyện chuẩn hóa có checkpointing)
* `src/inference.py` (Module suy diễn độc lập xuất kết quả nhãn & % xác suất cho Web Demo)
* `models/best_bilstm.pt` (Checkpoint Bi-LSTM tối ưu)
* `models/best_phobert.pt` (Checkpoint PhoBERT tối ưu)
* `tests/test_inference.py` (Unit tests cho hàm suy diễn)

#### ⛔ CÁC TỆP TUYỆT ĐỐI ĐÓNG BĂNG - KHÔNG ĐƯỢC CHẠM VÀO (FROZEN FILES):
* `src/preprocess.py` (Do Hiếu phụ trách)
* `app.py` (Do Minh phụ trách)
* `src/evaluate.py` (Do Kiệt phụ trách)
* `data/processed/clean_shopee_reviews.csv` (Dữ liệu gốc đóng băng)
* `models/*.pkl` (Mô hình Baseline cũ đã nghiệm thu)

### 3. Yêu cầu kỹ thuật & Checklist thực thi
- [ ] **Huấn luyện Bi-LSTM 2 chiều:**
  - Kế thừa `src/models.py`, `embedding_dim=128`, `hidden_dim=128`, `dropout=0.3`.
  - Tích hợp PyTorch `pack_padded_sequence` để tối ưu thời gian huấn luyện.
- [ ] **Fine-tuning PhoBERT v2:**
  - Nạp pre-trained `vinai/phobert-base-v2` từ thư viện `transformers`.
  - Thiết lập Optimizer `AdamW` (learning rate = 2e-5 cho backbone, 1e-3 cho classification head).
  - Áp dụng Linear Warmup Scheduler (10% warmup steps).
- [ ] **Hiện thực hóa hàm mất mát Weighted Loss (RQ3):**
  - Tính trọng số: `w_c = N / (K * N_c)` trên 18.520 mẫu train.
  - Đưa vào `nn.CrossEntropyLoss(weight=class_weights.to(device))` nhằm phạt nặng lỗi ở lớp thiểu số Trung tính (NEU).
- [ ] **Xây dựng module suy diễn `src/inference.py`:**
  - Viết hàm `predict_sentiment(text: str, model_type="phobert") -> dict` trả về nhãn `POS/NEU/NEG`, độ tin cậy `confidence` và thời gian suy diễn `latency_ms`.
- [ ] **Viết Unit Test & Export:**
  - Tạo `tests/test_inference.py` kiểm tra suy diễn với câu đơn và câu rỗng.

### 4. Tiêu chuẩn nghiệm thu (Definition of Done)
- [ ] PhoBERT đạt Macro F1 trên tập kiểm thử 5.292 mẫu >= 75.0%.
- [ ] File trọng số `models/best_phobert.pt` được xuất an toàn.
- [ ] Thời gian suy diễn dưới 100ms/câu.
- [ ] Chạy lệnh `pytest tests/test_inference.py` đạt 100% pass.
- [ ] Tạo Pull Request vào nhánh `develop` và không gây merge conflicts.

---

## ISSUE #2: [Feature][Data] Tối ưu hóa Pipeline Tiền xử lý & Tokenizer PhoBERT cho Web Demo

* **Assignee:** `@hieu` (Bùi Đình Hiếu)
* **Labels:** `data-engineering`, `preprocessing`, `tokenizer`, `nlp`
* **Milestone:** Tuần 4 - Deep Learning & Web Demo
* **Branch:** `feature/w4-preprocessing-optimization-hieu`
* **Target Branch for PR:** `develop`
* **Deadline:** 23:59 Thứ 4

### 1. Bối cảnh & Mục tiêu
Pipeline tiền xử lý text hiện tại cần được tối ưu hóa để đảm bảo độ trễ suy diễn cực thấp (< 2ms/câu) khi tích hợp vào giao diện Web Demo thực tế. Ngoài ra, để giao diện của Minh có thể giải thích trực quan cho giảng viên và hội đồng thấy thuật toán đã chuẩn hóa teencode và loại bỏ nhiễu như thế nào, bạn cần bổ sung hàm bóc tách sai khác (diff inspection) và hoàn thiện module Tokenizer tương thích chuẩn của PhoBERT.

### 2. Phân vùng tệp mã nguồn (File Boundaries)

#### ✅ Các tệp ĐƯỢC PHÉP tạo mới hoặc chỉnh sửa:
* `src/preprocess.py` (Tối ưu hóa regex, bổ sung từ điển teencode Shopee mới)
* `src/tokenizer_utils.py` (Module tiền xử lý và encoding chuỗi tương thích PhoBERT)
* `tests/test_preprocess.py` (Bổ sung test case kiểm tra tốc độ và các trường hợp biên)

#### ⛔ CÁC TỆP TUYỆT ĐỐI ĐÓNG BĂNG - KHÔNG ĐƯỢC CHẠM VÀO (FROZEN FILES):
* `src/models.py`, `src/train.py`, `src/dataset.py` (Do Hoàng phụ trách)
* `app.py` (Do Minh phụ trách)
* `src/evaluate.py` (Do Kiệt phụ trách)
* `data/processed/clean_shopee_reviews.csv` (Dữ liệu chuẩn Tuần 2)

### 3. Yêu cầu kỹ thuật & Checklist thực thi
- [ ] **Tối ưu tốc độ tiền xử lý `src/preprocess.py`:**
  - Pre-compile toàn bộ biểu thức chính quy (`re.compile`).
  - Dùng bảng băm cấu trúc O(1) tra cứu từ điển teencode.
  - Đảm bảo thời gian làm sạch chuỗi < 2 mili-giây/câu.
- [ ] **Viết hàm bóc tách thay đổi `clean_text_with_diff(raw_text: str)`:**
  - Đầu ra trả về `(clean_text, changes_list)` ghi nhận chi tiết: từ gốc, từ sau chuẩn hóa, và loại biến đổi (teencode, elongation, emoji, punctuation).
- [ ] **Hoàn thiện module Tokenizer `src/tokenizer_utils.py`:**
  - Nạp `AutoTokenizer.from_pretrained("vinai/phobert-base-v2")`.
  - Viết hàm `encode_reviews(text_list, max_len=64)` xuất PyTorch tensor chuẩn `input_ids` và `attention_mask`.
- [ ] **Cập nhật Unit Test:**
  - Bổ sung các bài test về corner cases: chuỗi chỉ chứa icon, chuỗi rỗng, số điện thoại dài, chuỗi teencode lồng nhau.

### 4. Tiêu chuẩn nghiệm thu (Definition of Done)
- [ ] Hàm `clean_text_with_diff` hoạt động ổn định, không ném exception trong mọi tình huống.
- [ ] Tốc độ làm sạch văn bản đạt trung bình dưới 2ms/câu.
- [ ] Chạy lệnh `pytest tests/test_preprocess.py` đạt 100% pass.
- [ ] Tạo Pull Request vào nhánh `develop` và không gây merge conflicts.

---

## ISSUE #3: [Feature][App] Xây dựng Giao diện Web Live Demo Streamlit (Side-by-Side Comparison)

* **Assignee:** `@minh` (Đặng Đức Minh)
* **Labels:** `frontend`, `streamlit`, `web-demo`, `ui-ux`, `integration`
* **Milestone:** Tuần 4 - Deep Learning & Web Demo
* **Branch:** `feature/w4-streamlit-web-minh`
* **Target Branch for PR:** `develop`
* **Deadline:** 23:59 Thứ 6

### 1. Bối cảnh & Mục tiêu
Sản phẩm cốt lõi của Tuần 4 phục vụ buổi bảo vệ đồ án là một ứng dụng Web tương tác thời gian thực bằng Streamlit. Ứng dụng này sẽ nạp trọng số mô hình PhoBERT tối ưu từ Hoàng và module làm sạch văn bản từ Hiếu, cho phép giảng viên nhập thử bình luận tùy ý hoặc chọn các câu thử thách hiểm hóc để trực tiếp đối chứng hiệu năng giữa Baseline (LinearSVC) và Deep Learning (PhoBERT).

### 2. Phân vùng tệp mã nguồn (File Boundaries)

#### ✅ Các tệp ĐƯỢC PHÉP tạo mới hoặc chỉnh sửa:
* `app.py` (File mã nguồn chính của ứng dụng Streamlit)
* `requirements.txt` (Khai báo thêm `streamlit>=1.35.0`, `plotly>=5.20.0`)
* `static/style.css` (Tùy biến CSS giao diện phong cách tối giản màu trắng / xanh ngọc)
* `tests/test_app_integration.py` (Kiểm thử tải mô hình và luồng giao diện cơ bản)

#### ⛔ CÁC TỆP TUYỆT ĐỐI ĐÓNG BĂNG - KHÔNG ĐƯỢC CHẠM VÀO (FROZEN FILES):
* `src/preprocess.py` (Chỉ import hàm, không sửa code do Hiếu quản lý)
* `src/models.py`, `src/inference.py` (Chỉ import hàm, không sửa code do Hoàng quản lý)
* `src/evaluate.py` (Do Kiệt quản lý)
* `data/processed/*` (Không đụng vào dữ liệu gốc)

### 3. Yêu cầu kỹ thuật & Checklist thực thi
- [ ] **Tab 1: Phân tích Thời gian thực (Live Sentiment Analyzer):**
  - Ô nhập liệu văn bản tự do (Text Area).
  - 15 nút chọn nhanh các ca thử thách (Presets) do Kiệt chuẩn bị (Mỉa mai, Khen ngầm, Teencode).
  - Pipeline Inspector: Hiển thị câu trước và sau khi làm sạch (sử dụng `clean_text_with_diff` của Hiếu).
  - Khối hiển thị nhãn dự đoán POS / NEU / NEG màu sắc chuẩn UI và biểu đồ thanh Plotly thể hiện xác suất 3 lớp.
- [ ] **Tab 2: Chế độ So sánh Đối chứng (Side-by-Side Comparison):**
  - Chia màn hình làm 2 cột: Cột trái hiển thị Baseline (LinearSVC), Cột phải hiển thị PhoBERT.
  - Làm nổi bật trực quan trường hợp Baseline đoán sai nhưng PhoBERT sửa sai thành công.
- [ ] **Tab 3: Phân tích Tệp Hàng loạt (Batch CSV Analysis):**
  - Widget tải lên file CSV chứa cột `comment`.
  - Tiến trình phân loại (Progress bar) và xuất biểu đồ tròn phân phối cảm xúc khách hàng.
  - Nút tải xuống kết quả đã gán nhãn `download_predictions.csv`.
- [ ] **Tùy biến giao diện (UI/UX Styling):**
  - Thiết kế CSS tone trắng tối giản, đồng bộ với bộ Slide thuyết trình.

### 4. Tiêu chuẩn nghiệm thu (Definition of Done)
- [ ] Khởi chạy mượt mà bằng lệnh `streamlit run app.py` không phát sinh lỗi nạp model.
- [ ] Tích hợp trơn tru với `src/inference.py` và `src/preprocess.py`.
- [ ] Độ trễ phản hồi giao diện < 0.2s cho mỗi truy vấn dự đoán.
- [ ] Chạy lệnh `pytest tests/test_app_integration.py` đạt 100% pass.
- [ ] Tạo Pull Request vào nhánh `develop` và không gây merge conflicts.

---

## ISSUE #4: [QA][Benchmark] Thẩm định Đánh giá Mô hình & Soạn 15 Ca Kiểm thử Thách thức cho Web

* **Assignee:** `@kiet` (Châu Tuấn Kiệt)
* **Labels:** `evaluation`, `qa`, `benchmarking`, `metrics`, `documentation`
* **Milestone:** Tuần 4 - Deep Learning & Web Demo
* **Branch:** `feature/w4-evaluation-benchmark-kiet`
* **Target Branch for PR:** `develop`
* **Deadline:** 23:59 Thứ 6

### 1. Bối cảnh & Mục tiêu
Để báo cáo khoa học và phần thuyết trình bảo vệ trước hội đồng thuyết phục tuyệt đối, bạn chịu trách nhiệm thẩm định thực nghiệm toàn diện trên 5.292 mẫu test. Bạn sẽ đối chứng khoa học 3 thế hệ mô hình (Baseline vs Bi-LSTM vs PhoBERT), xuất ma trận nhầm lẫn độ phân giải cao 300 DPI, và soạn thảo 15 ca kiểm thử thực tế điển hình chứng minh việc PhoBERT khắc phục triệt để các điểm mù ngữ nghĩa của túi từ TF-IDF.

### 2. Phân vùng tệp mã nguồn (File Boundaries)

#### ✅ Các tệp ĐƯỢC PHÉP tạo mới hoặc chỉnh sửa:
* `src/evaluate.py` (Mở rộng hàm tính toán chỉ số đa lớp và latency)
* `docs/benchmark_comparison.csv` (Bảng số liệu tổng hợp đối chứng)
* `docs/phobert_confusion_matrix.png` (Ảnh ma trận nhầm lẫn 300 DPI)
* `docs/demo_challenge_cases.json` (Bộ 15 câu kiểm thử thách thức cho Web Demo)
* `tests/test_evaluate.py` (Unit tests cho các hàm tính toán chỉ số)

#### ⛔ CÁC TỆP TUYỆT ĐỐI ĐÓNG BĂNG - KHÔNG ĐƯỢC CHẠM VÀO (FROZEN FILES):
* `src/preprocess.py` (Do Hiếu phụ trách)
* `src/models.py`, `src/train.py` (Do Hoàng phụ trách)
* `app.py` (Do Minh phụ trách)

### 3. Yêu cầu kỹ thuật & Checklist thực thi
- [ ] **Bảng Benchmark so sánh 3 mô hình (`docs/benchmark_comparison.csv`):**
  - Chạy đánh giá trên tập Test độc lập 5.292 mẫu.
  - Các cột bắt buộc: Model Name, Accuracy, Macro Precision, Macro Recall, Macro F1, F1-POS, F1-NEU, F1-NEG, Inference Latency (ms).
  - 3 mô hình so sánh: LinearSVC (Baseline), Bi-LSTM, PhoBERT v2.
- [ ] **Ma trận nhầm lẫn PhoBERT 300 DPI (`docs/phobert_confusion_matrix.png`):**
  - Chuẩn hóa phần trăm theo hàng thực tế, hiển thị rõ độ chính xác lớp NEU đạt >= 55%.
- [ ] **Soạn thảo 15 ca thử thách (`docs/demo_challenge_cases.json`):**
  - 5 câu Mỉa mai (Sarcasm): Khen hình thức nhưng chê thực tế (Baseline đoán POS -> PhoBERT đoán đúng NEG).
  - 5 câu Khen ngầm (Implicit praise): Dùng từ ngữ tiêu cực nhưng mang hàm ý khen ngợi cực độ (Baseline đoán NEG -> PhoBERT đoán đúng POS).
  - 5 câu Teencode & Từ lóng Shopee phức tạp.
  - Format file JSON chuẩn hóa theo mẫu để bàn giao cho Minh trước 12:00 trưa Thứ 6 nạp vào Web Demo.

### 4. Tiêu chuẩn nghiệm thu (Definition of Done)
- [ ] Tệp `docs/benchmark_comparison.csv` đầy đủ số liệu thực nghiệm.
- [ ] Ảnh ma trận nhầm lẫn 300 DPI sắc nét, đúng tỷ lệ phần trăm.
- [ ] Tệp `docs/demo_challenge_cases.json` đúng cú pháp JSON và được bàn giao đúng hạn cho Minh.
- [ ] Chạy lệnh `pytest tests/test_evaluate.py` đạt 100% pass.
- [ ] Tạo Pull Request vào nhánh `develop` và không gây merge conflicts.
