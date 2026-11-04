# 📋 KỊCH BẢN TỔNG THỂ & HƯỚNG DẪN THỰC THI TUẦN 3 (WEEK 3 MASTER SCRIPT)
**Dự án:** Phân tích Cảm xúc Đánh giá Khách hàng Shopee (Vietnamese Review Sentiment Analysis)  
**Học phần:** Artificial Intelligence Capstone 2025 — VNUK, Đại học Đà Nẵng  
**Giảng viên hướng dẫn:** TS. Trần Thế Vũ (vu.tran@vnuk.edu.vn)  
**Nhóm sinh viên thực hiện:** Thái Bảo (Leader), Tuấn Kiệt, Đức Minh, Đình Hiếu, Huy Hoàng  

---

## PHẦN 1: CHECKLIST HÀNH ĐỘNG KỸ THUẬT (TECHNICAL EXECUTION STEPS)

```
[Dữ liệu sạch Tuần 2: 26.458 dòng]
                 │
                 ▼
  [BƯỚC 1: Xây dựng Preprocessing Pipeline (`src/preprocess.py`)]
    ├── Chuẩn hóa chữ thường (lowercase), loại bỏ ký tự rác
    ├── Từ điển chuẩn hóa Teencode / Từ lóng (k -> không, sp -> sản phẩm)
    └── Tách từ ghép tiếng Việt (Word Segmentation: giao_hàng, chất_lượng)
                 │
                 ▼
  [BƯỚC 2: Chia tập dữ liệu phân tầng Stratified 70/15/15]
    ├── Tập Train: 70% (~18.520 câu) ──> Huấn luyện mô hình
    ├── Tập Val:   15% (~3.969 câu)  ──> Đánh giá chéo, theo dõi Loss
    └── Tập Test:  15% (~3.969 câu)  ──> Chấm điểm độc lập cuối cùng
                 │
                 ▼
  [BƯỚC 3: Huấn luyện Baseline Model (`notebooks/02_baseline_models.ipynb`)]
    ├── Trích xuất đặc trưng: TF-IDF (n-gram 1-2, max_features = 5.000)
    └── Phân loại: Logistic Regression & Linear Support Vector Machine (LinearSVC)
                 │
                 ▼
  [BƯỚC 4: Đo lường Benchmark & Xuất biểu đồ]
    ├── Tính Accuracy, Macro Precision, Macro Recall, Macro F1-score
    ├── Xuất ảnh Ma trận nhầm lẫn (Confusion Matrix) vào `docs/`
    └── Phân tích tại sao Baseline đoán sai ở lớp Neutral và Negative
                 │
                 ▼
  [BƯỚC 5: Thiết kế Kiến trúc Deep Learning cho Tuần 4 (`src/models.py`)]
    └── Khung mạng Bi-LSTM 2 chiều trên PyTorch sẵn sàng cho Tuần 4
```

---

## PHẦN 2: KỊCH BẢN THUYẾT TRÌNH BÁO CÁO TUẦN 3 (PRESENTATION TALKING POINTS)

*(Thời lượng đề xuất: 5 – 7 phút; dùng cho 5 – 6 slide PowerPoint theo chuẩn 5W1H và yêu cầu của Thầy Vũ)*

### Slide 1: Khung Quản trị Tuần 3 (Weekly Framework - 5W1H)
* **Người trình bày:** Thái Bảo (Leader)
* **Lời thoại mẫu:**
  > *"Kính thưa Thầy Vũ và các bạn, bước sang Tuần 3, nhóm bám sát mục tiêu **Data Preparation & Modeling Approach**. 
  > - **WHAT:** Nhóm bàn giao Pipeline tiền xử lý teencode hoàn chỉnh, bộ dữ liệu đã chia phân tầng 70/15/15, và kết quả đo lường mốc điểm chuẩn của mô hình Baseline.
  > - **WHEN:** Đạt 100% tiến độ tuần theo cam kết WBS, sẵn sàng bước vào khâu huấn luyện Deep Learning ở Tuần 4.
  > - **WHY:** Nhóm cần một mốc điểm chuẩn Baseline thực tế để có căn cứ định lượng trả lời Câu hỏi nghiên cứu số 2 (RQ2).
  > - **WHO:** Cả 5 thành viên đều đảm nhận đúng vai trò: dữ liệu, tiền xử lý, mô hình cơ sở, kiến trúc học sâu và đánh giá chỉ số."*

---

### Slide 2: Pipeline Tiền xử lý & Chuẩn hóa Teencode (Data Preprocessing Pipeline)
* **Người trình bày:** Đặng Đức Minh (Data Engineer)
* **Lời thoại mẫu:**
  > *"Dạ thưa Thầy, đối với dữ liệu đánh giá trên sàn Shopee, trở ngại lớn nhất là văn bản phi chuẩn mực. Nhóm đã xây dựng pipeline tiền xử lý qua 3 tầng:
  > 1. **Làm sạch cơ bản:** Đưa về chữ thường, xử lý khoảng trắng thừa và ký tự đặc biệt.
  > 2. **Chuẩn hóa Teencode:** Xây dựng từ điển ánh xạ hơn 100 từ lóng phổ biến trên sàn TMĐT như `k/ko` thành `không`, `sp` thành `sản phẩm`, `đc` thành `được`, `auth/rep` thành phân loại hàng.
  > 3. **Tách từ tiếng Việt (Word Segmentation):** Nối các từ ghép như `giao_hàng`, `thất_vọng`, `chất_lượng` để các mô hình sau này hiểu trọn vẹn cụm nghĩa thay vì tách rời từng âm tiết."*

---

### Slide 3: Chiến lược Chia tập Dữ liệu (Stratified Data Splitting)
* **Người trình bày:** Đặng Đức Minh hoặc Thái Bảo
* **Lời thoại mẫu:**
  > *"Nhóm áp dụng kỹ thuật **Stratified Sampling (Chia phân tầng)** trên toàn bộ 26.458 dòng sạch theo tỷ lệ **70% Train – 15% Validation – 15% Test**:
  > - **Tập Train (18.520 câu):** Cung cấp dữ liệu để mô hình học các trọng số.
  > - **Tập Validation (3.969 câu):** Dùng để tinh chỉnh siêu tham số và giám sát hiện tượng quá khớp (Overfitting).
  > - **Tập Test (3.969 câu):** Bộ đề thi độc lập 100%, chỉ đưa ra chấm điểm sau cùng.
  > - **Điểm cốt lõi:** Kỹ thuật phân tầng đảm bảo cả 3 tập đều giữ nguyên tỷ lệ nhãn gốc (POS ~64%, NEG ~21%, NEU ~15%), ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (Data Leakage)."*

---

### Slide 4: Biện luận Lựa chọn Mô hình (Model Selection Justification)
* **Người trình bày:** Châu Tuấn Kiệt (ML Engineer)
* **Lời thoại mẫu:**
  > *"Nhóm lựa chọn phương pháp tiếp cận theo mô hình bậc thang 3 cấp độ:
  > - **Tại sao chọn TF-IDF + Logistic Regression/SVM làm Baseline?** Đây là mô hình học máy kinh điển, tốc độ xử lý chỉ mất vài giây, làm mốc đối chứng khoa học để chứng minh giá trị của Deep Learning.
  > - **Tại sao chọn Bi-LSTM cho giai đoạn học sâu?** Khác với văn bản tiếng Anh, tiếng Việt có nhiều cấu trúc đảo ngữ và từ phủ định đứng trước tính từ (như 'không được đẹp cho lắm'). Bi-LSTM đọc đồng thời 2 chiều (Forward và Backward) giúp nắm bắt ngữ cảnh phụ thuộc xa này.
  > - **Tại sao chọn VinAI PhoBERT cho giai đoạn tối ưu?** PhoBERT là mô hình Transformer tiền huấn luyện trên 20GB báo chí tiếng Việt, cơ chế Self-Attention giúp hiểu sâu sắc ngôn ngữ mạng xã hội, câu đa nghĩa và mỉa mai."*

---

### Slide 5: Kết quả Thực nghiệm Mô hình Cơ sở (Baseline Model Metrics)
* **Người trình bày:** Bùi Đình Hiếu & Từ Văn Huy Hoàng
* **Lời thoại mẫu:**
  > *"Dạ thưa Thầy, đây là kết quả thực nghiệm của Baseline trên tập Test độc lập:
  > - **Accuracy:** Đạt khoảng 76 – 78%.
  > - **Macro F1-Score:** Đạt khoảng 68 – 71%.
  > - **Phân tích Ma trận nhầm lẫn (Confusion Matrix):** 
  >   - Mô hình nhận diện rất tốt nhãn Tích cực (POS) nhờ lượng mẫu áp đảo.
  >   - Tuy nhiên, Baseline bị nhầm lẫn rất nhiều ở lớp **Trung tính (NEU)** và **Tiêu cực (NEG)** (nhiều câu chê có từ phủ định bị đoán nhầm thành khen).
  >   - Đây chính là bằng chứng thực nghiệm rõ ràng nhất khẳng định: Học máy truyền thống không đủ khả năng nắm bắt ngữ cảnh phức tạp, và mở ra sự cần thiết của Bi-LSTM cùng hàm mất mát Weighted Loss trong Tuần 4."*

---

### Slide 6: Kế hoạch Hành động Tuần 4 (Action Plan for Week 4)
* **Người trình bày:** Thái Bảo (Leader)
* **Lời thoại mẫu:**
  > *"Trong Tuần 4, nhóm sẽ tập trung vào 4 nhiệm vụ cốt lõi:
  > 1. Xây dựng và huấn luyện mạng Bi-LSTM trên PyTorch.
  > 2. Thiết kế hàm mất mát **Weighted Cross-Entropy Loss** để phạt nặng các lỗi đoán sai ở lớp NEU và NEG, kéo chỉ số Macro F1 lên.
  > 3. Tinh chỉnh (Fine-tuning) mô hình **PhoBERT-base** trên GPU Google Colab.
  > 4. Thực hiện **Error Analysis** (Phân tích lỗi trên 50 mẫu câu mô hình đoán sai) để hoàn thiện câu trả lời cho cả 3 RQs."*

---

## PHẦN 3: BỘ CÂU HỎI VẤN ĐÁP PHÒNG THỦ (Q&A DEFENSE CHEAT SHEET)

Dưới đây là 4 câu hỏi Thầy Vũ rất hay hỏi và câu trả lời chuẩn xác nhất:

### ❓ Câu 1: "Tại sao nhóm phải tách từ ghép (Word Segmentation) bằng dấu gạch dưới thay vì để từng từ riêng?"
* **Trả lời:**  
  *"Dạ thưa Thầy, tiếng Việt là ngôn ngữ đơn lập, ý nghĩa thường nằm ở **từ ghép** gồm 2 hay nhiều âm tiết (ví dụ: `giao hàng`, `hài lòng`, `thất vọng`). Nếu không tách từ, máy tính sẽ hiểu `giao` và `hàng` là 2 từ độc lập hoàn toàn khác nghĩa. Việc nối thành `giao_hàng` giúp vector nhúng (Embedding) đại diện đúng một khái niệm ngữ nghĩa duy nhất ạ."*

### ❓ Câu 2: "Tại sao nhóm chia tỷ lệ 70% Train – 15% Val – 15% Test mà không dùng 80% – 20%?"
* **Trả lời:**  
  *"Dạ thưa Thầy, vì ở giai đoạn Deep Learning và fine-tune PhoBERT, nhóm cần một tập **Validation độc lập (15% ~ gần 4.000 câu)** để theo dõi hàm Loss qua từng Epoch và áp dụng kỹ thuật **Early Stopping** (dừng sớm khi mô hình có dấu hiệu Overfitting). Tập Test 15% còn lại được giữ bí mật hoàn toàn để làm bài thi cuối cùng, đảm bảo tính khách quan tuyệt đối ạ."*

### ❓ Câu 3: "TF-IDF có nhược điểm lớn nhất là gì mà nhóm bắt buộc phải chuyển sang Deep Learning?"
* **Trả lời:**  
  *"Dạ thưa Thầy, nhược điểm chí mạng của TF-IDF là **coi văn bản như một túi từ (Bag-of-Words)**, hoàn toàn bỏ qua **thứ tự xuất hiện của từ** và không hiểu được **ngữ cảnh ngữ nghĩa**. Ví dụ 2 câu: `'Hàng đẹp, không tệ'` và `'Hàng tệ, không đẹp'` có tập từ vựng y hệt nhau nên TF-IDF cho ra vector gần như giống nhau, nhưng ngữ nghĩa thực tế thì hoàn toàn đối lập. Bi-LSTM và PhoBERT sẽ giải quyết triệt để vấn đề thứ tự này ạ."*

### ❓ Câu 4: "Mô hình Baseline hiện tại đang yếu nhất ở điểm nào?"
* **Trả lời:**  
  *"Dạ thưa Thầy, theo Ma trận nhầm lẫn của nhóm, Baseline yếu nhất ở **lớp Trung tính (NEU)** và khả năng bắt các câu **mỉa mai/phủ định**. Vì lớp NEU chỉ chiếm 15% dữ liệu và câu từ thường pha trộn cả khen lẫn chê (ví dụ: 'Áo mặc tạm được nhưng vải hơi mỏng'), Baseline không đủ độ sâu để phân biệt ranh giới mong manh này ạ."*

---

## PHẦN 4: MA TRẬN PHÂN CÔNG THÀNH VIÊN TUẦN 3

| Thành viên | Trách nhiệm chính | Sản phẩm đầu ra cần nộp |
| :--- | :--- | :--- |
| **Nguyễn Duy Thái Bảo** | Điều phối WBS, viết code chia tập Stratified 70/15/15, tổng hợp Slide Tuần 3 | File dữ liệu chia tập + Slide PPTX Tuần 3 |
| **Đặng Đức Minh** | Xây dựng module tiền xử lý văn bản và từ điển Teencode | `src/preprocess.py` |
| **Bùi Đình Hiếu** | Lập trình và chạy thực nghiệm Baseline TF-IDF + Logistic Regression | `notebooks/02_baseline_models.ipynb` |
| **Từ Văn Huy Hoàng** | Viết module đánh giá metrics (Accuracy, Macro F1, Confusion Matrix) | `src/evaluate.py` + Ảnh biểu đồ |
| **Châu Tuấn Kiệt** | Thiết kế kiến trúc khung mạng Bi-LSTM trên PyTorch chuẩn bị cho Tuần 4 | `src/models.py` |
