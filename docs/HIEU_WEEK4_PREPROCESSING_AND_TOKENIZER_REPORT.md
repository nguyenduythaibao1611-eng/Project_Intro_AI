# 📋 BÁO CÁO KỸ THUẬT TUẦN 4: TỐI ƯU HÓA ĐƯỜNG ỐNG TIỀN XỬ LÝ & MODULE TOKENIZER PHOBERT

> **Dự án:** Phân tích Cảm xúc Đánh giá Khách hàng Shopee (Vietnamese E-commerce Sentiment Analysis)  
> **Học phần:** Artificial Intelligence Capstone 2025 — Viện VNUK, Đại học Đà Nẵng  
> **Giảng viên hướng dẫn:** TS. Trần Thế Vũ (`vu.tran@vnuk.edu.vn`)  
> **Kỹ sư phụ trách:** **Bùi Đình Hiếu (Data Engineer / Preprocessing Specialist)**  
> **Nhánh tính năng:** `feature/w4-preprocessing-optimization-hieu`  
> **Nhánh đích hợp nhất (Target Branch):** `develop`  
> **Trạng thái:** Hoàn thành 100% tiêu chí nghiệm thu (Definition of Done)

---

## 1. TỔNG QUAN BỐI CẢNH & MỤC TIÊU TUẦN 4

Bước sang Tuần 4, hệ thống chuyển dịch sang mô hình học sâu **PhoBERT Transformer** và xây dựng ứng dụng **Streamlit Web Live Demo** phục vụ buổi bảo vệ đồ án cuối kỳ. Vai trò của Kỹ sư Tiền xử lý dữ liệu đòi hỏi giải quyết 3 bài toán trọng tâm:

1. **Tối ưu hóa độ trễ thời gian thực (Ultra Low-latency Inference):** Đảm bảo thời gian làm sạch và chuẩn hóa văn bản đạt trung bình dưới **2 mili-giây/câu** khi phục vụ người dùng gõ trực tiếp trên Web Demo.
2. **Cung cấp tính năng bóc tách biến đổi Teencode (Visual Diff Inspection):** Hỗ trợ ứng dụng Web của Đặng Đức Minh hiển thị bảng giải thích trực quan cho giảng viên và hội đồng thấy thuật toán đã chuẩn hóa teencode và loại bỏ nhiễu như thế nào.
3. **Module hóa Tokenizer tương thích PhoBERT (`src/tokenizer_utils.py`):** Cung cấp API chuẩn hóa xuất tensor `input_ids` và `attention_mask` độ dài 64 tokens cho pipeline huấn luyện của Từ Văn Huy Hoàng và thẩm định của Châu Tuấn Kiệt.

---

## 2. CÁC TỐI ƯU HÓA KỸ THUẬT ĐÃ TRIỂN KHAI

### A. Tối ưu hóa Tốc độ Tiền xử lý trong `src/preprocess.py`
- **Biên dịch sẵn toàn bộ biểu thức chính quy (Pre-compiled Regex):**
  - `_URL`, `_EMAIL`, `_PHONE`, `_WORD`, `_TOKENS`, `_HTML_TAGS`, `_CONTROL_CHARS`, `_WHITESPACE`.
  - Triệt tiêu hoàn toàn chi phí biên dịch lại biểu thức chính quy trong từng lần gọi hàm.
- **Bộ nhớ đệm LRU Cache cho ký tự lặp kéo dài (`_collapse_word_elongation`):**
  - Tích hợp `@lru_cache(maxsize=16384)` giúp các từ cảm thán lặp lại nhiều lần (`đẹpppp`, `quáaaaa`, `tệeeee`) được xử lý tức thời trong $O(1)$.
- **Tra cứu Teencode theo bảng băm $O(1)$:**
  - Mở rộng từ điển hơn 20 cụm từ lóng TMĐT Shopee mới (`hỏa tốc`, `chuẩn bị hàng`, `đáng tiền`, `xịn xò`, `không thích`, `lừa đảo`...).
  - Tìm kiếm và thay thế theo nguyên tắc tiền tố dài nhất (Longest-match preference), tránh nhầm lẫn giữa đơn vị kích thước/tiền tệ (`100k`, `size_k`) với từ phủ định (`k` $\rightarrow$ `không`).

### B. Hàm Bóc tách Biến đổi `clean_text_with_diff`
Cung cấp chữ ký hàm chuẩn hóa:
```python
def clean_text_with_diff(
    raw_text: str,
    apply_segmentation: bool = True
) -> Tuple[str, List[Dict[str, str]]]:
```
- **Đầu ra:**
  - `clean_text`: Chuỗi đã chuẩn hóa hoàn toàn (làm sạch cơ bản, mở rộng teencode, và tách từ ghép tiếng Việt).
  - `changes`: Danh sách chi tiết các can thiệp gồm 3 trường:
    - `"original"`: Từ ngữ/ký tự gốc trong bình luận thô.
    - `"normalized"`: Từ ngữ sau khi chuẩn hóa.
    - `"type"`: Phân loại can thiệp (`'teencode'`, `'elongation'`, `'html'`, `'url'`, `'email'`, `'phone'`).
- **An toàn tuyệt đối (Fault-tolerant):** Xử lý trơn tru các trường hợp biên: chuỗi rỗng `""`, giá trị `None`, chuỗi chỉ chứa icon/emoji, ký tự điều khiển UTF-8, số điện thoại dài, và teencode lồng nhau.

### C. Module `src/tokenizer_utils.py` cho PhoBERT v2
- Tích hợp `AutoTokenizer.from_pretrained("vinai/phobert-base-v2")`.
- **Cơ chế Singleton Cache:** Hàm `get_phobert_tokenizer()` nạp trọng số một lần duy nhất vào bộ nhớ, tái sử dụng xuyên suốt phiên làm việc.
- **Hàm mã hóa tensor chuẩn hóa:**
  - `encode_reviews(texts, max_len=64)`: Trả về PyTorch tensor `input_ids` và `attention_mask` chuẩn kích thước `[batch_size, 64]`, kiểu `torch.long`.
  - `batch_encode(texts, batch_size=32)`: Bộ tạo (generator) phục vụ phân tích hàng loạt tệp CSV trên Web Demo.
  - `decode_tokens(input_ids)`: Giải mã tensor ngược lại chuỗi từ để minh họa cơ chế tách từ Byte-Pair Encoding (BPE).

---

## 3. KẾT QUẢ ĐO LƯỜNG THỰC NGHIỆM & BENCHMARK

### A. Đo lường Độ trễ Xử lý (Latency Benchmark)
Thử nghiệm trên 100 câu đánh giá thực tế đa dạng của người dùng Shopee:
- **Thời gian xử lý trung bình:** **0.728 ms / câu**.
- **Chỉ tiêu kỹ thuật cam kết (DoD KPI):** **< 2.0 ms / câu**.
- **Kết luận:** Tốc độ thực tế **nhanh hơn 2.7 lần** so với ngưỡng yêu cầu, đảm bảo giao diện Streamlit phản hồi mượt mà không gây giật lag.

### B. Kết quả Kiểm thử Toàn diện (Unit Tests)
- **Tổng số bài kiểm thử:** **37 / 37 test cases** toàn bộ repository đạt trạng thái **PASS 100%**.
  - `tests/test_preprocess.py`: 13 tests (Làm sạch, Teencode, Tách từ, Corner cases, Phân chia phân tầng, Latency < 2ms).
  - `tests/test_tokenizer_utils.py`: 7 tests (Singleton, Single/Batch encoding, Truncation/Padding, BPE Decoding).
  - `tests/test_baseline.py`: 6 tests (Mốc chuẩn Baseline Tuần 3).
  - `tests/test_dataset.py` & `tests/test_models.py`: 11 tests (Mô hình Bi-LSTM).

---

## 4. HƯỚNG DẪN TÍCH HỢP CHO CÁC THÀNH VIÊN TRONG NHÓM

### Dành cho Đặng Đức Minh (`app.py` - Web Demo):
```python
from src.preprocess import clean_text_with_diff

# Gọi hàm khi người dùng nhấn nút "Phân tích cảm xúc"
clean_text, changes = clean_text_with_diff(user_input, apply_segmentation=True)

# Hiển thị bảng can thiệp teencode trên giao diện Streamlit
if changes:
    st.dataframe(pd.DataFrame(changes))
```

### Dành cho Từ Văn Huy Hoàng (`src/train.py` & `src/inference.py`):
```python
from src.tokenizer_utils import encode_reviews

# Tokenize lô dữ liệu trước khi đưa vào mô hình PhoBERT
inputs = encode_reviews(batch_texts, max_len=64)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
outputs = phobert_model(input_ids=input_ids, attention_mask=attention_mask)
```
