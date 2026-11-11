# Web demo Tuần 4 — Đặng Đức Minh

Ứng dụng Streamlit đã tích hợp ba mô hình thực tế: Baseline LinearSVC, BiLSTM và PhoBERT v2. Các checkpoint người dùng cung cấp được nạp nghiêm ngặt; không dùng mô hình khởi tạo ngẫu nhiên khi thiếu file.

## Chức năng

- Chọn Baseline / PhoBERT / BiLSTM để phân tích một bình luận.
- Xem văn bản trước/sau xử lý, gạch chân cụm được chuẩn hóa và xem bảng biến đổi của Hiếu.
- Thử 15 câu của Kiệt; phân biệt nhãn tham chiếu với nhãn dự đoán thực tế.
- So sánh đồng thời ba mô hình. Thông báo sửa đúng lỗi Baseline chỉ xuất hiện khi dự đoán thật khớp nhãn tham chiếu.
- Chọn mô hình để phân tích CSV UTF-8 có cột `comment`, tối đa 5.000 dòng / 10 MB; xem phân phối và tải `download_predictions.csv`.
- Bảo toàn thứ tự, cột gốc và dòng rỗng. Dòng không còn chữ/số sau xử lý có trạng thái `skipped_empty`, không được gán nhãn giả.
- Mô hình được lưu đệm theo đường dẫn, kích thước và thời điểm thay đổi checkpoint; kiểm tra phiên bản trước khi nạp. Có nút chạy lại so sánh.

## Chạy trên máy

Máy Minh đã được chuẩn bị `.venv` và hai checkpoint. Mở `run_demo.cmd`, rồi truy cập http://127.0.0.1:8501. Đây là ứng dụng chạy cục bộ, chưa được đưa lên máy chủ công khai.

Thiết lập một máy khác bằng Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo-ai.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Để sử dụng GPU CUDA trên Windows với phiên bản đã kiểm tra ở máy Minh, cài PyTorch tương ứng trước khi cài requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-demo-ai.txt
```

Chỉ chạy Baseline: dùng `requirements-demo.txt`. Chạy thêm hai mô hình neural: dùng `requirements-demo-ai.txt`. GPU được ưu tiên nếu PyTorch hỗ trợ CUDA; nếu không, chương trình dùng CPU. Nạp checkpoint lần đầu mất lâu hơn các lượt dự đoán sau.

## Các tệp mô hình

- `models/baseline_tfidf_vectorizer.pkl`, `models/baseline_linear_svc.pkl`: Baseline đóng băng của nhóm.
- `models/best_bilstm.pt`: checkpoint BiLSTM; dùng cùng `models/vocab.json` có 3.348 token.
- `models/best_phobert.pt`: checkpoint PhoBERT v2.
- `models/phobert_assets/`: cấu hình và tokenizer chuẩn PhoBERT đã lưu cục bộ. Không cần tải backbone tiền huấn luyện khi mở web.
- `models/model_manifest.json`: SHA-256, dung lượng, epoch và chỉ số đi kèm hai file gốc và bản dùng trên web.

Hai file gốc trong `D:/Downloads` được giữ nguyên. Bản dùng trên web bỏ trạng thái optimizer, giữ nguyên mọi tensor trọng số; đã kiểm tra toàn bộ tensor giống hệt nguồn. Dung lượng bản suy luận khoảng 4,36 MB (BiLSTM) và 540,08 MB (PhoBERT), thay vì 13,09 MB và 1,62 GB của file gốc.

File `.pt` đang được `.gitignore` bỏ qua, nên không tự đi theo Git hoặc PR. Khi chuyển máy phải bàn giao checkpoint riêng, cùng vocabulary/config/tokenizer tương ứng. Các file `.pkl` gốc không bị sửa hay train lại.

Khi nhận checkpoint mới, đặt vào đúng đường dẫn trong `models/` và chạy lại so sánh. Bộ nạp kiểm tra đủ khóa trọng số, cấu hình và kích thước vocabulary; không tạo vocabulary rỗng hoặc mô hình ngẫu nhiên để thay thế.

## Lưu ý về chất lượng checkpoint đã nhận

| Mô hình | Epoch lưu trong file | Macro F1 validation lưu trong file |
|---|---:|---:|
| BiLSTM | 4 | 66,39% |
| PhoBERT | 1 | 12,85% |

Đây là metadata đi kèm checkpoint, không phải kết quả benchmark trên test độc lập do web đo. Thử ba câu tích cực/tiêu cực/trung tính: BiLSTM trả POS/NEG/NEU; PhoBERT trả NEU cho cả ba. Không thể suy ra chất lượng toàn bộ tập test từ ba câu này, nhưng cần kiểm tra lại checkpoint PhoBERT trước khi dùng để khẳng định chất lượng mô hình.

Giao diện hiển thị kết quả thật và chỉ số validation đi kèm file. Các câu demo có nhãn tham chiếu được hiển thị riêng; không lấy `phobert_expected` trong JSON làm kết quả dự đoán.

## Ý nghĩa điểm số

- LinearSVC: softmax của `decision_function` chỉ minh họa mức ủng hộ tương đối giữa ba lớp; không phải xác suất đã hiệu chỉnh.
- BiLSTM / PhoBERT: phân phối softmax từ logits thực tế của mô hình đã nạp trọng số. Tổng ba xác suất bằng 1; con số này không thay thế phép đánh giá trên dữ liệu có nhãn.
- CSV xuất thêm tên mô hình và loại điểm số, giữ các cột gốc; nếu trùng tên cột kết quả, chương trình tự thêm tiền tố `ai_`.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests/test_app_integration.py tests/test_demo_models.py tests/test_preprocess.py -q
```

Đã kiểm tra với Python 3.12, PyTorch 2.7.1+cu128, Transformers 4.57.6, Streamlit 1.67.0, Plotly 7.1.0, scikit-learn 1.7.2, pandas 3.1.1, PyVi 0.1.1; GPU RTX 3050 Laptop. Không có xung đột theo `pip check`.

Kiểm thử bao gồm nạp trọng số thật, đổi checkpoint không nhầm cache, từ chối file thiếu/thay đổi, đổi mô hình trên UI, CSV với mô hình neural, dữ liệu rỗng và bảo toàn các cột. Các bài cần checkpoint sẽ được bỏ qua nếu máy mới chưa có file `.pt` riêng.

Ngày 12/11/2026: toàn bộ 32/32 bài kiểm thử của ba file phía trên đạt, không có bài bị bỏ qua trên máy đã nhận hai checkpoint.

### Chạy thật trên 15 câu demo

| Mô hình | Số nhãn khớp nhãn tham chiếu |
|---|---:|
| Baseline | 3/15 |
| BiLSTM | 8/15 |
| PhoBERT | 2/15 |

PhoBERT trả NEU cho cả 15 câu. Hai câu có nhãn tham chiếu NEU nên được tính là khớp. Đây là bộ câu demo được lựa chọn có chủ đích, không phải tập test độc lập hoặc phép đo chất lượng đại diện toàn bộ dữ liệu. Kết quả từng câu nằm trong `docs/WEB_DEMO_15_CASE_RESULTS.csv`.

Tài liệu AppTest chính thức: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest

## Phạm vi thay đổi

- Giao diện và luồng web: `app.py`; bộ nạp checkpoint riêng của web: `demo_models.py`.
- CSS: `static/style.css`; cấu hình Streamlit: `.streamlit/config.toml`.
- Không sửa `src/preprocess.py`, `src/models.py`, `src/train.py`, `src/inference.py`, `src/evaluate.py`, dữ liệu gốc hoặc các mô hình Baseline của thành viên khác.
