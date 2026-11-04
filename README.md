# Vietnamese Review Sentiment Analysis (Shopee Reviews)
> **AI Capstone Project 2025 • Viện Nghiên cứu & Đào tạo Việt - Anh (VNUK) • Đại học Đà Nẵng**  
> **Giảng viên hướng dẫn:** TS. Trần Thế Vũ (`vu.tran@vnuk.edu.vn`)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-PhoBERT-yellow.svg)](https://huggingface.co/vinai/phobert-base)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()

---

## 📌 1. Giới thiệu Đề tài (Project Overview)
Trong thời đại thương mại điện tử bùng nổ, các sàn TMĐT (Shopee, Tiki) tiếp nhận hàng trăm nghìn lượt đánh giá phản hồi mỗi ngày. Dự án xây dựng một hệ thống **Deep Learning đầu-cuối (End-to-End)** có khả năng tự động xử lý và phân loại chính xác sắc thái cảm xúc từ các câu đánh giá tiếng Việt thô thành **03 nhãn**:
* **`POS` (Positive - Tích cực):** Đánh giá hài lòng, khen ngợi.
* **`NEU` (Neutral - Trung tính):** Nhận xét bình thường, ý kiến hỗn hợp.
* **`NEG` (Negative - Tiêu cực):** Khiếu nại, bức xúc, chất lượng kém.

---

## 🔬 2. Ba Câu hỏi Nghiên cứu (Research Questions - RQs)
1. **RQ1 (Data Preprocessing & Teencode):** Việc xây dựng pipeline tiền xử lý chuẩn hóa teencode tiếng Việt và tách từ (Word Segmentation) cải thiện hiệu năng mô hình như thế nào so với văn bản thô?
2. **RQ2 (Model Architecture):** Mô hình học sâu (Bi-LSTM) và Transformer (VinAI PhoBERT) vượt trội hơn mô hình Baseline truyền thống (TF-IDF + Logistic Regression/SVM) ở những khía cạnh ngữ cảnh nào?
3. **RQ3 (Class Imbalance):** Việc áp dụng kỹ thuật giải quyết mất cân bằng nhãn (Weighted Cross-Entropy Loss / Resampling) cải thiện chỉ số Macro F1-score ở các lớp thiểu số (Negative & Neutral) ra sao?

---

## 📂 3. Cấu trúc Thư mục Dự án (Project Structure)
```text
Project_Intro_AI/
├── .gitignore                      # Quy tắc chặn đẩy file rác/trọng số lớn
├── README.md                       # Tài liệu hướng dẫn dự án
├── requirements.txt                # Thư viện phụ thuộc
├── data/
│   ├── raw/                        # Dữ liệu gốc thu thập từ Shopee (31.460 dòng)
│   │   └── shopee_reviews.csv
│   └── processed/                  # Dữ liệu đã làm sạch & phân chia
│       └── shopee_reviews_cleaned.csv  (26.458 dòng sạch)
├── docs/                           # Tài liệu, báo cáo & biểu đồ
│   ├── references.md               # Trích dẫn bài báo khoa học chuẩn APA/BibTeX
│   ├── class_distribution.png      # Biểu đồ phân bố 3 nhãn cảm xúc
│   ├── sequence_length.png         # Biểu đồ phân tích độ dài câu
│   └── KICH_BAN_TUAN_3_AI_CAPSTONE.docx # Kịch bản báo cáo & checklist Tuần 3
├── notebooks/                      # Sổ tay thực nghiệm Jupyter
│   ├── 01_eda_and_cleaning.ipynb   # Khám phá EDA, làm sạch và thẩm định dữ liệu
│   ├── 02_baseline_models.ipynb    # Huấn luyện mô hình cơ sở TF-IDF + Machine Learning
│   └── 03_deep_learning_models.ipynb # Huấn luyện Bi-LSTM và PhoBERT
└── src/                            # Mã nguồn đóng gói chuẩn
    ├── preprocess.py               # Tiền xử lý teencode, tách từ, chia tập dữ liệu
    ├── dataset.py                  # PyTorch Dataset & DataLoader
    ├── models.py                   # Kiến trúc mô hình (Bi-LSTM, PhoBERT Classifier)
    └── evaluate.py                 # Hàm tính Macro F1, Accuracy & Confusion Matrix
```

---

## 📊 4. Thẩm định Dữ liệu (Data Quality Review)
* **Tổng dòng dữ liệu gốc:** 31.460 dòng (3 cột: `content`, `label`, `start`).
* **Làm sạch:** Xóa bỏ 24 dòng rỗng (`dropna`) và 4.920 dòng spam trùng lặp (`drop_duplicates`).
* **Dữ liệu sạch thu được:** **26.458 dòng**.
* **Phân bố nhãn (Class Distribution):**
  * `POS`: 63.8% (16.920 dòng) - Lớp áp đảo
  * `NEG`: 21.2% (5.618 dòng)
  * `NEU`: 15.0% (3.920 dòng)
* **Độ dài câu:** 95% số câu ngắn dưới 27 từ $\rightarrow$ Thiết lập `max_length = 64` giúp tiết kiệm GPU và tăng tốc huấn luyện.

---

## 🚀 5. Hướng dẫn Cài đặt & Chạy Thực nghiệm (Quickstart)

### Cài đặt môi trường
```bash
# Clone repository về máy
git clone https://github.com/nguyenduythaibao1611-eng/Project_Intro_AI.git
cd Project_Intro_AI

# Khởi tạo môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate      # Windows

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### Chạy Thực nghiệm Notebooks
```bash
# Khởi động Jupyter Notebook
jupyter notebook

# Mở notebooks/01_eda_and_cleaning.ipynb để xem phân tích dữ liệu
# Mở notebooks/02_baseline_models.ipynb để chạy mô hình Baseline
```

---

## 👥 6. Thành viên Nhóm & Phân công (Team Members)
| Thành viên | MSSV | Vai trò chính |
| :--- | :--- | :--- |
| **Nguyễn Duy Thái Bảo** | 24020015 | **Team Leader / Project Manager & DevOps** |
| **Châu Tuấn Kiệt** | 240020010 | **Senior ML Engineer (Deep Learning & PhoBERT)** |
| **Đặng Đức Minh** | 24020001 | **Data Engineer (Preprocessing & Teencode Pipeline)** |
| **Bùi Đình Hiếu** | 24020006 | **AI/ML Engineer (Baseline TF-IDF & Experiments)** |
| **Từ Văn Huy Hoàng** | 2402009 | **Training & Evaluation Engineer (Metrics & Colab GPU)** |

---
*© 2025 VNUK Institute for Research and Executive Education, Danang University.*
