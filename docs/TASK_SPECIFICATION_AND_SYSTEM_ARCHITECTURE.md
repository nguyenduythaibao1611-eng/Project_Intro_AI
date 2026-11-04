# 🛠️ TÀI LIỆU ĐẶC TẢ KỸ THUẬT & PHÂN RÃ NHIỆM VỤ HỆ THỐNG
## (SYSTEM ARCHITECTURE SPECIFICATION & MEMBER TASK BREAKDOWN)

> **Dự án:** Hệ thống Phân tích Cảm xúc Đánh giá Khách hàng Tiếng Việt (Vietnamese Review Sentiment Analysis)  
> **Khung chương trình:** Artificial Intelligence Capstone 2025 • Viện Nghiên cứu & Đào tạo Việt - Anh (VNUK)  
> **Mục đích tài liệu:** Đặc tả chi tiết ở cấp độ kỹ thuật (Level 3 Engineering Specification) dành cho kỹ sư và các hệ thống AI Agents. Tài liệu quy định chuẩn hóa cấu trúc dữ liệu, tên biến, chữ ký hàm (function signatures), siêu tham số (hyperparameters), và luồng tương tác giữa các module.

---

## MỤC LỤC
1. [QUY ƯỚC TOÀN CỤC & HẰNG SỐ HỆ THỐNG (GLOBAL CONSTANTS)](#1-quy-ước-toàn-cục--hằng-số-hệ-thống-global-constants)
2. [ĐẶC TẢ CHI TIẾT TỪNG THÀNH VIÊN](#2-đặc-tả-chi-tiết-từng-thành-viên)
   * [2.1. Đặng Đức Minh — Data Engineer (`src/preprocess.py`)](#21-đặng-đức-minh--data-engineer-srcpreprocesspy)
   * [2.2. Bùi Đình Hiếu — AI/ML Engineer (`notebooks/02_baseline_models.ipynb`)](#22-bùi-đình-hiếu--aiml-engineer-notebooks02_baseline_modelsipynb)
   * [2.3. Từ Văn Huy Hoàng — Evaluation Engineer (`src/evaluate.py`)](#23-từ-văn-huy-hoàng--evaluation-engineer-srcevaluatepy)
   * [2.4. Châu Tuấn Kiệt — Senior ML Engineer (`src/dataset.py` & `src/models.py`)](#24-châu-tuấn-kiệt--senior-ml-engineer-srcdatasetpy--srcmodelspy)
   * [2.5. Nguyễn Duy Thái Bảo — Project Manager & DevOps (Pipeline Integration & Git Flow)](#25-nguyễn-duy-thái-bảo--project-manager--devops)
3. [SƠ ĐỒ LUỒNG DỮ LIỆU ĐẦU-CUỐI (END-TO-END DATA FLOW)](#3-sơ-đồ-luồng-dữ-liệu-đầu-cuối-end-to-end-data-flow)

---

## 1. QUY ƯỚC TOÀN CỤC & HẰNG SỐ HỆ THỐNG (GLOBAL CONSTANTS)

Mọi module và script trong dự án bắt buộc phải tuân thủ nghiêm ngặt các quy chuẩn định danh sau:

```python
# Cấu hình hạt giống ngẫu nhiên (Reproducibility Seed)
RANDOM_SEED = 42

# Mã hóa nhãn chuẩn (Label Mapping) - 3 Phân lớp
LABEL_TO_ID = {
    "NEG": 0,   # Negative (Tiêu cực - 1-2 sao)
    "NEU": 1,   # Neutral (Trung tính - 3 sao)
    "POS": 2    # Positive (Tích cực - 4-5 sao)
}
ID_TO_LABEL = {0: "NEG", 1: "NEU", 2: "POS"}

# Tham số độ dài chuỗi tối đa (Max Sequence Length)
MAX_SEQUENCE_LENGTH = 64

# Tỷ lệ phân chia tập dữ liệu (Stratified Data Split Ratio)
SPLIT_RATIOS = {
    "TRAIN": 0.70,  # 70% (~18.520 mẫu)
    "VAL":   0.15,  # 15% (~3.969 mẫu)
    "TEST":  0.15   # 15% (~3.969 mẫu)
}

# Đường dẫn thư mục tiêu chuẩn (Standard Paths)
PATH_RAW_DATA = "data/raw/shopee_reviews.csv"
PATH_CLEANED_DATA = "data/processed/shopee_reviews_cleaned.csv"
PATH_TRAIN_CSV = "data/processed/train.csv"
PATH_VAL_CSV = "data/processed/val.csv"
PATH_TEST_CSV = "data/processed/test.csv"
```

---

## 2. ĐẶC TẢ CHI TIẾT TỪNG THÀNH VIÊN

---

### 2.1. ĐẶNG ĐỨC MINH — Data Engineer
* **Tệp mã nguồn phụ trách:** `src/preprocess.py`
* **Nhiệm vụ cốt lõi:** Xây dựng pipeline tiền xử lý ngôn ngữ tự nhiên tiếng Việt, chuẩn hóa teencode TMĐT, tách từ ghép, và thực hiện phân chia tập dữ liệu phân tầng (Stratified Splitting).

#### Danh sách biến & Cấu trúc dữ liệu:
* `TEENCODE_DICT: Dict[str, str]`: Bảng tra cứu từ điển gồm hơn 100 từ lóng và viết tắt trên sàn Shopee.
* `STOPWORDS_SET: Set[str]`: Tập hợp các từ dừng không mang giá trị ngữ nghĩa cảm xúc (loại trừ các từ phủ định như *không, chẳng, chưa*).

#### Chữ ký Hàm (Function Signatures & Implementation Specs):

```python
def clean_basic_text(text: str) -> str:
    """
    1. Chuyển đổi toàn bộ chuỗi về chữ thường (lowercase).
    2. Sử dụng Regex xóa bỏ URL (http\S+|www\S+), email, số điện thoại.
    3. Chuẩn hóa các ký tự lặp vô nghĩa (ví dụ: 'đẹpppppp' -> 'đẹp', 'quáaaaa' -> 'quá').
    4. Xóa khoảng trắng thừa đầu cuối và khoảng trắng kép.
    """
    pass

def normalize_teencode(text: str, teencode_dict: Dict[str, str]) -> str:
    """
    Thay thế các từ viết tắt theo bảng ánh xạ:
    - 'k', 'ko', 'khong' -> 'không'
    - 'sp' -> 'sản phẩm'
    - 'đc', 'dk', 'dc' -> 'được'
    - 'shop' -> 'cửa hàng'
    - 'ship' -> 'giao hàng'
    - 'auth', 'aut' -> 'chính hãng'
    - 'rep' -> 'trả lời'
    - 'fb' -> 'phản hồi'
    """
    pass

def segment_vietnamese(text: str) -> str:
    """
    Sử dụng thư viện PyVi (ViTokenizer.tokenize) hoặc RDRSegmenter:
    Nối các từ ghép có nghĩa bằng dấu gạch dưới (_).
    Ví dụ: 'sản phẩm rất tốt giao hàng nhanh' -> 'sản_phẩm rất tốt giao_hàng nhanh'
    """
    pass

def preprocess_pipeline(text: str) -> str:
    """
    Hàm tổng hợp chạy tuần tự: clean_basic_text -> normalize_teencode -> segment_vietnamese.
    """
    pass

def split_and_save_stratified(
    input_csv_path: str,
    output_dir: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1. Đọc dữ liệu từ input_csv_path (26.458 dòng).
    2. Áp dụng preprocess_pipeline lên cột 'content' tạo cột 'processed_content'.
    3. Sử dụng sklearn.model_selection.train_test_split với stratify=df['label'].
    4. Lưu 3 tệp tin CSV:
       - output_dir/train.csv (cột: content, processed_content, label, label_id, start)
       - output_dir/val.csv
       - output_dir/test.csv
    """
    pass
```

---

### 2.2. BÙI ĐÌNH HIẾU — AI/ML Engineer
* **Tệp mã nguồn phụ trách:** `notebooks/02_baseline_models.ipynb`
* **Nhiệm vụ cốt lõi:** Thiết lập hệ thống thực nghiệm mô hình học máy truyền thống làm mốc chuẩn (Benchmark Baseline) để định lượng hóa câu hỏi nghiên cứu RQ1 và RQ2.

#### Danh sách biến & Siêu tham số (Hyperparameters):
* `vectorizer: TfidfVectorizer`:
  * `ngram_range = (1, 2)` (Unigram & Bigram)
  * `max_features = 5000` (Giữ lại 5.000 từ vựng có trọng số TF-IDF cao nhất)
  * `min_df = 2` (Loại bỏ từ hiếm chỉ xuất hiện 1 lần)
  * `sublinear_tf = True` (Áp dụng thang logarit $1 + \log(TF)$)
* `clf_logistic: LogisticRegression`:
  * `C = 1.0`, `solver = 'lbfgs'`, `max_iter = 1000`, `class_weight = 'balanced'`
* `clf_svm: LinearSVC`:
  * `C = 0.5`, `class_weight = 'balanced'`, `random_state = 42`

#### Quy trình thực thi trong Notebook:
1. **Nạp dữ liệu:** Đọc `train.csv` và `test.csv` từ `data/processed/`.
2. **Khởi tạo ma trận đặc trưng:**
   ```python
   X_train = vectorizer.fit_transform(df_train['processed_content'])
   y_train = df_train['label_id'].values
   X_test = vectorizer.transform(df_test['processed_content'])
   y_test = df_test['label_id'].values
   ```
3. **Thí nghiệm bóc tách (Ablation Study cho RQ1):**
   * Huấn luyện mô hình trên `content` thô $\rightarrow$ Ghi nhận Macro F1.
   * Huấn luyện mô hình trên `processed_content` (đã tiền xử lý teencode) $\rightarrow$ Ghi nhận Macro F1 chênh lệch.
4. **Lưu trữ Model Artifacts:**
   * Xuất `models/baseline_tfidf_vectorizer.pkl` và `models/baseline_logistic_regression.pkl`.

---

### 2.3. TỪ VĂN HUY HOÀNG — Training & Evaluation Engineer
* **Tệp mã nguồn phụ trách:** `src/evaluate.py`
* **Nhiệm vụ cốt lõi:** Chuẩn hóa các thước đo hiệu năng, xây dựng hàm tạo báo cáo phân loại và vẽ đồ thị ma trận nhầm lẫn (Confusion Matrix) dùng chung cho cả Baseline, Bi-LSTM và PhoBERT.

#### Chữ ký Hàm (Function Signatures & Implementation Specs):

```python
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = ["NEG", "NEU", "POS"]
) -> Dict[str, Union[float, Dict]]:
    """
    Tính toán và trả về Dictionary gồm:
    - 'accuracy': float
    - 'macro_precision': float (average='macro')
    - 'macro_recall': float (average='macro')
    - 'macro_f1': float (average='macro')
    - 'weighted_f1': float (average='weighted')
    - 'per_class': Dict chứa F1-score riêng của từng nhãn (NEG, NEU, POS)
    """
    pass

def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["NEG", "NEU", "POS"],
    save_path: str = "docs/confusion_matrix.png",
    title: str = "Confusion Matrix"
) -> None:
    """
    1. Tính confusion_matrix(y_true, y_pred, normalize='true') (tỷ lệ phần trăm).
    2. Sử dụng Seaborn heatmap (cmap='Blues', annot=True, fmt='.2%').
    3. Thêm trục X: Predicted Label, trục Y: True Label.
    4. Xuất ảnh với độ phân giải cao (dpi=300) vào đường dẫn save_path.
    """
    pass

def analyze_error_samples(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id_to_label: Dict[int, str],
    top_k: int = 50
) -> pd.DataFrame:
    """
    Phục vụ Error Analysis (Tuần 4):
    Trích xuất danh sách top_k mẫu câu mô hình đoán sai nghiêm trọng nhất
    (ví dụ: Thực tế là NEG nhưng mô hình đoán thành POS do dính từ khen mỉa mai).
    Trả về DataFrame gồm: ['text', 'true_label', 'pred_label'].
    """
    pass
```

---

### 2.4. CHÂU TUẤN KIỆT — Senior ML Engineer
* **Tệp mã nguồn phụ trách:** `src/dataset.py`, `src/models.py`, và `notebooks/03_deep_learning_models.ipynb`
* **Nhiệm vụ cốt lõi:** Thiết kế cấu trúc dữ liệu PyTorch Dataset, hiện thực hóa kiến trúc mạng nơ-ron hồi quy hai chiều Bi-LSTM và khung tinh chỉnh mô hình Transformer PhoBERT.

#### A. Tệp `src/dataset.py` (PyTorch Dataset & Vocab Builder):
```python
class Vocabulary:
    """Xây dựng từ điển ánh xạ token -> ID cho mô hình Bi-LSTM."""
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.itos = {0: "<PAD>", 1: "<UNK>"}
    
    def build_vocab(self, sentence_list: List[str]) -> None:
        pass

class SentimentDataset(torch.utils.data.Dataset):
    """
    Dataset cho PyTorch Bi-LSTM:
    Nhận vào văn bản đã tách từ, đệm (padding) hoặc cắt ngắn (truncating) về MAX_SEQUENCE_LENGTH = 64.
    Trả về Dict: {'input_ids': torch.Tensor, 'label': torch.Tensor, 'length': int}
    """
    pass
```

#### B. Tệp `src/models.py` (Kiến trúc Mạng Bi-LSTM):
```python
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout_rate: float = 0.3
    ):
        super(BiLSTMClassifier, self).__init__()
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Bidirectional LSTM Layer (2 chiều đọc song song)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 3. Dropout & Fully Connected Output Layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 vì là 2 chiều Forward + Backward

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Nhúng vector: [batch_size, seq_len, embed_dim]
        embedded = self.dropout(self.embedding(input_ids))
        
        # Đóng gói chuỗi (Pack padded sequence) để tối ưu tính toán
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Lấy hidden state cuối cùng của cả 2 chiều: forward (layer cuối) và backward (layer cuối)
        # hidden shape: [num_layers * 2, batch_size, hidden_dim]
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1) # [batch_size, hidden_dim * 2]
        
        logits = self.fc(self.dropout(hidden_concat))
        return logits
```

#### C. Thiết kế Trọng số Hàm mất mát (Weighted Cross-Entropy Loss cho RQ3):
Để xử lý tình trạng lệch nhãn (POS 63.8%, NEG 21.2%, NEU 15.0%), công thức tính trọng số phạt nghịch đảo tỷ lệ:
$$w_c = \frac{N}{K \cdot N_c}$$
Trong đó $N = 18.520$ (số mẫu train), $K = 3$ (số lớp), $N_c$ là số mẫu của lớp $c$.
```python
# Cài đặt trong PyTorch
class_weights = torch.tensor([1.57, 2.22, 0.52], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

### 2.5. NGUYỄN DUY THÁI BẢO — Project Manager & DevOps
* **Nhiệm vụ cốt lõi:**
  1. **Quản trị Luồng Phân nhánh Git (Git Flow Management):**
     * Nhánh `main`: Chỉ chứa bản phát hành ổn định cuối mỗi sprint tuần.
     * Nhánh `develop`: Môi trường tích hợp chung của toàn nhóm.
     * Các nhánh tính năng riêng: `feature/preprocess`, `feature/baseline`, `feature/bilstm`.
  2. **Kiểm soát Tái lập Môi trường (Reproducibility & Dependencies):**
     * Đảm bảo mọi thành viên chạy thống nhất trên môi trường được định nghĩa trong `requirements.txt`.
  3. **Tích hợp Pipeline (Continuous Integration):**
     * Kiểm thử tính tương thích giữa đầu ra của `src/preprocess.py` và đầu vào của `src/dataset.py`.
  4. **Quản trị Báo cáo & Tài liệu Học thuật (Documentation Lead):**
     * Biên tập slide PowerPoint hàng tuần theo khung 5W1H.
     * Quản trị file `references.md` và biên soạn báo cáo tổng kết cuối kỳ (`docs/final_project_report.pdf`).

---

## 3. SƠ ĐỒ LUỒNG DỮ LIỆU ĐẦU-CUỐI (END-TO-END DATA FLOW)

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. DỮ LIỆU GỐC: data/raw/shopee_reviews.csv (31.460 dòng)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Đức Minh: 01_eda_and_cleaning.ipynb)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DỮ LIỆU SẠCH: shopee_reviews_cleaned.csv (26.458 dòng)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Đức Minh: src/preprocess.py)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. PIPELINE TIỀN XỬ LÝ & TÁCH TỪ TIẾNG VIỆT                │
│    Clean Basic -> Normalize Teencode -> Segment Words       │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Thái Bảo: Stratified Split 70/15/15)
                               ▼
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ train.csv (70%) │   │  val.csv (15%)  │   │  test.csv (15%) │
│   18.520 dòng   │   │   3.969 dòng    │   │   3.969 dòng    │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         ├─────────────────────┼─────────────────────┤
         ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. MÔ HÌNH HỌC MÁY & HỌC SÂU                                │
│                                                             │
│  [MÔ HÌNH 1: BASELINE (Đình Hiếu)]                          │
│   TF-IDF (1,2-gram) + Logistic Regression / LinearSVC       │
│                                                             │
│  [MÔ HÌNH 2: DEEP LEARNING (Tuấn Kiệt)]                     │
│   PyTorch Bi-LSTM (2 layers, Hidden 128, Dropout 0.3)       │
│                                                             │
│  [MÔ HÌNH 3: SOTA TRANSFORMER (Tuấn Kiệt & Hoàng)]          │
│   VinAI PhoBERT-base Fine-tuning                            │
│                                                             │
│  [HÀM MẤT MÁT TỐI ƯU (Cả nhóm)]                            │
│   Weighted Cross-Entropy Loss (Khắc phục lệch nhãn POS 64%) │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Huy Hoàng: src/evaluate.py)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ĐÁNH GIÁ CHỈ SỐ BENCHMARK & ERROR ANALYSIS               │
│    - Macro F1-score, Accuracy, Precision, Recall            │
│    - Confusion Matrix (docs/confusion_matrix.png)           │
│    - Trích xuất 50 mẫu lỗi (docs/error_analysis.csv)        │
└─────────────────────────────────────────────────────────────┘
```

---
*Tài liệu này là căn cứ kỹ thuật chính thức của nhóm Capstone 2025. Mọi thay đổi về cấu trúc biến hoặc hàm phải được sự đồng thuận của Team Leader và cập nhật trực tiếp vào tệp đặc tả này.*
