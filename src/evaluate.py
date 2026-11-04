"""
Module Đánh giá Hiệu năng Mô hình (Model Evaluation & Metrics)
=============================================================
Dự án: Vietnamese Review Sentiment Analysis (Shopee Reviews)
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách chính: Từ Văn Huy Hoàng (Training & Evaluation Engineer)

Tệp mã nguồn cung cấp các hàm chuẩn hóa:
1. compute_classification_metrics: Tính Accuracy, Macro Precision, Recall, Macro F1, Weighted F1, Per-class F1.
2. plot_and_save_confusion_matrix: Vẽ ma trận nhầm lẫn chuẩn hóa (heatmap %) và xuất ảnh 300 DPI.
3. analyze_error_samples: Trích xuất các mẫu câu mô hình đoán sai nghiêm trọng nhất phục vụ Error Analysis.
"""

import os
import sys
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

# Cấu hình UTF-8 an toàn cho Windows Console tránh lỗi charmap / cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # Chế độ headless an toàn cho máy chủ/CI/non-GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def _to_numpy(data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
    """Chuyển đổi linh hoạt đầu vào (List, Series, PyTorch Tensor) sang numpy array 1D."""
    # Hỗ trợ PyTorch Tensor nếu có truyền vào
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    elif isinstance(data, pd.Series):
        data = data.to_numpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)
    return data.squeeze()


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_pred: Union[np.ndarray, List[int], pd.Series],
    label_names: List[str] = ["NEG", "NEU", "POS"]
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Tính toán các chỉ số đánh giá phân loại sắc thái cảm xúc.

    Tham số:
    ---------
    y_true : nhãn thực tế dạng số (0, 1, 2) hoặc mảng tương ứng.
    y_pred : nhãn dự đoán từ mô hình (0, 1, 2).
    label_names : danh sách tên nhãn tương ứng theo thứ tự index [0, 1, 2].
                  Mặc định: ["NEG", "NEU", "POS"].

    Trả về:
    -------
    Dict[str, Union[float, Dict[str, float]]] chứa:
    - 'accuracy': float
    - 'macro_precision': float (Macro Precision)
    - 'macro_recall': float (Macro Recall)
    - 'macro_f1': float (Macro F1-score - thước đo cốt lõi xử lý mất cân bằng nhãn)
    - 'weighted_f1': float (F1-score có trọng số theo kích thước từng lớp)
    - 'per_class': Dict chứa F1-score chi tiết của từng nhãn (ví dụ: {'NEG': 0.72, ...})
    """
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Kích thước y_true ({len(y_t)}) và y_pred ({len(y_p)}) không khớp nhau!"
        )

    # 1. Tính Accuracy
    acc = float(accuracy_score(y_t, y_p))

    # 2. Tính Macro Precision, Recall, F1
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_t, y_p, average="macro", zero_division=0
    )

    # 3. Tính Weighted F1
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    # 4. Tính F1 riêng cho từng lớp (Per-class F1)
    class_indices = list(range(len(label_names)))
    _, _, per_class_f1_arr, _ = precision_recall_fscore_support(
        y_t, y_p, labels=class_indices, average=None, zero_division=0
    )

    per_class_f1: Dict[str, float] = {}
    for idx, name in enumerate(label_names):
        score = float(per_class_f1_arr[idx]) if idx < len(per_class_f1_arr) else 0.0
        per_class_f1[name] = round(score, 4)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_f1,
    }


def plot_and_save_confusion_matrix(
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_pred: Union[np.ndarray, List[int], pd.Series],
    class_names: List[str] = ["NEG", "NEU", "POS"],
    save_path: str = "docs/confusion_matrix.png",
    title: str = "Confusion Matrix (Normalized)"
) -> None:
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) chuẩn hóa theo phần trăm và xuất file ảnh 300 DPI.

    Quy trình:
    ----------
    1. Tính confusion matrix với `normalize='true'` (chuẩn hóa theo hàng thực tế - Recall từng lớp).
    2. Trực quan hóa bằng heatmap của Seaborn với bảng màu `Blues`, hiển thị định dạng `.2%`.
    3. Trục X: Predicted Label, Trục Y: True Label.
    4. Tự động tạo thư mục chứa nếu chưa tồn tại và lưu file ảnh với độ phân giải cao (`dpi=300`).
    """
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)

    class_indices = list(range(len(class_names)))
    cm = confusion_matrix(y_t, y_p, labels=class_indices, normalize="true")

    # Tạo thư mục lưu ảnh nếu chưa có
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 12, "weight": "bold"},
        linewidths=0.5,
        linecolor="gray",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATION] Đã lưu Confusion Matrix chất lượng cao vào: {save_path}")


def analyze_error_samples(
    texts: Union[List[str], pd.Series, np.ndarray],
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_pred: Union[np.ndarray, List[int], pd.Series],
    id_to_label: Dict[int, str] = {0: "NEG", 1: "NEU", 2: "POS"},
    top_k: int = 50,
    save_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Trích xuất danh sách top_k mẫu câu mô hình đoán sai nghiêm trọng nhất phục vụ Error Analysis.

    Cơ chế xếp hạng độ nghiêm trọng (Severity Ranking):
    --------------------------------------------------
    Lỗi phân cực đối đầu cực đoan (Ví dụ: Thật là NEG nhưng đoán thành POS do dính từ khen mỉa mai,
    hoặc Thật là POS nhưng đoán thành NEG do câu phủ định kép) có mức độ lệch |y_true - y_pred| = 2.
    Những mẫu này được ưu tiên trích xuất lên đầu bảng để phân tích định tính chuyên sâu.

    Trả về:
    -------
    pd.DataFrame gồm 3 cột chuẩn: ['text', 'true_label', 'pred_label']
    """
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)

    if isinstance(texts, (pd.Series, np.ndarray)):
        text_list = texts.tolist()
    else:
        text_list = list(texts)

    if len(text_list) != len(y_t) or len(y_t) != len(y_p):
        raise ValueError("Độ dài texts, y_true và y_pred phải bằng nhau!")

    # Tìm các mẫu đoán sai
    error_indices = np.where(y_t != y_p)[0]

    # Tính độ nghiêm trọng của lỗi: |y_true - y_pred|
    severities = np.abs(y_t[error_indices] - y_p[error_indices])

    # Sắp xếp giảm dần theo độ nghiêm trọng (lỗi phân cực 2 điểm lên trước)
    sorted_order = np.argsort(-severities)
    selected_indices = error_indices[sorted_order][:top_k]

    results = []
    for idx in selected_indices:
        t_id = int(y_t[idx])
        p_id = int(y_p[idx])
        results.append({
            "text": text_list[idx],
            "true_label": id_to_label.get(t_id, str(t_id)),
            "pred_label": id_to_label.get(p_id, str(p_id)),
        })

    df_errors = pd.DataFrame(results, columns=["text", "true_label", "pred_label"])

    if save_csv_path:
        out_dir = os.path.dirname(save_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_errors.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[EVALUATION] Đã lưu {len(df_errors)} mẫu lỗi vào: {save_csv_path}")

    return df_errors


def print_metrics_summary(metrics: Dict[str, Union[float, Dict[str, float]]]) -> None:
    """In bảng tổng hợp các chỉ số đánh giá một cách trực quan ra màn hình Console/Notebook."""
    print("=" * 55)
    print("          BÁO CÁO HIỆU NĂNG MÔ HÌNH (EVALUATION METRICS) ")
    print("=" * 55)
    print(f" • Accuracy:         {metrics['accuracy'] * 100:.2f}%")
    print(f" • Macro Precision:  {metrics['macro_precision'] * 100:.2f}%")
    print(f" • Macro Recall:     {metrics['macro_recall'] * 100:.2f}%")
    print(f" • Macro F1-Score:   {metrics['macro_f1'] * 100:.2f}%  <-- (Chỉ số cốt lõi)")
    print(f" • Weighted F1:      {metrics['weighted_f1'] * 100:.2f}%")
    print("-" * 55)
    print(" F1-Score riêng từng lớp (Per-Class F1):")
    per_class = metrics["per_class"]
    for label, score in per_class.items():
        bar = "#" * int(score * 20)
        print(f"   [{label:^4}]: {score * 100:6.2f}% | {bar}")
    print("=" * 55)


if __name__ == "__main__":
    # Khối kiểm thử tự động (Unit Test / Demo Run)
    print("[TEST] Đang chạy kiểm thử module src/evaluate.py ...")

    # Tạo dữ liệu giả định mô phỏng bài toán 3 nhãn: NEG(0), NEU(1), POS(2)
    np.random.seed(42)
    sample_y_true = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 2, 0, 2])
    sample_y_pred = np.array([0, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 0, 0])
    sample_texts = [
        "Chất lượng quá kém, đóng gói rách nát",
        "Giao hàng chậm trễ, thái độ nhân viên tệ",
        "Áo mặc tạm được, vải hơi mỏng",
        "Bình thường, tiền nào của nấy thôi",
        "Cũng được, giao đủ số lượng đặt",
        "Sản phẩm rất đẹp, đúng như hình chụp",
        "Shop phục vụ tuyệt vời, 5 sao!",
        "Mùi thơm dễ chịu, bôi lên da mát dịu",
        "Màu sắc giống ảnh, đóng gói kỹ càng",
        "Rất ưng ý, sẽ ủng hộ shop lần sau",
        "Hàng fake lừa đảo, shop bảo hàng auth",
        "Mức giá hợp lý so với chất lượng",
        "Giao hàng siêu tốc trong 2 ngày",
        "Vừa mở hộp ra đã vỡ nắp",
        "Chất vải mềm mịn mặc rất thoáng mát",
    ]

    # 1. Kiểm thử compute_classification_metrics
    metrics = compute_classification_metrics(sample_y_true, sample_y_pred)
    print_metrics_summary(metrics)

    # 2. Kiểm thử plot_and_save_confusion_matrix
    test_cm_path = "docs/test_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        sample_y_true, sample_y_pred, save_path=test_cm_path, title="Test Confusion Matrix"
    )

    # 3. Kiểm thử analyze_error_samples
    test_error_path = "docs/test_error_analysis.csv"
    df_err = analyze_error_samples(
        sample_texts, sample_y_true, sample_y_pred, top_k=5, save_csv_path=test_error_path
    )
    print("\n[TEST] Top mẫu lỗi nghiêm trọng nhất:")
    print(df_err.to_string(index=False))

    print("\n[SUCCESS] Kiểm thử toàn bộ module src/evaluate.py thành công!")

