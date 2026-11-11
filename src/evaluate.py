"""
Module Đánh giá Hiệu năng Mô hình (Model Evaluation & Metrics)
=============================================================
Dự án: Vietnamese Review Sentiment Analysis (Shopee Reviews)
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách chính: Từ Văn Huy Hoàng (Week 3) & Châu Tuấn Kiệt (Week 4 QA/Benchmark)

Tệp mã nguồn cung cấp các hàm chuẩn hóa:
1. compute_classification_metrics: Tính Accuracy, Macro Precision, Recall, Macro F1, Weighted F1, Per-class F1 & Recall.
2. plot_and_save_confusion_matrix: Vẽ ma trận nhầm lẫn chuẩn hóa (heatmap %) và xuất ảnh 300 DPI.
3. analyze_error_samples: Trích xuất các mẫu câu mô hình đoán sai nghiêm trọng nhất phục vụ Error Analysis.
4. measure_inference_latency & evaluate_model_with_latency: Đo độ trễ suy diễn (ms/câu) và chỉ số đa lớp.
5. build_benchmark_dataframe & run_benchmark_suite: Đối chứng mô hình (LinearSVC, Bi-LSTM, PhoBERT v2) và xuất CSV.
6. validate_challenge_cases: Thẩm định cấu trúc bộ 15 ca kiểm thử thách thức cho Web Demo.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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

LABEL_NAMES: List[str] = ["NEG", "NEU", "POS"]
ID_TO_LABEL: Dict[int, str] = {0: "NEG", 1: "NEU", 2: "POS"}
BENCHMARK_COLUMNS: List[str] = [
    "Model Name",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "F1-POS",
    "F1-NEU",
    "F1-NEG",
    "Inference Latency (ms)",
]
CHALLENGE_CATEGORIES: Tuple[str, ...] = ("sarcasm", "implicit_praise", "teencode_slang")


def _to_numpy(data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
    """Chuyển đổi linh hoạt đầu vào (List, Series, PyTorch Tensor) sang numpy array 1D."""
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    elif isinstance(data, pd.Series):
        data = data.to_numpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)
    return np.atleast_1d(data.squeeze())


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_pred: Union[np.ndarray, List[int], pd.Series],
    label_names: List[str] = LABEL_NAMES,
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
    - 'per_class': Dict chứa F1-score chi tiết của từng nhãn
    - 'per_class_precision': Dict chứa Precision chi tiết của từng nhãn
    - 'per_class_recall': Dict chứa Recall chi tiết của từng nhãn
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

    # 4. Tính Precision, Recall, F1 riêng cho từng lớp
    class_indices = list(range(len(label_names)))
    per_class_prec_arr, per_class_rec_arr, per_class_f1_arr, _ = (
        precision_recall_fscore_support(
            y_t, y_p, labels=class_indices, average=None, zero_division=0
        )
    )

    per_class_f1: Dict[str, float] = {}
    per_class_precision: Dict[str, float] = {}
    per_class_recall: Dict[str, float] = {}
    for idx, name in enumerate(label_names):
        f1_val = float(per_class_f1_arr[idx]) if idx < len(per_class_f1_arr) else 0.0
        prec_val = float(per_class_prec_arr[idx]) if idx < len(per_class_prec_arr) else 0.0
        rec_val = float(per_class_rec_arr[idx]) if idx < len(per_class_rec_arr) else 0.0
        per_class_f1[name] = round(f1_val, 4)
        per_class_precision[name] = round(prec_val, 4)
        per_class_recall[name] = round(rec_val, 4)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


def plot_and_save_confusion_matrix(
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_pred: Union[np.ndarray, List[int], pd.Series],
    class_names: List[str] = LABEL_NAMES,
    save_path: str = "docs/confusion_matrix.png",
    title: str = "Confusion Matrix (Normalized)",
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


def check_neu_recall_threshold(
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_pred: Union[np.ndarray, List[int], pd.Series],
    class_names: List[str] = LABEL_NAMES,
    min_neu_recall: float = 0.55,
) -> Dict[str, Any]:
    """
    Kiểm tra tỷ lệ chuẩn hóa theo hàng thực tế của ma trận nhầm lẫn và xác nhận ngưỡng lớp NEU.

    Tham số:
    ---------
    min_neu_recall : Ngưỡng độ chính xác tối thiểu của lớp NEU (mặc định >= 55% = 0.55).
    """
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    class_indices = list(range(len(class_names)))
    cm = confusion_matrix(y_t, y_p, labels=class_indices, normalize="true")

    per_class_recall = {
        name: round(float(cm[idx, idx]), 4) for idx, name in enumerate(class_names)
    }
    neu_recall = per_class_recall.get("NEU", 0.0)
    return {
        "matrix": cm,
        "per_class_recall": per_class_recall,
        "neu_recall": neu_recall,
        "min_neu_recall": min_neu_recall,
        "passed": bool(neu_recall >= min_neu_recall),
    }


def analyze_error_samples(
    texts: Union[List[str], pd.Series, np.ndarray],
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_pred: Union[np.ndarray, List[int], pd.Series],
    id_to_label: Dict[int, str] = ID_TO_LABEL,
    top_k: int = 50,
    save_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Trích xuất danh sách top_k mẫu câu mô hình đoán sai nghiêm trọng nhất phục vụ Error Analysis.

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

    error_indices = np.where(y_t != y_p)[0]
    severities = np.abs(y_t[error_indices] - y_p[error_indices])
    sorted_order = np.argsort(-severities)
    selected_indices = error_indices[sorted_order][:top_k]

    results = []
    for idx in selected_indices:
        t_id = int(y_t[idx])
        p_id = int(y_p[idx])
        results.append(
            {
                "text": text_list[idx],
                "true_label": id_to_label.get(t_id, str(t_id)),
                "pred_label": id_to_label.get(p_id, str(p_id)),
            }
        )

    df_errors = pd.DataFrame(results, columns=["text", "true_label", "pred_label"])

    if save_csv_path:
        out_dir = os.path.dirname(save_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_errors.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[EVALUATION] Đã lưu {len(df_errors)} mẫu lỗi vào: {save_csv_path}")

    return df_errors


def measure_inference_latency(
    predict_fn: Callable[[Sequence[Any]], Any],
    samples: Sequence[Any],
    warmup: int = 1,
    per_sample: bool = False,
    max_timed_samples: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Đo độ trễ suy diễn trung bình (ms/câu) và trả về mảng dự đoán trên toàn bộ tập samples.

    Tham số:
    ---------
    predict_fn : Hàm nhận Sequence đầu vào và trả về nhãn dự đoán [N].
    samples : Chuỗi mẫu đầu vào để đánh giá.
    warmup : Số lần chạy làm nóng trước khi bấm giờ (tránh sai lệch JIT/cache).
    per_sample : True để đo trung bình thời gian suy diễn từng câu riêng lẻ (mô phỏng Web Demo);
                 False để đo suy diễn toàn bộ lô chia cho số lượng mẫu.
    max_timed_samples : Giới hạn số mẫu đo khi per_sample=True để tối ưu thời gian benchmark.
    """
    sample_list = list(samples)
    total_samples = len(sample_list)
    if total_samples == 0:
        raise ValueError("Danh sách samples để đo latency không được rỗng!")

    # Chạy warmup
    warm_slice = sample_list[: min(4, total_samples)]
    for _ in range(max(0, warmup)):
        _ = predict_fn(warm_slice)

    if per_sample:
        timed_slice = (
            sample_list[:max_timed_samples] if max_timed_samples else sample_list
        )
        t0 = time.perf_counter()
        for item in timed_slice:
            _ = predict_fn([item])
        t1 = time.perf_counter()
        avg_latency_ms = ((t1 - t0) * 1000.0) / len(timed_slice)
        y_pred = _to_numpy(predict_fn(sample_list))
    else:
        t0 = time.perf_counter()
        raw_preds = predict_fn(sample_list)
        t1 = time.perf_counter()
        y_pred = _to_numpy(raw_preds)
        avg_latency_ms = ((t1 - t0) * 1000.0) / total_samples

    return y_pred.astype(int), round(float(avg_latency_ms), 4)


def evaluate_model_with_latency(
    model_name: str,
    y_true: Union[np.ndarray, List[int], pd.Series],
    predict_fn: Callable[[Sequence[Any]], Any],
    samples: Sequence[Any],
    label_names: List[str] = LABEL_NAMES,
    warmup: int = 1,
    per_sample_latency: bool = True,
    max_latency_samples: int = 200,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Đánh giá toàn diện một mô hình gồm chỉ số phân loại đa lớp và độ trễ suy diễn (ms).

    Trả về:
    -------
    (record_dict, y_pred_array) với record_dict khớp chính xác các cột của BENCHMARK_COLUMNS.
    """
    y_pred, latency_ms = measure_inference_latency(
        predict_fn=predict_fn,
        samples=samples,
        warmup=warmup,
        per_sample=per_sample_latency,
        max_timed_samples=max_latency_samples,
    )
    metrics = compute_classification_metrics(y_true, y_pred, label_names=label_names)
    per_class = metrics["per_class"]

    record: Dict[str, Any] = {
        "Model Name": model_name,
        "Accuracy": round(float(metrics["accuracy"]) * 100.0, 2),
        "Macro Precision": round(float(metrics["macro_precision"]) * 100.0, 2),
        "Macro Recall": round(float(metrics["macro_recall"]) * 100.0, 2),
        "Macro F1": round(float(metrics["macro_f1"]) * 100.0, 2),
        "F1-POS": round(float(per_class["POS"]) * 100.0, 2),
        "F1-NEU": round(float(per_class["NEU"]) * 100.0, 2),
        "F1-NEG": round(float(per_class["NEG"]) * 100.0, 2),
        "Inference Latency (ms)": round(latency_ms, 3),
    }
    return record, y_pred


def make_pending_benchmark_record(model_name: str, status: str = "PENDING_CHECKPOINT") -> Dict[str, Any]:
    """Tạo dòng giữ chỗ minh bạch khi checkpoint mô hình chưa được cung cấp."""
    return {
        "Model Name": model_name,
        "Accuracy": status,
        "Macro Precision": status,
        "Macro Recall": status,
        "Macro F1": status,
        "F1-POS": status,
        "F1-NEU": status,
        "F1-NEG": status,
        "Inference Latency (ms)": status,
    }


def build_benchmark_dataframe(
    records: List[Dict[str, Any]],
    save_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Tổng hợp bảng so sánh benchmark chuẩn theo các cột quy định và xuất file CSV nếu cần.
    """
    for idx, rec in enumerate(records):
        missing = [col for col in BENCHMARK_COLUMNS if col not in rec]
        if missing:
            raise ValueError(f"Bản ghi thứ {idx} thiếu cột bắt buộc: {missing}")

    df = pd.DataFrame(records)[BENCHMARK_COLUMNS]
    if save_csv_path:
        out_dir = os.path.dirname(save_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[BENCHMARK] Đã lưu bảng đối chứng mô hình tại: {save_csv_path}")
    return df


def validate_challenge_cases(
    cases_input: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
    expected_total: int = 15,
    per_category: int = 5,
    valid_labels: Sequence[str] = LABEL_NAMES,
) -> Dict[str, Any]:
    """
    Kiểm tra tính hợp lệ của bộ 15 ca kiểm thử thách thức (docs/demo_challenge_cases.json).

    Yêu cầu:
    --------
    - Đủ `expected_total` (15) mẫu: 5 `sarcasm`, 5 `implicit_praise`, 5 `teencode_slang`.
    - Mỗi phần tử phải có các trường: `id`, `title`, `category`, `text`, `expected_label`, `baseline_pred`.
    - Nhãn phải thuộc tập hợp lệ (`NEG`, `NEU`, `POS`).
    """
    if isinstance(cases_input, (str, Path)):
        raw_data = json.loads(Path(cases_input).read_text(encoding="utf-8"))
    else:
        raw_data = cases_input

    cases = raw_data["cases"] if isinstance(raw_data, dict) and "cases" in raw_data else raw_data
    if not isinstance(cases, list):
        raise ValueError("Dữ liệu ca kiểm thử phải là list hoặc dict có khóa 'cases'.")

    if len(cases) != expected_total:
        raise ValueError(f"Kỳ vọng đúng {expected_total} ca kiểm thử, nhận được {len(cases)}.")

    required_fields = {"id", "title", "category", "text", "expected_label", "baseline_pred"}
    counts: Dict[str, int] = {cat: 0 for cat in CHALLENGE_CATEGORIES}
    seen_ids = set()

    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Phần tử thứ {idx} không phải dict.")
        missing = required_fields - set(case.keys())
        if missing:
            raise ValueError(f"Ca kiểm thử #{idx} thiếu trường: {sorted(missing)}")

        cid = str(case["id"]).strip()
        if not cid or cid in seen_ids:
            raise ValueError(f"ID ca kiểm thử không hợp lệ hoặc trùng lặp: '{cid}'")
        seen_ids.add(cid)

        cat = case["category"]
        if cat not in counts:
            raise ValueError(f"Category '{cat}' ở '{cid}' phải thuộc {CHALLENGE_CATEGORIES}")
        counts[cat] += 1

        if not str(case["text"]).strip():
            raise ValueError(f"Nội dung 'text' của '{cid}' không được rỗng.")

        for lbl_field in ("expected_label", "baseline_pred"):
            if case[lbl_field] not in valid_labels:
                raise ValueError(
                    f"Nhãn {lbl_field}='{case[lbl_field]}' ở '{cid}' không thuộc {valid_labels}"
                )

    for cat, count in counts.items():
        if count != per_category:
            raise ValueError(f"Category '{cat}' cần đúng {per_category} câu, hiện có {count}.")

    return {
        "total_cases": len(cases),
        "category_counts": counts,
        "valid": True,
    }


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


def _evaluate_linearsvc_baseline(
    texts: List[str],
    y_true: np.ndarray,
    vec_path: Path,
    svc_path: Path,
) -> Optional[Dict[str, Any]]:
    if not (vec_path.exists() and svc_path.exists()):
        return None
    import joblib

    vectorizer = joblib.load(vec_path)
    clf_svc = joblib.load(svc_path)

    def predict_fn(batch_texts: Sequence[str]) -> np.ndarray:
        return clf_svc.predict(vectorizer.transform(batch_texts))

    record, _ = evaluate_model_with_latency(
        model_name="LinearSVC (Baseline)",
        y_true=y_true,
        predict_fn=predict_fn,
        samples=texts,
        warmup=2,
        per_sample_latency=True,
        max_latency_samples=200,
    )
    return record


def _load_checkpoint_payload(ckpt_path: Path, device: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tải state_dict và metadata tùy chọn từ checkpoint (.pt) tương thích với src/train.py và src/inference.py."""
    import torch

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload
    return payload, {}


def _evaluate_bilstm_if_available(
    texts: List[str],
    y_true: np.ndarray,
    ckpt_path: Path,
    vocab_path: Path,
) -> Optional[Dict[str, Any]]:
    if not (ckpt_path.exists() and vocab_path.exists()):
        return None
    import torch
    from src.dataset import MAX_SEQUENCE_LENGTH, SentimentDataset, Vocabulary, create_data_loader
    from src.models import BiLSTMClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(vocab_path)
    state_dict, meta = _load_checkpoint_payload(ckpt_path, device)
    model = BiLSTMClassifier(
        vocab_size=int(meta.get("vocab_size", len(vocab))),
        embedding_dim=int(meta.get("embedding_dim", 128)),
        hidden_dim=int(meta.get("hidden_dim", 128)),
        output_dim=3,
        num_layers=int(meta.get("num_layers", 2)),
        dropout_rate=float(meta.get("dropout_rate", 0.3)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    def predict_fn(batch_texts: Sequence[str]) -> np.ndarray:
        ds = SentimentDataset(texts=list(batch_texts), vocab=vocab, max_length=MAX_SEQUENCE_LENGTH)
        loader = create_data_loader(ds, batch_size=64 if len(batch_texts) > 1 else 1, shuffle=False)
        preds: List[int] = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        return np.array(preds)

    record, _ = evaluate_model_with_latency(
        model_name="Bi-LSTM",
        y_true=y_true,
        predict_fn=predict_fn,
        samples=texts,
        warmup=1,
        per_sample_latency=True,
        max_latency_samples=100,
    )
    return record


def _evaluate_phobert_if_available(
    texts: List[str],
    y_true: np.ndarray,
    ckpt_path: Path,
    cm_save_path: Path,
    pretrained_name: str = "vinai/phobert-base-v2",
) -> Optional[Dict[str, Any]]:
    if not ckpt_path.exists():
        return None
    try:
        import torch
        from transformers import AutoTokenizer
        from src.models import PhoBERTClassifier
    except ImportError:
        print("[BENCHMARK] Có checkpoint PhoBERT nhưng thiếu thư viện `transformers`.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, meta = _load_checkpoint_payload(ckpt_path, device)
    model_name = str(meta.get("model_name", pretrained_name))
    num_classes = int(meta.get("num_classes", 3))
    dropout_rate = float(meta.get("dropout_rate", 0.2))

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = PhoBERTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    def predict_fn(batch_texts: Sequence[str]) -> np.ndarray:
        preds: List[int] = []
        batch_size = 32 if len(batch_texts) > 1 else 1
        with torch.no_grad():
            for start in range(0, len(batch_texts), batch_size):
                chunk = list(batch_texts[start : start + batch_size])
                enc = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                token_type_ids = enc.get("token_type_ids")
                logits = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                    token_type_ids=token_type_ids.to(device) if token_type_ids is not None else None,
                )
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        return np.array(preds)

    record, y_pred = evaluate_model_with_latency(
        model_name="PhoBERT v2",
        y_true=y_true,
        predict_fn=predict_fn,
        samples=texts,
        warmup=1,
        per_sample_latency=True,
        max_latency_samples=50,
    )
    plot_and_save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=LABEL_NAMES,
        save_path=str(cm_save_path),
        title="PhoBERT v2 Normalized Confusion Matrix (Test Set)",
    )
    neu_check = check_neu_recall_threshold(y_true, y_pred, min_neu_recall=0.55)
    print(
        f"[BENCHMARK] PhoBERT NEU Recall: {neu_check['neu_recall']*100:.2f}% "
        f"(Threshold >= 55%: {'PASS' if neu_check['passed'] else 'FAIL'})"
    )
    return record


def run_benchmark_pipeline(
    test_csv: Union[str, Path] = "data/processed/test.csv",
    output_csv: Union[str, Path] = "docs/benchmark_comparison.csv",
    phobert_cm_path: Union[str, Path] = "docs/phobert_confusion_matrix.png",
    models_dir: Union[str, Path] = "models",
) -> pd.DataFrame:
    """
    Chạy quy trình đánh giá đối chứng 3 mô hình trên tập test độc lập và xuất bảng benchmark CSV.
    Các mô hình chưa có file trọng số trong `models/` được ghi nhận rõ trạng thái PENDING_CHECKPOINT.
    """
    test_path = Path(test_csv)
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tập test tại: {test_path}")

    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    texts = df_test["processed_content"].fillna("").astype(str).tolist()
    y_true = df_test["label_id"].to_numpy()

    m_dir = Path(models_dir)
    records: List[Dict[str, Any]] = []

    svc_rec = _evaluate_linearsvc_baseline(
        texts,
        y_true,
        m_dir / "baseline_tfidf_vectorizer.pkl",
        m_dir / "baseline_linear_svc.pkl",
    )
    records.append(svc_rec if svc_rec is not None else make_pending_benchmark_record("LinearSVC (Baseline)"))

    bilstm_vocab = m_dir / "vocab.json"
    if not bilstm_vocab.exists():
        bilstm_vocab = m_dir / "bilstm_vocab.json"

    bilstm_rec = _evaluate_bilstm_if_available(
        texts,
        y_true,
        m_dir / "best_bilstm.pt",
        bilstm_vocab,
    )
    records.append(bilstm_rec if bilstm_rec is not None else make_pending_benchmark_record("Bi-LSTM"))

    phobert_rec = _evaluate_phobert_if_available(
        texts,
        y_true,
        m_dir / "best_phobert.pt",
        Path(phobert_cm_path),
    )
    records.append(phobert_rec if phobert_rec is not None else make_pending_benchmark_record("PhoBERT v2"))

    return build_benchmark_dataframe(records, save_csv_path=str(output_csv))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy benchmark 3 mô hình và xuất CSV/Confusion Matrix.")
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--output-csv", default="docs/benchmark_comparison.csv")
    parser.add_argument("--phobert-cm", default="docs/phobert_confusion_matrix.png")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    df = run_benchmark_pipeline(
        test_csv=args.test_csv,
        output_csv=args.output_csv,
        phobert_cm_path=args.phobert_cm,
        models_dir=args.models_dir,
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
