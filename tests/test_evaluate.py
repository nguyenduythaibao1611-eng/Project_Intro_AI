"""
Kiểm thử Unit Tests cho Module Đánh giá & Bộ 15 Ca Thách thức (src/evaluate.py)
================================================================================
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách: Châu Tuấn Kiệt (kiet00394-collab)
Chạy kiểm thử: pytest tests/test_evaluate.py -v
"""

import sys
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import (
    BENCHMARK_COLUMNS,
    analyze_error_samples,
    build_benchmark_dataframe,
    check_neu_recall_threshold,
    compute_classification_metrics,
    evaluate_model_with_latency,
    make_pending_benchmark_record,
    measure_inference_latency,
    plot_and_save_confusion_matrix,
    validate_challenge_cases,
)
from src.preprocess import ID_TO_LABEL, preprocess_pipeline


def test_compute_classification_metrics_exact_and_mismatch():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert set(metrics["per_class"].keys()) == {"NEG", "NEU", "POS"}
    assert set(metrics["per_class_recall"].keys()) == {"NEG", "NEU", "POS"}
    assert metrics["per_class_recall"]["NEU"] == pytest.approx(1.0, abs=1e-4)

    with pytest.raises(ValueError):
        compute_classification_metrics([0, 1], [0])


def test_measure_inference_latency_and_benchmark_record(tmp_path):
    samples = ["sản_phẩm tốt", "giao_hàng chậm", "tạm được", "rất ưng_ý"]
    y_true = [2, 0, 1, 2]

    def fake_predict(batch):
        time.sleep(0.001)
        mapping = {"sản_phẩm tốt": 2, "giao_hàng chậm": 0, "tạm được": 1, "rất ưng_ý": 2}
        return [mapping.get(x, 1) for x in batch]

    preds, latency_ms = measure_inference_latency(
        fake_predict, samples, warmup=1, per_sample=True, max_timed_samples=3
    )
    assert len(preds) == len(samples)
    assert latency_ms > 0.0

    rec, y_pred = evaluate_model_with_latency(
        model_name="MockModel",
        y_true=y_true,
        predict_fn=fake_predict,
        samples=samples,
        warmup=1,
        per_sample_latency=False,
    )
    assert list(y_pred) == y_true
    assert rec["Accuracy"] == 100.0
    assert rec["Macro F1"] == 100.0
    assert set(rec.keys()) == set(BENCHMARK_COLUMNS)

    pending = make_pending_benchmark_record("PhoBERT v2")
    csv_out = tmp_path / "benchmark.csv"
    df = build_benchmark_dataframe([rec, pending], save_csv_path=str(csv_out))
    assert csv_out.exists()
    assert list(df.columns) == BENCHMARK_COLUMNS
    assert len(df) == 2

    with pytest.raises(ValueError):
        measure_inference_latency(fake_predict, [])


def test_build_benchmark_dataframe_missing_column():
    with pytest.raises(ValueError):
        build_benchmark_dataframe([{"Model Name": "Incomplete"}])


def test_confusion_matrix_plot_and_neu_threshold(tmp_path):
    # NEU (1): 3/4 đúng = 75% (>= 55%)
    y_true = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 2, 1, 1, 1, 2, 2, 2])

    cm_path = tmp_path / "phobert_cm.png"
    plot_and_save_confusion_matrix(
        y_true, y_pred, save_path=str(cm_path), title="Test 300 DPI CM"
    )
    assert cm_path.exists()
    assert cm_path.stat().st_size > 1000

    pass_check = check_neu_recall_threshold(y_true, y_pred, min_neu_recall=0.55)
    assert pass_check["neu_recall"] == pytest.approx(0.75, abs=1e-4)
    assert pass_check["passed"] is True

    # Trường hợp NEU thấp hơn 55% (1/4 = 25%)
    y_pred_low = np.array([0, 0, 1, 2, 2, 2, 2, 2])
    fail_check = check_neu_recall_threshold(y_true, y_pred_low, min_neu_recall=0.55)
    assert fail_check["neu_recall"] == pytest.approx(0.25, abs=1e-4)
    assert fail_check["passed"] is False


def test_analyze_error_samples_severity_order(tmp_path):
    texts = ["câu lỗi nhẹ", "câu đúng", "câu lỗi nặng đối cực"]
    y_true = [0, 2, 0]
    y_pred = [1, 2, 2]  # |0-1|=1, |0-2|=2 -> lỗi đối cực phải lên đầu

    err_csv = tmp_path / "errors.csv"
    df_err = analyze_error_samples(texts, y_true, y_pred, top_k=5, save_csv_path=str(err_csv))
    assert err_csv.exists()
    assert len(df_err) == 2
    assert df_err.iloc[0]["text"] == "câu lỗi nặng đối cực"
    assert df_err.iloc[0]["true_label"] == "NEG"
    assert df_err.iloc[0]["pred_label"] == "POS"


def test_demo_challenge_cases_schema_and_baseline_behavior():
    cases_path = REPO_ROOT / "docs" / "demo_challenge_cases.json"
    assert cases_path.exists(), "Không tìm thấy docs/demo_challenge_cases.json"

    summary = validate_challenge_cases(cases_path)
    assert summary["valid"] is True
    assert summary["total_cases"] == 15
    assert summary["category_counts"] == {
        "sarcasm": 5,
        "implicit_praise": 5,
        "teencode_slang": 5,
    }

    import json

    data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = data["cases"]

    vec_path = REPO_ROOT / "models" / "baseline_tfidf_vectorizer.pkl"
    svc_path = REPO_ROOT / "models" / "baseline_linear_svc.pkl"
    if vec_path.exists() and svc_path.exists():
        vec = joblib.load(vec_path)
        svc = joblib.load(svc_path)
        for case in cases:
            proc = preprocess_pipeline(case["text"])
            pred = ID_TO_LABEL[int(svc.predict(vec.transform([proc]))[0])]
            assert pred == case["baseline_pred"], (
                f"Baseline mismatch ở {case['id']}: thực tế={pred}, json={case['baseline_pred']}"
            )
            if case["category"] == "sarcasm":
                assert case["expected_label"] == "NEG"
                assert pred == "POS"
            elif case["category"] == "implicit_praise":
                assert case["expected_label"] == "POS"
                assert pred == "NEG"


def test_benchmark_comparison_csv_schema():
    csv_path = REPO_ROOT / "docs" / "benchmark_comparison.csv"
    assert csv_path.exists(), "Không tìm thấy docs/benchmark_comparison.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert list(df.columns) == BENCHMARK_COLUMNS
    assert list(df["Model Name"]) == ["LinearSVC (Baseline)", "Bi-LSTM", "PhoBERT v2"]
