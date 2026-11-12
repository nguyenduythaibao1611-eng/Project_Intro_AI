"""
Kiểm thử Unit Tests cho Mô hình Cơ sở Baseline (TF-IDF + Logistic Regression / LinearSVC)
========================================================================================
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách: Bùi Đình Hiếu (AI/ML Engineer)
Chạy kiểm thử: python -m unittest tests/test_baseline.py -v
"""

from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluate import compute_classification_metrics


class TestBaselineModelArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.models_dir = cls.root_dir / "models"
        cls.vec_path = cls.models_dir / "baseline_tfidf_vectorizer.pkl"
        cls.lr_path = cls.models_dir / "baseline_logistic_regression.pkl"
        cls.svm_path = cls.models_dir / "baseline_linear_svc.pkl"

    def test_artifact_files_exist(self):
        self.assertTrue(self.vec_path.exists(), f"Không tìm thấy {self.vec_path}")
        self.assertTrue(self.lr_path.exists(), f"Không tìm thấy {self.lr_path}")
        self.assertTrue(self.svm_path.exists(), f"Không tìm thấy {self.svm_path}")

    def test_vectorizer_hyperparameters(self):
        vectorizer: TfidfVectorizer = joblib.load(self.vec_path)
        self.assertIsInstance(vectorizer, TfidfVectorizer)
        self.assertEqual(vectorizer.ngram_range, (1, 2))
        self.assertEqual(vectorizer.max_features, 5000)
        self.assertEqual(vectorizer.min_df, 2)
        self.assertTrue(vectorizer.sublinear_tf)
        self.assertLessEqual(len(vectorizer.vocabulary_), 5000)

    def test_logistic_regression_hyperparameters(self):
        clf_lr: LogisticRegression = joblib.load(self.lr_path)
        self.assertIsInstance(clf_lr, LogisticRegression)
        self.assertEqual(clf_lr.C, 1.0)
        self.assertEqual(clf_lr.solver, "lbfgs")
        self.assertEqual(clf_lr.class_weight, "balanced")
        self.assertGreaterEqual(clf_lr.max_iter, 1000)

    def test_linear_svc_hyperparameters(self):
        clf_svm: LinearSVC = joblib.load(self.svm_path)
        self.assertIsInstance(clf_svm, LinearSVC)
        self.assertEqual(clf_svm.C, 0.5)
        self.assertEqual(clf_svm.class_weight, "balanced")


class TestBaselineInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.vec = joblib.load(cls.root_dir / "models" / "baseline_tfidf_vectorizer.pkl")
        cls.lr = joblib.load(cls.root_dir / "models" / "baseline_logistic_regression.pkl")

    def test_sample_prediction_and_shapes(self):
        sample_texts = [
            "sản_phẩm rất tốt giao_hàng nhanh đóng_gói cẩn_thận",
            "hàng quá tệ không giống hình chất_lượng kém",
            "tạm được đóng_gói bình_thường chất_lượng ổn"
        ]
        features = self.vec.transform(sample_texts)
        self.assertEqual(features.shape, (len(sample_texts), 5000))

        predictions = self.lr.predict(features)
        self.assertEqual(len(predictions), 3)
        for pred in predictions:
            self.assertIn(pred, [0, 1, 2])

        # Phân phối xác suất (Predict Proba)
        probs = self.lr.predict_proba(features)
        self.assertEqual(probs.shape, (3, 3))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-5)

        # Mẫu tích cực phải ra POS (2)
        self.assertEqual(predictions[0], 2)
        # Mẫu tiêu cực phải ra NEG (0)
        self.assertEqual(predictions[1], 0)


class TestBaselineBenchmarkQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.test_csv = cls.root_dir / "data" / "processed" / "test.csv"
        cls.vec = joblib.load(cls.root_dir / "models" / "baseline_tfidf_vectorizer.pkl")
        cls.lr = joblib.load(cls.root_dir / "models" / "baseline_logistic_regression.pkl")

    def test_benchmark_metrics_threshold(self):
        if not self.test_csv.exists():
            self.skipTest("test.csv chưa được khởi tạo, bỏ qua kiểm thử chỉ số benchmark")

        df_test = pd.read_csv(self.test_csv)
        X_test = self.vec.transform(df_test["processed_content"].fillna(""))
        y_test = df_test["label_id"].values

        y_pred = self.lr.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred, label_names=["NEG", "NEU", "POS"])

        # Kiểm tra chỉ số đạt ngưỡng cam kết theo đặc tả và slide
        self.assertGreaterEqual(metrics["accuracy"], 0.70)
        self.assertGreaterEqual(metrics["macro_f1"], 0.63)
        self.assertIn("NEG", metrics["per_class"])
        self.assertIn("NEU", metrics["per_class"])
        self.assertIn("POS", metrics["per_class"])


if __name__ == "__main__":
    unittest.main()
