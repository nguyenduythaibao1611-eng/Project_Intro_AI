"""
Unit Tests cho Module Suy diễn Độc lập (tests/test_inference.py)
==============================================================
Dự án: Vietnamese Review Sentiment Analysis (Shopee Reviews)
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách chính: Từ Văn Huy Hoàng (Training & Evaluation Engineer)

Kiểm thử toàn diện:
1. Suy diễn câu đơn (Single Sentence Inference)
2. Xử lý câu rỗng và khoảng trắng (Empty / Whitespace String Handling)
3. Kiểm tra định dạng đầu ra (label, confidence, latency_ms, probabilities)
4. Tiêu chuẩn thời gian suy diễn < 100ms/câu
5. Tính nhất quán giữa Bi-LSTM và PhoBERT
"""

import unittest
import time
from src.inference import predict_sentiment, clear_inference_cache


class TestInferenceModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clear_inference_cache()
        # Warm-up suy diễn lần đầu để nạp mô hình vào bộ nhớ đệm
        _ = predict_sentiment("Sản phẩm rất tốt", model_type="bilstm")
        _ = predict_sentiment("Sản phẩm rất tốt", model_type="phobert")

    def test_bilstm_single_sentence(self):
        text = "Sản phẩm dùng rất ưng ý, giao hàng siêu nhanh!"
        result = predict_sentiment(text, model_type="bilstm")

        self.assertIsInstance(result, dict)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("latency_ms", result)
        self.assertIn("probabilities", result)
        self.assertIn("processed_text", result)

        self.assertIn(result["label"], ["POS", "NEU", "NEG"])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        # Tổng xác suất 3 lớp phải bằng 1.0 (sai số làm tròn <= 1e-3)
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_bilstm_empty_and_whitespace_sentence(self):
        empty_texts = ["", "   ", "\n\t", None]
        for t in empty_texts:
            result = predict_sentiment(t, model_type="bilstm")
            self.assertIsInstance(result, dict)
            self.assertIn(result["label"], ["POS", "NEU", "NEG"])
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLess(result["latency_ms"], 100.0)

    def test_phobert_single_sentence(self):
        text = "Áo vải mỏng nhưng mặc mát, tạm ổn so với giá."
        result = predict_sentiment(text, model_type="phobert")

        self.assertIsInstance(result, dict)
        self.assertIn(result["label"], ["POS", "NEU", "NEG"])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_phobert_empty_and_whitespace_sentence(self):
        empty_texts = ["", "   ", "\n"]
        for t in empty_texts:
            result = predict_sentiment(t, model_type="phobert")
            self.assertIsInstance(result, dict)
            self.assertIn(result["label"], ["POS", "NEU", "NEG"])
            self.assertGreaterEqual(result["confidence"], 0.0)

    def test_latency_under_100ms_criteria(self):
        """Tiêu chuẩn nghiệm thu: Thời gian suy diễn dưới 100ms/câu."""
        sample_texts = [
            "Giao hàng nhanh đóng gói cẩn thận",
            "Chất lượng kém, không giống mô tả",
            "Bình thường, tiền nào của nấy",
            "Rất thích shop phục vụ nhiệt tình",
            "Đóng gói ẩu bị móp méo hộp",
        ]

        latencies = []
        for text in sample_texts:
            res = predict_sentiment(text, model_type="phobert")
            latencies.append(res["latency_ms"])

        avg_latency = sum(latencies) / len(latencies)
        self.assertLess(avg_latency, 100.0, f"Thời gian suy diễn trung bình ({avg_latency:.2f}ms) vượt quá 100ms!")

    def test_invalid_model_type_raises_error(self):
        with self.assertRaises(ValueError):
            predict_sentiment("Sản phẩm tốt", model_type="unsupported_model")


if __name__ == "__main__":
    unittest.main()
