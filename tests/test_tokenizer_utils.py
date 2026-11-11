"""Kiểm thử Unit Tests cho Module Tokenizer PhoBERT (src/tokenizer_utils.py)
==========================================================================
Dự án: Vietnamese Review Sentiment Analysis (Shopee Reviews)
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách: Bùi Đình Hiếu (Data Engineer / Preprocessing Specialist)
Chạy kiểm thử: python -m unittest discover -s tests -v -p "test_tokenizer_utils.py"
"""

import unittest
import torch

from src.tokenizer_utils import (
    DEFAULT_MODEL_NAME,
    MAX_SEQUENCE_LENGTH,
    batch_encode,
    decode_tokens,
    encode_reviews,
    get_phobert_tokenizer,
)


class TestPhoBERTTokenizerUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Nạp tokenizer một lần cho toàn bộ test suite để tối ưu tốc độ
        cls.tokenizer = get_phobert_tokenizer()

    def test_tokenizer_loading_and_singleton(self):
        tok1 = get_phobert_tokenizer()
        tok2 = get_phobert_tokenizer()
        self.assertIs(tok1, tok2, "get_phobert_tokenizer phải trả về cùng một Singleton instance")
        self.assertEqual(tok1.vocab_size, 64000, "Từ điển PhoBERT v2 phải có đúng 64.000 token")

    def test_encode_single_string(self):
        text = "sản_phẩm rất tốt giao_hàng nhanh"
        encoded = encode_reviews(text, max_len=MAX_SEQUENCE_LENGTH, tokenizer=self.tokenizer)

        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(input_ids.shape, torch.Size([1, MAX_SEQUENCE_LENGTH]))
        self.assertEqual(attention_mask.shape, torch.Size([1, MAX_SEQUENCE_LENGTH]))
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(attention_mask.dtype, torch.long)

        # Token đầu tiên là <s> (ID = 0)
        self.assertEqual(input_ids[0, 0].item(), self.tokenizer.bos_token_id)
        # Các vị trí có từ thực tế phải có mask = 1
        self.assertEqual(attention_mask[0, 0].item(), 1)

    def test_encode_multiple_reviews_batch(self):
        texts = [
            "áo đẹp mặc rất ưng ý",
            "chất lượng tệ không giống hình quảng cáo",
            "tạm được, đóng gói bình thường"
        ]
        encoded = encode_reviews(texts, max_len=32, tokenizer=self.tokenizer)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        self.assertEqual(input_ids.shape, torch.Size([3, 32]))
        self.assertEqual(attention_mask.shape, torch.Size([3, 32]))

    def test_encode_with_preprocessing_integration(self):
        # Chuỗi thô chứa teencode và ký tự lặp kéo dài
        raw_text = "sp này đẹpppp quáaaaa ko chê vào đâu đc"
        encoded = encode_reviews(
            raw_text,
            max_len=MAX_SEQUENCE_LENGTH,
            tokenizer=self.tokenizer,
            apply_preprocessing=True
        )

        decoded = decode_tokens(encoded["input_ids"][0], tokenizer=self.tokenizer)
        # Kiểm tra teencode đã được chuẩn hóa trong chuỗi token
        self.assertIn("sản_phẩm", decoded)
        self.assertIn("đẹp", decoded)
        self.assertIn("không", decoded)

    def test_batch_encode_generator(self):
        sample_texts = [f"Bình luận đánh giá mẫu số {i}" for i in range(10)]
        batch_size = 4
        batches = list(batch_encode(sample_texts, batch_size=batch_size, max_len=16, tokenizer=self.tokenizer))

        self.assertEqual(len(batches), 3)  # 4 + 4 + 2 = 10 mẫu
        self.assertEqual(batches[0]["input_ids"].shape, torch.Size([4, 16]))
        self.assertEqual(batches[1]["input_ids"].shape, torch.Size([4, 16]))
        self.assertEqual(batches[2]["input_ids"].shape, torch.Size([2, 16]))

    def test_decode_tokens_1d_and_2d(self):
        text = "giao_hàng nhanh đóng_gói cẩn_thận"
        encoded = encode_reviews(text, max_len=16, tokenizer=self.tokenizer)

        # 1D decode
        decoded_1d = decode_tokens(encoded["input_ids"][0], tokenizer=self.tokenizer)
        self.assertIsInstance(decoded_1d, str)
        self.assertIn("giao_hàng", decoded_1d)

        # 2D decode
        decoded_2d = decode_tokens(encoded["input_ids"], tokenizer=self.tokenizer)
        self.assertIsInstance(decoded_2d, list)
        self.assertEqual(len(decoded_2d), 1)
        self.assertIn("giao_hàng", decoded_2d[0])

    def test_truncation_and_padding_behavior(self):
        long_text = "sản_phẩm " * 100  # Vượt quá 64 tokens
        encoded = encode_reviews(long_text, max_len=MAX_SEQUENCE_LENGTH, tokenizer=self.tokenizer)
        self.assertEqual(encoded["input_ids"].shape, torch.Size([1, MAX_SEQUENCE_LENGTH]))

        # Chuỗi rỗng
        encoded_empty = encode_reviews("", max_len=MAX_SEQUENCE_LENGTH, tokenizer=self.tokenizer)
        self.assertEqual(encoded_empty["input_ids"].shape, torch.Size([1, MAX_SEQUENCE_LENGTH]))


if __name__ == "__main__":
    unittest.main()
