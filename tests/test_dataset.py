"""
Kiểm thử Unit Tests cho Module Dataset & Vocabulary (src/dataset.py)
====================================================================
Học phần: AI Capstone 2025 - VNUK
Phụ trách: Châu Tuấn Kiệt (Senior ML Engineer)
Chạy kiểm thử: python -m unittest tests/test_dataset.py -v
"""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import (
    MAX_SEQUENCE_LENGTH,
    PAD_IDX,
    PAD_TOKEN,
    UNK_IDX,
    UNK_TOKEN,
    SentimentDataset,
    Vocabulary,
    collate_sentiment_batch,
    create_data_loader,
)


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            "sản_phẩm rất tốt giao_hàng nhanh",
            "sản_phẩm đóng_gói cẩn_thận chất_lượng tốt",
            "giao_hàng chậm chất_lượng kém",
            "áo đẹp rất ưng_ý",
        ]

    def test_special_tokens_initialization(self):
        vocab = Vocabulary(min_freq=1)
        self.assertEqual(vocab[PAD_TOKEN], PAD_IDX)
        self.assertEqual(vocab[UNK_TOKEN], UNK_IDX)
        self.assertEqual(vocab.lookup_token(PAD_IDX), PAD_TOKEN)
        self.assertEqual(vocab.lookup_token(UNK_IDX), UNK_TOKEN)
        self.assertEqual(len(vocab), 2)

    def test_build_vocab_and_min_freq(self):
        # Với min_freq=2: các từ xuất hiện >= 2 lần mới được thêm
        # "sản_phẩm": 2, "rất": 2, "tốt": 2, "giao_hàng": 2, "chất_lượng": 2
        vocab = Vocabulary(min_freq=2)
        vocab.build_vocab(self.sentences)

        self.assertIn("sản_phẩm", vocab)
        self.assertIn("tốt", vocab)
        self.assertIn("giao_hàng", vocab)
        self.assertIn("chất_lượng", vocab)

        # "áo" và "đẹp" chỉ xuất hiện 1 lần -> không được thêm vào vocab
        self.assertNotIn("áo", vocab)
        self.assertNotIn("đẹp", vocab)
        self.assertEqual(vocab["áo"], UNK_IDX)
        self.assertEqual(vocab["đẹp"], UNK_IDX)

    def test_lookup_methods(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build_vocab(self.sentences)

        tokens = ["sản_phẩm", "tốt", "từ_lạ_chưa_gặp"]
        indices = vocab.lookup_indices(tokens)
        self.assertEqual(len(indices), 3)
        self.assertEqual(indices[2], UNK_IDX)

        recovered_tokens = vocab.lookup_tokens(indices)
        self.assertEqual(recovered_tokens[0], "sản_phẩm")
        self.assertEqual(recovered_tokens[1], "tốt")
        self.assertEqual(recovered_tokens[2], UNK_TOKEN)

    def test_serialization_save_and_load(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build_vocab(self.sentences)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vocab.json"
            vocab.save(path)
            self.assertTrue(path.exists())

            loaded_vocab = Vocabulary.load(path)
            self.assertEqual(len(loaded_vocab), len(vocab))
            self.assertEqual(loaded_vocab.stoi, vocab.stoi)
            self.assertEqual(loaded_vocab.itos, vocab.itos)


class TestSentimentDataset(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            "sản_phẩm rất tốt",
            "giao_hàng chậm dịch_vụ tệ",
            "bình_thường không có gì đặc_biệt",
            "",  # Mẫu rỗng để kiểm thử độ bền (robustness)
        ]
        self.labels = ["POS", "NEG", "NEU", "NEU"]
        self.vocab = Vocabulary(min_freq=1)
        self.vocab.build_vocab(self.sentences)

    def test_dataset_item_structure_and_lengths(self):
        dataset = SentimentDataset(
            texts=self.sentences,
            labels=self.labels,
            vocab=self.vocab,
            max_length=MAX_SEQUENCE_LENGTH,
        )

        self.assertEqual(len(dataset), 4)

        # Kiểm tra mẫu 1
        item0 = dataset[0]
        self.assertIn("input_ids", item0)
        self.assertIn("length", item0)
        self.assertIn("label", item0)
        self.assertIn("text", item0)

        self.assertEqual(item0["input_ids"].shape, torch.Size([MAX_SEQUENCE_LENGTH]))
        self.assertEqual(item0["length"], 3)  # "sản_phẩm rất tốt" có 3 tokens
        self.assertEqual(item0["label"].item(), 2)  # POS -> 2

        # Kiểm tra phần padding (các vị trí từ 3 trở đi phải là 0)
        self.assertTrue((item0["input_ids"][3:] == PAD_IDX).all())

    def test_empty_string_handling(self):
        dataset = SentimentDataset(
            texts=[""],
            labels=[0],
            vocab=self.vocab,
            max_length=10,
        )
        item = dataset[0]
        # Câu rỗng vẫn phải có length >= 1 để an toàn cho pack_padded_sequence
        self.assertGreaterEqual(item["length"], 1)
        self.assertEqual(item["input_ids"].shape, torch.Size([10]))

    def test_truncation(self):
        long_sentence = " ".join(["từ"] * 100)
        dataset = SentimentDataset(
            texts=[long_sentence],
            labels=[1],
            vocab=self.vocab,
            max_length=64,
        )
        item = dataset[0]
        self.assertEqual(item["input_ids"].shape, torch.Size([64]))
        self.assertEqual(item["length"], 64)

    def test_collate_fn_and_dataloader(self):
        dataset = SentimentDataset(
            texts=self.sentences,
            labels=self.labels,
            vocab=self.vocab,
            max_length=32,
        )
        loader = create_data_loader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        self.assertIn("input_ids", batch)
        self.assertIn("lengths", batch)
        self.assertIn("labels", batch)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 32]))
        self.assertEqual(batch["lengths"].shape, torch.Size([2]))
        self.assertEqual(batch["labels"].shape, torch.Size([2]))


if __name__ == "__main__":
    unittest.main()
