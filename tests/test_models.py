"""
Kiểm thử Unit Tests cho Module Mô hình Học sâu (src/models.py)
==============================================================
Học phần: AI Capstone 2025 - VNUK
Phụ trách: Châu Tuấn Kiệt (Senior ML Engineer)
Chạy kiểm thử: python -m unittest tests/test_models.py -v
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from src.models import (
    BiLSTMClassifier,
    compute_class_weights,
    get_weighted_criterion,
)


class TestBiLSTMClassifier(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 50
        self.embedding_dim = 16
        self.hidden_dim = 32
        self.output_dim = 3
        self.num_layers = 2
        self.dropout_rate = 0.2

        self.model = BiLSTMClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
        )

    def test_model_architecture_attributes(self):
        self.assertEqual(self.model.embedding.num_embeddings, self.vocab_size)
        self.assertEqual(self.model.embedding.embedding_dim, self.embedding_dim)
        self.assertEqual(self.model.embedding.padding_idx, 0)
        self.assertTrue(self.model.lstm.bidirectional)
        self.assertEqual(self.model.lstm.hidden_size, self.hidden_dim)
        self.assertEqual(self.model.fc.in_features, self.hidden_dim * 2)
        self.assertEqual(self.model.fc.out_features, self.output_dim)

    def test_forward_pass_shape_and_values(self):
        batch_size = 4
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        lengths = torch.tensor([10, 8, 5, 2], dtype=torch.long)

        logits = self.model(input_ids, lengths)
        self.assertEqual(logits.shape, torch.Size([batch_size, self.output_dim]))

        # Kiểm tra không bị giá trị NaN hoặc Inf
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_predict_proba_and_predict(self):
        batch_size = 3
        seq_len = 8
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        lengths = torch.tensor([8, 6, 4], dtype=torch.long)

        probs = self.model.predict_proba(input_ids, lengths)
        self.assertEqual(probs.shape, torch.Size([batch_size, 3]))

        # Tổng xác suất mỗi hàng phải xấp xỉ 1.0
        row_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

        preds = self.model.predict(input_ids, lengths)
        self.assertEqual(preds.shape, torch.Size([batch_size]))
        for p in preds.tolist():
            self.assertIn(p, [0, 1, 2])

    def test_backward_pass_and_optimization(self):
        batch_size = 2
        seq_len = 6
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        lengths = torch.tensor([6, 3], dtype=torch.long)
        labels = torch.tensor([0, 2], dtype=torch.long)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        logits = self.model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()

        # Kiểm tra gradients được tính toán đầy đủ cho các tầng
        self.assertIsNotNone(self.model.embedding.weight.grad)
        self.assertIsNotNone(self.model.fc.weight.grad)
        optimizer.step()


class TestClassWeightsCalculation(unittest.TestCase):
    def test_weight_formula_exactness(self):
        # Giả lập dữ liệu huấn luyện:
        # Tổng N = 1000 mẫu, 3 lớp: NEG=200, NEU=150, POS=650
        # w_c = N / (K * N_c)
        # w_NEG = 1000 / (3 * 200) = 1.6667
        # w_NEU = 1000 / (3 * 150) = 2.2222
        # w_POS = 1000 / (3 * 650) = 0.5128
        labels = [0] * 200 + [1] * 150 + [2] * 650
        weights = compute_class_weights(labels, num_classes=3)

        self.assertEqual(weights.shape, torch.Size([3]))
        expected_neg = 1000.0 / (3.0 * 200.0)
        expected_neu = 1000.0 / (3.0 * 150.0)
        expected_pos = 1000.0 / (3.0 * 650.0)

        self.assertAlmostEqual(weights[0].item(), expected_neg, places=3)
        self.assertAlmostEqual(weights[1].item(), expected_neu, places=3)
        self.assertAlmostEqual(weights[2].item(), expected_pos, places=3)

    def test_weighted_criterion_usage(self):
        weights = torch.tensor([1.57, 2.22, 0.52], dtype=torch.float)
        criterion = get_weighted_criterion(weights)

        logits = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float)
        label = torch.tensor([0], dtype=torch.long)
        loss = criterion(logits, label)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
