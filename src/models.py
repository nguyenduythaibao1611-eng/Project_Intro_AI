"""
Module Kiến trúc Mô hình Học sâu (Deep Learning Architectures)
==============================================================
Dự án: Vietnamese Review Sentiment Analysis (Shopee Reviews)
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách chính: Châu Tuấn Kiệt — Senior ML Engineer

Tệp mã nguồn cung cấp các kiến trúc mạng nơ-ron:
1. BiLSTMClassifier: Mạng nơ-ron hồi quy hai chiều 2 lớp (Bidirectional LSTM) với tầng nhúng Embedding,
   cơ chế đóng gói chuỗi (Pack Padded Sequence) và liên kết hidden states 2 chiều.
2. compute_class_weights & get_weighted_criterion: Tính trọng số nghịch đảo tần suất lớp (w_c = N / (K * N_c))
   để phục vụ Weighted Cross-Entropy Loss giải quyết câu hỏi nghiên cứu RQ3.
3. PhoBERTClassifier: Kiến trúc mô hình Transformer tinh chỉnh (Fine-tuning) dựa trên VinAI PhoBERT.
4. get_phobert_optimizer_and_scheduler: Khởi tạo tối ưu hóa phân tầng (AdamW) & Linear Warmup Scheduler.
5. Các hàm tiện ích huấn luyện (train_one_epoch, evaluate_model, save_checkpoint, load_checkpoint).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Hằng số toàn cục
RANDOM_SEED: int = 42
LABEL_TO_ID: Dict[str, int] = {"NEG": 0, "NEU": 1, "POS": 2}
ID_TO_LABEL: Dict[int, str] = {0: "NEG", 1: "NEU", 2: "POS"}


def compute_class_weights(
    labels: Union[Sequence[int], np.ndarray, torch.Tensor, Dict[int, int]],
    num_classes: int = 3,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Tính toán trọng số phạt nghịch đảo tỷ lệ theo công thức chuẩn trong đặc tả:
    
        w_c = N / (K * N_c)
        
    Trong đó:
    - N: Tổng số mẫu của tập dữ liệu
    - K: Số lớp (3 lớp: NEG, NEU, POS)
    - N_c: Số mẫu thuộc lớp c
    
    Returns:
        torch.Tensor: Vector trọng số kích thước [num_classes] với dtype float.
    """
    if isinstance(labels, dict):
        counts = [labels.get(c, 0) for c in range(num_classes)]
        total_samples = sum(counts)
    else:
        if isinstance(labels, torch.Tensor):
            arr = labels.detach().cpu().numpy()
        elif isinstance(labels, (list, tuple)):
            arr = np.array(labels)
        else:
            arr = np.asarray(labels)

        total_samples = len(arr)
        counts = [int(np.sum(arr == c)) for c in range(num_classes)]

    weights: List[float] = []
    for c in range(num_classes):
        n_c = counts[c]
        if n_c > 0 and total_samples > 0:
            w = total_samples / (num_classes * n_c)
        else:
            w = 1.0
        weights.append(float(w))

    weight_tensor = torch.tensor(weights, dtype=torch.float)
    if device is not None:
        weight_tensor = weight_tensor.to(device)

    return weight_tensor


def get_weighted_criterion(
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.CrossEntropyLoss:
    """Khởi tạo hàm mất mát CrossEntropyLoss có trọng số phạt (Weighted Loss) cho RQ3."""
    if class_weights is not None:
        if device is not None:
            class_weights = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()


class BiLSTMClassifier(nn.Module):
    """
    Kiến trúc mạng nơ-ron hồi quy hai chiều Bi-LSTM phân loại cảm xúc tiếng Việt.
    
    Thiết kế 3 tầng theo đặc tả Level 3:
    1. Embedding Layer: Nhúng từ vựng thành vector embedding_dim = 128 (padding_idx = 0).
    2. Bidirectional LSTM: 2 tầng (num_layers = 2), hidden_dim = 128 mỗi chiều, dropout = 0.3.
    3. Fully Connected Output: Ghép nối hidden state cuối của cả 2 chiều (Forward + Backward)
       tạo vector 256 chiều, chuyển đổi qua Linear sang 3 nhãn cảm xúc.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
        super(BiLSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # 1. Tầng nhúng từ vựng (Word Embedding)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # 2. Tầng LSTM 2 chiều (Bidirectional LSTM)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # 3. Dropout & Fully Connected Output Layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 vì ghép Forward và Backward

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Luồng tính toán Forward pass của mô hình Bi-LSTM.
        
        Args:
            input_ids: Tensor chỉ số token [batch_size, seq_len]
            lengths: Tensor độ dài thực tế của từng câu [batch_size]
            
        Returns:
            torch.Tensor: Logits dự đoán [batch_size, output_dim]
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(input_ids))

        # Sử dụng pack_padded_sequence nếu có cung cấp độ dài chuỗi
        if lengths is not None:
            # Đảm bảo lengths nằm trên CPU và không có giá trị < 1
            cpu_lengths = torch.clamp(lengths.cpu(), min=1)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                cpu_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, cell) = self.lstm(packed_embedded)
        else:
            _, (hidden, cell) = self.lstm(embedded)

        # Trích xuất hidden state cuối cùng của tầng trên cùng cho cả 2 chiều:
        # hidden shape: [num_layers * 2, batch_size, hidden_dim]
        # - hidden[-2]: Forward direction của layer cuối cùng
        # - hidden[-1]: Backward direction của layer cuối cùng
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]

        # Ghép nối 2 chiều: [batch_size, hidden_dim * 2]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Phân loại đầu ra
        logits = self.fc(self.dropout(hidden_concat))
        return logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính toán phân phối xác suất dự đoán (Softmax probabilities)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, lengths)
            probs = F.softmax(logits, dim=-1)
        return probs

    def predict(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dự đoán nhãn cảm xúc cụ thể (Argmax class ID)."""
        probs = self.predict_proba(input_ids, lengths)
        return torch.argmax(probs, dim=-1)


class PhoBERTClassifier(nn.Module):
    """
    Kiến trúc mô hình tinh chỉnh (Fine-tuning) Transformer VinAI PhoBERT.
    
    Sử dụng lớp biểu diễn ngữ cảnh tiền huấn luyện từ PhoBERT-base-v2,
    kết hợp tầng Pooling ([CLS] token hoặc Mean pooling) và Linear Classifier.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_classes: int = 3,
        dropout_rate: float = 0.2,
        freeze_bert: bool = False,
    ) -> None:
        super(PhoBERTClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        try:
            from transformers import AutoModel
            self.phobert = AutoModel.from_pretrained(model_name)
        except ImportError as exc:
            raise ImportError(
                "Để sử dụng PhoBERT, vui lòng cài đặt thư viện transformers: pip install transformers"
            ) from exc

        if freeze_bert:
            for param in self.phobert.parameters():
                param.requires_grad = False

        hidden_size = getattr(self.phobert.config, "hidden_size", 768)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.phobert(**kwargs)
        # Lấy biểu diễn tại token đầu tiên [CLS]
        if hasattr(outputs, "last_hidden_state"):
            cls_repr = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_repr = outputs.pooler_output
        else:
            cls_repr = outputs[0][:, 0, :]

        logits = self.classifier(self.dropout(cls_repr))
        return logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính toán phân phối xác suất dự đoán (Softmax probabilities)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            probs = F.softmax(logits, dim=-1)
        return probs

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dự đoán nhãn cảm xúc cụ thể (Argmax class ID)."""
        probs = self.predict_proba(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return torch.argmax(probs, dim=-1)


def get_phobert_optimizer_and_scheduler(
    model: PhoBERTClassifier,
    total_steps: int,
    lr_backbone: float = 2e-5,
    lr_head: float = 1e-4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    Thiết lập Optimizer AdamW với learning rate phân tầng và Linear Warmup Scheduler:
    - Backbone (PhoBERT): lr = 2e-5
    - Classification Head: lr = 1e-4 (tránh gradient explosion và mode collapse)
    - Linear Warmup Scheduler: 10% tổng số bước huấn luyện
    """
    try:
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError("Cần cài đặt transformers để sử dụng get_linear_schedule_with_warmup") from exc

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.phobert.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": lr_backbone,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.phobert.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": lr_backbone,
            "weight_decay": 0.0,
        },
        {
            "params": [p for p in model.classifier.parameters() if p.requires_grad],
            "lr": lr_head,
            "weight_decay": weight_decay,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _forward_batch(model: nn.Module, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Helper điều phối forward pass linh hoạt cho cả BiLSTM và PhoBERT."""
    input_ids = batch["input_ids"].to(device)
    if isinstance(model, PhoBERTClassifier) or "attention_mask" in batch:
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        return model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    elif "lengths" in batch:
        lengths = batch["lengths"].to(device)
        return model(input_ids, lengths)
    else:
        return model(input_ids)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 1.0,
    scheduler: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Huấn luyện mô hình qua 1 Epoch (tương thích cả Bi-LSTM và PhoBERT).
    
    Returns:
        Tuple[float, float]: (loss trung bình, accuracy trung bình)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        labels = (batch["label"] if "label" in batch else batch["labels"]).to(device)

        optimizer.zero_grad()
        logits = _forward_batch(model, batch, device)
        loss = criterion(logits, labels)

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Đánh giá mô hình trên tập Validation hoặc Test (tương thích cả Bi-LSTM và PhoBERT).
    
    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]:
            (loss trung bình, accuracy, mảng nhãn thực tế y_true, mảng dự đoán y_pred)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = (batch["label"] if "label" in batch else batch["labels"]).to(device)
            logits = _forward_batch(model, batch, device)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc, np.array(all_targets), np.array(all_preds)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    save_path: Union[str, Path] = "models/bilstm_sentiment.pt",
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Lưu trữ checkpoint trọng số mô hình."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra_state is not None:
        state.update(extra_state)
    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Nạp trọng số mô hình từ tệp checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
