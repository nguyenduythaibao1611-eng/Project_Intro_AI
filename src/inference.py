"""
Module Suy diễn Độc lập Phân loại Cảm xúc (Inference Module)
=============================================================
Dự án: Vietnamese Review Sentiment Analysis (Shopee Reviews)
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách chính: Từ Văn Huy Hoàng & Châu Tuấn Kiệt

Cung cấp hàm dự đoán phục vụ Web Demo (app.py) và các dịch vụ tích hợp:
- predict_sentiment(text: str, model_type="phobert") -> Dict[str, Any]
- Hỗ trợ cả 2 mô hình: PhoBERT v2 và Bi-LSTM hai chiều.
- Đo lường chính xác thời gian suy diễn (latency_ms) với yêu cầu < 100ms/câu.
- Xử lý linh hoạt, an toàn tuyệt đối với mọi đầu vào đặc biệt: câu đơn, câu rỗng, ký tự lạ.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import (
    MAX_SEQUENCE_LENGTH,
    PAD_IDX,
    UNK_IDX,
    ID_TO_LABEL,
    LABEL_TO_ID,
    Vocabulary,
)
from src.preprocess import preprocess_pipeline

# Bộ nhớ đệm mô hình toàn cục (Model & Tokenizer Singleton Cache)
# Tránh nạp lại trọng số nhiều lần gây trễ thời gian suy diễn (Inference Latency)
_MODEL_CACHE: Dict[str, Any] = {}
_VOCAB_CACHE: Optional[Vocabulary] = None
_TOKENIZER_CACHE: Optional[Any] = None


def get_inference_device(device_name: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Xác định thiết bị tính toán tối ưu (Ưu tiên GPU CUDA nếu có)."""
    if device_name is not None:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_inference_model(
    model_type: str = "phobert",
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Nạp mô hình suy diễn vào bộ nhớ đệm (Singleton Loader).
    
    Args:
        model_type: "phobert" hoặc "bilstm".
        device: Thiết bị chạy (torch.device hoặc None).
        checkpoint_path: Đường dẫn tệp trọng số .pt tùy chọn.
        
    Returns:
        Mô hình PyTorch đã ở chế độ eval().
    """
    global _MODEL_CACHE, _VOCAB_CACHE, _TOKENIZER_CACHE
    dev = get_inference_device(device)
    cache_key = f"{model_type}_{dev}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if model_type == "phobert":
        from src.models import PhoBERTClassifier

        ckpt = checkpoint_path or "models/best_phobert.pt"
        model = PhoBERTClassifier(model_name="vinai/phobert-base-v2", num_classes=3).to(dev)

        if Path(ckpt).exists():
            checkpoint = torch.load(ckpt, map_location=dev, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)

        model.eval()
        _MODEL_CACHE[cache_key] = model
        return model

    elif model_type == "bilstm":
        from src.models import BiLSTMClassifier

        vocab_path = Path("models/vocab.json")
        if _VOCAB_CACHE is None:
            if vocab_path.exists():
                _VOCAB_CACHE = Vocabulary.load(vocab_path)
            else:
                train_csv = Path("data/processed/train.csv")
                if train_csv.exists():
                    import pandas as pd
                    df_tr = pd.read_csv(train_csv)
                    _VOCAB_CACHE = Vocabulary(min_freq=2)
                    _VOCAB_CACHE.build_vocab(df_tr["processed_content"].fillna("").astype(str).tolist())
                    _VOCAB_CACHE.save(vocab_path)
                else:
                    _VOCAB_CACHE = Vocabulary(min_freq=1)

        ckpt = checkpoint_path or "models/best_bilstm.pt"
        vocab_size = len(_VOCAB_CACHE)
        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            output_dim=3,
            num_layers=2,
            dropout_rate=0.3,
        ).to(dev)

        if Path(ckpt).exists():
            checkpoint = torch.load(ckpt, map_location=dev, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)

        model.eval()
        _MODEL_CACHE[cache_key] = model
        return model
    else:
        raise ValueError(f"Không hỗ trợ model_type: '{model_type}'. Chỉ hỗ trợ 'phobert' hoặc 'bilstm'.")


def get_phobert_tokenizer():
    """Nạp và lưu đệm Tokenizer cho PhoBERT."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    return _TOKENIZER_CACHE


def get_bilstm_vocabulary() -> Vocabulary:
    """Nạp và lưu đệm bộ từ điển Vocabulary cho Bi-LSTM."""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        vocab_path = Path("models/vocab.json")
        if vocab_path.exists():
            _VOCAB_CACHE = Vocabulary.load(vocab_path)
        else:
            train_csv = Path("data/processed/train.csv")
            if train_csv.exists():
                import pandas as pd
                df_tr = pd.read_csv(train_csv)
                _VOCAB_CACHE = Vocabulary(min_freq=2)
                _VOCAB_CACHE.build_vocab(df_tr["processed_content"].fillna("").astype(str).tolist())
                _VOCAB_CACHE.save(vocab_path)
            else:
                _VOCAB_CACHE = Vocabulary(min_freq=1)
    return _VOCAB_CACHE


def predict_sentiment(
    text: str,
    model_type: str = "phobert",
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Suy diễn nhãn cảm xúc và phân phối xác suất cho chuỗi đánh giá tiếng Việt.
    
    Yêu cầu kỹ thuật & Chuẩn hóa:
    - Trả về dictionary gồm: 'label' (POS/NEU/NEG), 'confidence', 'latency_ms'.
    - Kèm theo: 'probabilities' (phần trăm từng nhãn) và 'processed_text'.
    - Đảm bảo thời gian suy diễn < 100ms/câu.
    - Xử lý an toàn với mọi câu đơn và câu rỗng (empty string/whitespace).

    Args:
        text: Văn bản thô của bình luận cần đánh giá.
        model_type: Loại mô hình ("phobert" hoặc "bilstm"). Mặc định: "phobert".
        device: Thiết bị ("cuda", "cpu" hoặc None để tự động chọn).
        checkpoint_path: Đường dẫn tệp weights .pt tùy biến.

    Returns:
        Dict[str, Any]:
            {
                "label": "POS" | "NEU" | "NEG",
                "confidence": float,
                "latency_ms": float,
                "probabilities": {"NEG": float, "NEU": float, "POS": float},
                "model_type": str,
                "processed_text": str
            }
    """
    start_time = time.perf_counter()
    dev = get_inference_device(device)

    # 1. Tiền xử lý văn bản theo chuẩn pipeline tiếng Việt TMĐT
    if text is None or not isinstance(text, str) or text.strip() == "":
        cleaned_text = ""
    else:
        cleaned_text = preprocess_pipeline(text)

    # 2. Suy diễn theo loại mô hình
    if model_type == "phobert":
        model = load_inference_model("phobert", device=dev, checkpoint_path=checkpoint_path)
        tokenizer = get_phobert_tokenizer()

        # Tokenize (hỗ trợ câu rỗng thành <s> </s>, dùng dynamic length để tối ưu hóa < 50ms)
        token_input = cleaned_text if cleaned_text else "<s> </s>"
        encoded = tokenizer(
            token_input,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(dev)
        attention_mask = encoded["attention_mask"].to(dev)

        with torch.no_grad():
            probs_tensor = model.predict_proba(input_ids, attention_mask=attention_mask)[0]

    elif model_type == "bilstm":
        model = load_inference_model("bilstm", device=dev, checkpoint_path=checkpoint_path)
        vocab = get_bilstm_vocabulary()

        tokens = cleaned_text.split() if cleaned_text else []
        if len(tokens) == 0:
            token_ids = [UNK_IDX]
            length = 1
        else:
            token_ids = [vocab[t] for t in tokens[:MAX_SEQUENCE_LENGTH]]
            length = len(token_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=dev)
        lengths = torch.tensor([length], dtype=torch.long, device=dev)

        with torch.no_grad():
            probs_tensor = model.predict_proba(input_ids, lengths=lengths)[0]
    else:
        raise ValueError(f"Loại mô hình không được hỗ trợ: '{model_type}'")

    probs = probs_tensor.cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = ID_TO_LABEL.get(pred_idx, "NEU")
    confidence = float(probs[pred_idx])

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "label": pred_label,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
        "probabilities": {
            "NEG": round(float(probs[0]), 4),
            "NEU": round(float(probs[1]), 4),
            "POS": round(float(probs[2]), 4),
        },
        "model_type": model_type,
        "processed_text": cleaned_text,
    }


def clear_inference_cache() -> None:
    """Xóa bộ nhớ đệm phục vụ reload hoặc kiểm thử unit test."""
    global _MODEL_CACHE, _VOCAB_CACHE, _TOKENIZER_CACHE
    _MODEL_CACHE.clear()
    _VOCAB_CACHE = None
    _TOKENIZER_CACHE = None
