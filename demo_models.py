"""Strict checkpoint loading for the web, without editing teammates' modules.

Only local tensor checkpoints/config/tokenizers are used. A missing or incompatible
checkpoint is an error; an initialized model is never returned as a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = {"baseline": "Baseline · LinearSVC", "phobert": "PhoBERT v2", "bilstm": "BiLSTM"}
CHECKPOINT_NAMES = {"phobert": "best_phobert.pt", "bilstm": "best_bilstm.pt"}
LABELS = {0: "NEG", 1: "NEU", 2: "POS"}


def checkpoint_signature(model_type: str) -> tuple[str, int, int]:
    if model_type not in CHECKPOINT_NAMES:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_type}")
    path = ROOT / "models" / CHECKPOINT_NAMES[model_type]
    info = path.stat()  # Missing checkpoints must fail before a model is created.
    if info.st_size == 0:
        raise ValueError(f"Checkpoint rỗng: {path.name}")
    return str(path.resolve()), info.st_size, info.st_mtime_ns


def available_models() -> list[str]:
    result = ["baseline"]
    for name in CHECKPOINT_NAMES:
        try:
            checkpoint_signature(name)
            result.append(name)
        except (OSError, ValueError):
            pass
    return result


@dataclass
class NeuralRuntime:
    model_type: str
    model: Any
    device: Any
    metadata: dict
    tokenizer: Any = None
    vocab: Any = None
    lock: Any = field(default_factory=RLock)


def load_neural_runtime(model_type: str, signature: tuple[str, int, int], device_name: str | None = None) -> NeuralRuntime:
    import torch

    path = Path(signature[0])
    info = path.stat()
    if (str(path.resolve()), info.st_size, info.st_mtime_ns) != signature:
        raise ValueError("Checkpoint đã thay đổi. Hãy tải lại trang để nạp đúng phiên bản.")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint không chứa model_state_dict hợp lệ.")
    if payload.get("model_type") != model_type:
        raise ValueError("Loại mô hình trong checkpoint không khớp mô hình đã chọn.")
    state = payload["model_state_dict"]
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    metadata = {key: payload.get(key) for key in ("epoch", "metrics", "model_type", "model_name")}
    metadata["checkpoint"] = path.name
    if model_type == "bilstm":
        from src.dataset import Vocabulary
        from src.models import BiLSTMClassifier

        vocab = Vocabulary.load(ROOT / "models/vocab.json")
        if len(vocab) != int(payload["vocab_size"]) or state["embedding.weight"].shape[0] != len(vocab):
            raise ValueError("Vocabulary hiện tại không khớp kích thước checkpoint BiLSTM.")
        model = BiLSTMClassifier(
            vocab_size=len(vocab), embedding_dim=int(payload["embedding_dim"]),
            hidden_dim=int(payload["hidden_dim"]), output_dim=3,
            num_layers=int(payload["num_layers"]), dropout_rate=float(payload["dropout_rate"]),
        )
        model.load_state_dict(state, strict=True)
        runtime = NeuralRuntime(model_type, model.to(device).eval(), device, metadata, vocab=vocab)
    elif model_type == "phobert":
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from src.models import PhoBERTClassifier

        assets = ROOT / "models/phobert_assets"
        config = AutoConfig.from_pretrained(assets, local_files_only=True)
        if int(payload.get("num_classes", 0)) != 3:
            raise ValueError("Checkpoint PhoBERT phải có đúng ba lớp NEG/NEU/POS.")
        if tuple(state["phobert.embeddings.word_embeddings.weight"].shape) != (config.vocab_size, config.hidden_size):
            raise ValueError("Cấu hình PhoBERT không khớp trọng số checkpoint.")
        dropout_rate = float(payload.get("dropout_rate", 0.2))

        class CheckpointPhoBERT(PhoBERTClassifier):
            def __init__(self):
                # Reuse Hoàng's forward/predict methods; initialize the architecture
                # from local config, then require ALL learned weights below.
                torch.nn.Module.__init__(self)
                self.model_name = "vinai/phobert-base-v2"
                self.num_classes = 3
                self.dropout_rate = dropout_rate
                self.phobert = AutoModel.from_config(config)
                self.dropout = torch.nn.Dropout(self.dropout_rate)
                self.classifier = torch.nn.Linear(config.hidden_size, 3)

        model = CheckpointPhoBERT()
        model.load_state_dict(state, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
        runtime = NeuralRuntime(model_type, model.to(device).eval(), device, metadata, tokenizer=tokenizer)
    else:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_type}")
    # Warm up the loaded checkpoint, not an untrained model.
    predict_neural_cleaned(runtime, ["sản_phẩm tốt"])
    return runtime


def predict_neural_cleaned(runtime: NeuralRuntime, texts: list[str]) -> tuple[list[str], np.ndarray]:
    import torch

    if not texts:
        return [], np.empty((0, 3))
    probabilities = []
    with runtime.lock, torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            if runtime.model_type == "phobert":
                encoded = runtime.tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
                probs = runtime.model.predict_proba(
                    encoded["input_ids"].to(runtime.device),
                    attention_mask=encoded["attention_mask"].to(runtime.device),
                    token_type_ids=encoded.get("token_type_ids").to(runtime.device) if "token_type_ids" in encoded else None,
                )
            else:
                sequences = [[runtime.vocab[token] for token in text.split()[:64]] or [1] for text in batch]
                lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
                width = int(lengths.max())
                ids = torch.tensor([sequence + [0] * (width - len(sequence)) for sequence in sequences],
                                   dtype=torch.long, device=runtime.device)
                probs = runtime.model.predict_proba(ids, lengths=lengths)
            probabilities.append(probs.detach().cpu().numpy())
    scores = np.vstack(probabilities)
    if not np.isfinite(scores).all() or scores.shape != (len(texts), 3):
        raise ValueError("Mô hình trả kết quả không hợp lệ.")
    return [LABELS[int(index)] for index in scores.argmax(axis=1)], scores
