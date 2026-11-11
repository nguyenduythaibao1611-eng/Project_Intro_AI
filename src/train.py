"""
Script Huấn luyện Chuẩn hóa có Checkpointing & Weighted Loss (Training Pipeline)
==============================================================================
Dự án: Vietnamese Review Sentiment Analysis (Shopee Reviews)
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách chính: Từ Văn Huy Hoàng & Châu Tuấn Kiệt

Quy trình thực thi:
1. Nạp dữ liệu phân tầng Train / Val / Test từ data/processed/
2. Tính trọng số phân lớp nghịch đảo tỷ lệ w_c = N / (K * N_c) cho Weighted Cross-Entropy Loss (RQ3)
3. Huấn luyện mô hình (Bi-LSTM hoặc PhoBERT v2) có điều chỉnh siêu tham số và Linear Warmup
4. Giám sát Macro F1 trên tập Validation và lưu checkpoint tối ưu (models/best_*.pt)
5. Đánh giá kiểm thử độc lập trên tập Test (RQ2) và xuất báo cáo chỉ số
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Cấu hình UTF-8 an toàn cho Windows Console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.dataset import (
    MAX_SEQUENCE_LENGTH,
    RANDOM_SEED,
    LABEL_TO_ID,
    ID_TO_LABEL,
    Vocabulary,
    SentimentDataset,
    build_vocab_and_datasets,
    create_data_loader,
)
from src.evaluate import (
    compute_classification_metrics,
    plot_and_save_confusion_matrix,
    analyze_error_samples,
    print_metrics_summary,
)
from src.models import (
    BiLSTMClassifier,
    PhoBERTClassifier,
    compute_class_weights,
    get_weighted_criterion,
    get_phobert_optimizer_and_scheduler,
    train_one_epoch,
    evaluate_model,
    save_checkpoint,
    load_checkpoint,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Thiết lập hạt giống ngẫu nhiên đảm bảo tính tái lập 100%."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PhoBERTReviewDataset(Dataset):
    """Dataset cho mô hình Transformer VinAI PhoBERT v2."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        self.texts = [str(t) if t is not None and not pd.isna(t) else "" for t in texts]
        self.labels = labels
        self.max_length = max_length

        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "text": text,
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["label"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            item["labels"] = item["label"]

        return item


def train_bilstm_pipeline(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    epochs: int = 5,
    batch_size: int = 64,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout_rate: float = 0.3,
    lr: float = 1e-3,
    device_name: Optional[str] = None,
    save_checkpoint_name: str = "best_bilstm.pt",
) -> Dict[str, Any]:
    """Huấn luyện mạng hồi quy hai chiều Bi-LSTM với hàm Weighted Loss."""
    set_seed(RANDOM_SEED)

    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n=======================================================")
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN BI-LSTM TRÊN THIẾT BỊ: {device}")
    print(f"=======================================================")

    # 1. Nạp dữ liệu
    df_train = pd.read_csv(Path(data_dir) / "train.csv")
    df_val = pd.read_csv(Path(data_dir) / "val.csv")
    df_test = pd.read_csv(Path(data_dir) / "test.csv")

    # 2. Xây dựng Vocabulary (chỉ từ tập train tránh rò rỉ)
    vocab, train_ds, val_ds, test_ds = build_vocab_and_datasets(
        df_train, df_val, df_test, text_column="processed_content", label_column="label_id"
    )
    vocab_path = Path(models_dir) / "vocab.json"
    vocab.save(vocab_path)
    print(f" • Đã lưu bộ từ điển Vocabulary ({len(vocab)} từ vựng) vào: {vocab_path}")

    train_loader = create_data_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_data_loader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = create_data_loader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Tính trọng số phân lớp nghịch đảo tỷ lệ w_c = N / (K * N_c) cho RQ3
    class_weights = compute_class_weights(df_train["label_id"].values, num_classes=3, device=device)
    print(f" • Trọng số Weighted Loss cho RQ3: {class_weights.cpu().numpy().tolist()}")
    criterion = get_weighted_criterion(class_weights, device=device)

    # 4. Khởi tạo mô hình Bi-LSTM
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=3,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_checkpoint_path = Path(models_dir) / save_checkpoint_name

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_grad=1.0,
        )

        val_loss, val_acc, y_val_true, y_val_pred = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        val_metrics = compute_classification_metrics(y_val_true, y_val_pred)
        val_macro_f1 = float(val_metrics["macro_f1"])
        elapsed = time.time() - t0

        print(
            f" Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_macro_f1*100:.2f}%"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            extra_state = {
                "model_type": "bilstm",
                "vocab_size": len(vocab),
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout_rate": dropout_rate,
                "class_weights": class_weights.cpu().tolist(),
            }
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_macro_f1": val_macro_f1, "val_acc": val_acc},
                save_path=best_checkpoint_path,
                extra_state=extra_state,
            )
            print(f"   --> 🌟 Checkpoint tối ưu mới được lưu: Val Macro F1 = {val_macro_f1*100:.2f}%")

    # 5. Đánh giá cuối kỳ trên tập Test
    print(f"\n📊 Đang nạp checkpoint tối ưu từ: {best_checkpoint_path} để kiểm thử...")
    load_checkpoint(model, best_checkpoint_path, device=device)

    test_loss, test_acc, y_test_true, y_test_pred = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    test_metrics = compute_classification_metrics(y_test_true, y_test_pred)
    print_metrics_summary(test_metrics)

    return {
        "model": model,
        "best_checkpoint_path": str(best_checkpoint_path),
        "test_metrics": test_metrics,
    }


def train_phobert_pipeline(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    epochs: int = 3,
    batch_size: int = 32,
    lr_backbone: float = 2e-5,
    lr_head: float = 1e-3,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    device_name: Optional[str] = None,
    save_checkpoint_name: str = "best_phobert.pt",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Fine-tuning mô hình Transformer VinAI PhoBERT v2 trên GPU."""
    set_seed(RANDOM_SEED)

    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n=======================================================")
    print(f"🤖 BẮT ĐẦU FINE-TUNING PHOBERT V2 TRÊN THIẾT BỊ: {device}")
    print(f"=======================================================")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    # 1. Nạp dữ liệu
    df_train = pd.read_csv(Path(data_dir) / "train.csv")
    df_val = pd.read_csv(Path(data_dir) / "val.csv")
    df_test = pd.read_csv(Path(data_dir) / "test.csv")

    if max_train_samples is not None and len(df_train) > max_train_samples:
        df_train = df_train.sample(n=max_train_samples, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f" • Đã lấy mẫu ngẫu nhiên {max_train_samples} dòng tập Train.")
    if max_val_samples is not None and len(df_val) > max_val_samples:
        df_val = df_val.sample(n=max_val_samples, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f" • Đã lấy mẫu ngẫu nhiên {max_val_samples} dòng tập Val.")
    if max_val_samples is not None and len(df_test) > max_val_samples:
        df_test = df_test.sample(n=max_val_samples, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f" • Đã lấy mẫu ngẫu nhiên {max_val_samples} dòng tập Test.")

    train_ds = PhoBERTReviewDataset(
        df_train["processed_content"].tolist(),
        df_train["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    val_ds = PhoBERTReviewDataset(
        df_val["processed_content"].tolist(),
        df_val["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    test_ds = PhoBERTReviewDataset(
        df_test["processed_content"].tolist(),
        df_test["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Trọng số Weighted Loss cho RQ3
    class_weights = compute_class_weights(df_train["label_id"].values, num_classes=3, device=device)
    print(f" • Trọng số Weighted Loss cho RQ3: {class_weights.cpu().numpy().tolist()}")
    criterion = get_weighted_criterion(class_weights, device=device)

    # 3. Khởi tạo mô hình PhoBERTClassifier
    model = PhoBERTClassifier(
        model_name="vinai/phobert-base-v2",
        num_classes=3,
        dropout_rate=0.2,
    ).to(device)

    total_steps = len(train_loader) * epochs
    optimizer, scheduler = get_phobert_optimizer_and_scheduler(
        model=model,
        total_steps=total_steps,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
    )

    best_val_f1 = -1.0
    best_checkpoint_path = Path(models_dir) / save_checkpoint_name

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_grad=1.0,
            scheduler=scheduler,
        )

        val_loss, val_acc, y_val_true, y_val_pred = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        val_metrics = compute_classification_metrics(y_val_true, y_val_pred)
        val_macro_f1 = float(val_metrics["macro_f1"])
        elapsed = time.time() - t0

        print(
            f" Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_macro_f1*100:.2f}%"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            extra_state = {
                "model_type": "phobert",
                "model_name": "vinai/phobert-base-v2",
                "num_classes": 3,
                "dropout_rate": 0.2,
                "class_weights": class_weights.cpu().tolist(),
            }
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_macro_f1": val_macro_f1, "val_acc": val_acc},
                save_path=best_checkpoint_path,
                extra_state=extra_state,
            )
            print(f"   --> 🌟 Checkpoint tối ưu mới được lưu: Val Macro F1 = {val_macro_f1*100:.2f}%")

    print(f"\n📊 Đang nạp checkpoint tối ưu từ: {best_checkpoint_path} để kiểm thử...")
    load_checkpoint(model, best_checkpoint_path, device=device)

    test_loss, test_acc, y_test_true, y_test_pred = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    test_metrics = compute_classification_metrics(y_test_true, y_test_pred)
    print_metrics_summary(test_metrics)

    return {
        "model": model,
        "best_checkpoint_path": str(best_checkpoint_path),
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Script Huấn luyện Mô hình Học sâu Shopee Reviews")
    parser.add_argument("--model-type", choices=["bilstm", "phobert"], default="bilstm", help="Loại mô hình")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng Epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (cho Bi-LSTM)")
    parser.add_argument("--lr-backbone", type=float, default=2e-5, help="Learning rate backbone (cho PhoBERT)")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate classification head (cho PhoBERT)")
    parser.add_argument("--device", type=str, default=None, help="cuda hoặc cpu")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Thư mục chứa dữ liệu")
    parser.add_argument("--models-dir", type=str, default="models", help="Thư mục lưu checkpoints")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Giới hạn số mẫu train")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Giới hạn số mẫu val/test")

    args = parser.parse_args()

    if args.model_type == "bilstm":
        train_bilstm_pipeline(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_name=args.device,
        )
    elif args.model_type == "phobert":
        train_phobert_pipeline(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            device_name=args.device,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )


if __name__ == "__main__":
    main()
