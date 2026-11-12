"""
Module Cấu trúc Dữ liệu & Xây dựng Từ điển (Dataset & Vocabulary Builder)
========================================================================
Dự án: Vietnamese Review Sentiment Analysis (Shopee Reviews)
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách chính: Châu Tuấn Kiệt — Senior ML Engineer

Tệp mã nguồn cung cấp các cấu trúc dữ liệu chuẩn hóa cho mạng học sâu:
1. Vocabulary: Quản lý bộ từ vựng, ánh xạ token <-> ID với các token đặc biệt <PAD> (0), <UNK> (1).
2. SentimentDataset: Kế thừa torch.utils.data.Dataset, hỗ trợ cắt ngắn/đệm chuỗi về MAX_SEQUENCE_LENGTH = 64.
3. collate_sentiment_batch: Hàm gom lô (collate_fn) tối ưu hóa cho PyTorch DataLoader và pack_padded_sequence.
4. build_vocab_and_datasets: Hàm tiện ích khởi tạo toàn bộ dataset từ DataFrames/CSVs.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Các hằng số toàn cục theo đặc tả hệ thống (docs/TASK_SPECIFICATION_AND_SYSTEM_ARCHITECTURE.md)
PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
PAD_IDX: int = 0
UNK_IDX: int = 1
MAX_SEQUENCE_LENGTH: int = 64
RANDOM_SEED: int = 42

LABEL_TO_ID: Dict[str, int] = {"NEG": 0, "NEU": 1, "POS": 2}
ID_TO_LABEL: Dict[int, str] = {0: "NEG", 1: "NEU", 2: "POS"}


class Vocabulary:
    """
    Xây dựng và quản lý từ điển ánh xạ token -> ID cho mô hình Bi-LSTM.

    Attributes:
        min_freq (int): Tần suất xuất hiện tối thiểu để giữ lại token.
        stoi (Dict[str, int]): Ánh xạ từ chuỗi từ (string) sang chỉ số nguyên (index).
        itos (Dict[int, str]): Ánh xạ từ chỉ số nguyên sang chuỗi từ.
    """

    def __init__(self, min_freq: int = 2) -> None:
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.itos: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self._frequencies: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self.stoi)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        """Trả về ID của token; nếu không có trong từ điển thì trả về UNK_IDX (1)."""
        return self.stoi.get(token, UNK_IDX)

    def lookup_token(self, idx: int) -> str:
        """Trả về token tương ứng với ID; nếu không có thì trả về UNK_TOKEN."""
        return self.itos.get(idx, UNK_TOKEN)

    def lookup_indices(self, tokens: Sequence[str]) -> List[int]:
        """Chuyển đổi danh sách tokens thành danh sách ID."""
        return [self[token] for token in tokens]

    def lookup_tokens(self, indices: Sequence[int]) -> List[str]:
        """Chuyển đổi danh sách ID thành danh sách tokens."""
        return [self.lookup_token(idx) for idx in indices]

    def build_vocab(self, sentence_list: Iterable[Union[str, List[str]]]) -> None:
        """
        Học từ vựng từ danh sách các câu đã được tách từ (Word Segmentation).
        
        Args:
            sentence_list: Tập hợp các câu văn (chuỗi ký tự phân tách bằng khoảng trắng)
                           hoặc danh sách tokens.
        """
        self._frequencies.clear()
        for sentence in sentence_list:
            if sentence is None or (isinstance(sentence, float) and pd.isna(sentence)):
                continue
            if isinstance(sentence, str):
                tokens = sentence.strip().split()
            elif isinstance(sentence, (list, tuple)):
                tokens = [str(t) for t in sentence]
            else:
                tokens = str(sentence).strip().split()
            self._frequencies.update(tokens)

        # Đặt lại từ điển chỉ chứa các token đặc biệt
        self.stoi = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.itos = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

        # Sắp xếp các từ theo tần suất giảm dần; nếu bằng tần suất thì xếp theo thứ tự bảng chữ cái
        # để đảm bảo tính tái lập 100% (deterministic reproducibility)
        sorted_tokens = sorted(
            self._frequencies.items(),
            key=lambda item: (-item[1], item[0])
        )

        current_idx = 2
        for token, freq in sorted_tokens:
            if freq >= self.min_freq:
                if token not in self.stoi:
                    self.stoi[token] = current_idx
                    self.itos[current_idx] = token
                    current_idx += 1

    def to_dict(self) -> Dict[str, Any]:
        """Xuất thông tin từ điển dưới dạng Dictionary có thể tuần tự hóa JSON."""
        return {
            "min_freq": self.min_freq,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "vocab_size": len(self.stoi),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Vocabulary:
        """Khôi phục đối tượng Vocabulary từ Dictionary."""
        vocab = cls(min_freq=data.get("min_freq", 2))
        vocab.stoi = dict(data["stoi"])
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        return vocab

    def save(self, file_path: Union[str, Path]) -> None:
        """Lưu từ điển ra tệp tin JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> Vocabulary:
        """Nạp từ điển từ tệp tin JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp từ điển tại: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class SentimentDataset(Dataset):
    """
    Dataset PyTorch dành cho bài toán phân loại cảm xúc với mô hình Bi-LSTM.
    
    Xử lý:
    - Nhận vào văn bản đã qua tiền xử lý tách từ tiếng Việt.
    - Ánh xạ tokens sang IDs thông qua Vocabulary.
    - Cắt ngắn (Truncation) hoặc đệm (Padding) về độ dài cố định `max_length = 64`.
    - Trả về Dictionary gồm: 'input_ids', 'label' (nếu có), 'length', 'text'.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[Union[int, str]]] = None,
        vocab: Optional[Vocabulary] = None,
        max_length: int = MAX_SEQUENCE_LENGTH,
        pad_idx: int = PAD_IDX,
        unk_idx: int = UNK_IDX,
    ) -> None:
        self.texts = list(texts)
        self.vocab = vocab
        self.max_length = max_length
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.labels: Optional[List[int]] = None
        if labels is not None:
            processed_labels: List[int] = []
            for lbl in labels:
                if isinstance(lbl, str):
                    if lbl in LABEL_TO_ID:
                        processed_labels.append(LABEL_TO_ID[lbl])
                    else:
                        raise ValueError(f"Nhãn không hợp lệ: '{lbl}'. Phải thuộc {list(LABEL_TO_ID.keys())}")
                elif isinstance(lbl, (int, torch.Tensor)):
                    processed_labels.append(int(lbl))
                elif pd.isna(lbl):
                    processed_labels.append(0)
                else:
                    processed_labels.append(int(lbl))
            self.labels = processed_labels
            if len(self.labels) != len(self.texts):
                raise ValueError("Số lượng nhãn không khớp với số lượng mẫu văn bản!")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_text = self.texts[idx]
        if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text)):
            text = ""
        else:
            text = str(raw_text).strip()

        tokens = text.split()
        raw_length = len(tokens)

        # Tính toán độ dài thực tế hữu hiệu (tối thiểu là 1 để an toàn cho pack_padded_sequence)
        effective_length = min(max(raw_length, 1), self.max_length)

        # Chuyển đổi tokens sang ID
        if self.vocab is not None:
            token_ids = [self.vocab[t] for t in tokens[:self.max_length]]
        else:
            token_ids = [self.unk_idx for _ in tokens[:self.max_length]]

        # Xử lý câu rỗng: ít nhất 1 token UNK để không bị length = 0 trong RNN
        if len(token_ids) == 0:
            token_ids = [self.unk_idx]
            effective_length = 1

        # Đệm chuỗi (Padding) về max_length
        padded_ids = token_ids + [self.pad_idx] * (self.max_length - len(token_ids))

        item: Dict[str, Any] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "length": int(effective_length),
            "text": text,
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def collate_sentiment_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Hàm gom lô (collate function) cho DataLoader.
    
    Đầu ra là Dictionary gồm:
    - 'input_ids': torch.Tensor kích thước [batch_size, max_length]
    - 'lengths': torch.Tensor kích thước [batch_size]
    - 'labels': torch.Tensor kích thước [batch_size] (nếu batch có chứa 'label')
    - 'texts': List[str] nội dung văn bản gốc
    """
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    texts = [item["text"] for item in batch]

    output: Dict[str, Any] = {
        "input_ids": input_ids,
        "lengths": lengths,
        "texts": texts,
    }

    if "label" in batch[0]:
        labels = torch.tensor([item["label"].item() for item in batch], dtype=torch.long)
        output["label"] = labels
        output["labels"] = labels  # Tiện ích alias cho code huấn luyện

    return output


def create_data_loader(
    dataset: SentimentDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Tạo DataLoader tiêu chuẩn với hàm gom lô collate_sentiment_batch."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sentiment_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_vocab_and_datasets(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    text_column: str = "processed_content",
    label_column: str = "label_id",
    min_freq: int = 2,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> Tuple[Vocabulary, SentimentDataset, Optional[SentimentDataset], Optional[SentimentDataset]]:
    """
    Xây dựng từ điển chỉ từ tập Train (tránh rò rỉ dữ liệu - Data Leakage),
    sau đó tạo các bộ Dataset cho Train, Validation và Test.
    """
    # Xây dựng từ điển chỉ từ tập Train
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(train_df[text_column].fillna("").astype(str).tolist())

    train_labels = train_df[label_column].tolist() if label_column in train_df.columns else None
    train_dataset = SentimentDataset(
        texts=train_df[text_column].fillna("").astype(str).tolist(),
        labels=train_labels,
        vocab=vocab,
        max_length=max_length,
    )

    val_dataset: Optional[SentimentDataset] = None
    if val_df is not None:
        val_labels = val_df[label_column].tolist() if label_column in val_df.columns else None
        val_dataset = SentimentDataset(
            texts=val_df[text_column].fillna("").astype(str).tolist(),
            labels=val_labels,
            vocab=vocab,
            max_length=max_length,
        )

    test_dataset: Optional[SentimentDataset] = None
    if test_df is not None:
        test_labels = test_df[label_column].tolist() if label_column in test_df.columns else None
        test_dataset = SentimentDataset(
            texts=test_df[text_column].fillna("").astype(str).tolist(),
            labels=test_labels,
            vocab=vocab,
            max_length=max_length,
        )

    return vocab, train_dataset, val_dataset, test_dataset
