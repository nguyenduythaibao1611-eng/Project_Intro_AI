"""Module Tiền xử lý & Tokenizer PhoBERT (vinai/phobert-base-v2).
==============================================================
Dự án: Vietnamese Review Sentiment Analysis (Shopee Reviews)
Học phần: AI Capstone 2025 - VNUK, Đại học Đà Nẵng
Phụ trách: Bùi Đình Hiếu (Data Engineer / Preprocessing Specialist)

Module cung cấp các hàm tiện ích chuẩn hóa:
1. get_phobert_tokenizer: Nạp và cache Singleton PhoBERT Tokenizer.
2. encode_reviews: Mã hóa danh sách câu sang PyTorch Tensor chuẩn độ dài 64 token.
3. batch_encode: Hỗ trợ mã hóa theo lô (mini-batches) phục vụ suy luận hàng loạt.
4. decode_tokens: Giải mã input_ids ngược lại chuỗi từ phục vụ visual inspection.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Iterator, List, Optional, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.preprocess import preprocess_pipeline

logger = logging.getLogger(__name__)

# Hằng số cấu hình hệ thống theo tài liệu đặc tả Level 3
DEFAULT_MODEL_NAME = "vinai/phobert-base-v2"
MAX_SEQUENCE_LENGTH = 64

_CACHED_TOKENIZER: Optional[PreTrainedTokenizerBase] = None


def get_phobert_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[str] = None
) -> PreTrainedTokenizerBase:
    """Nạp và lưu trữ Singleton instance của PhoBERT Tokenizer.

    Args:
        model_name: Tên hoặc đường dẫn trọng số trên HuggingFace (mặc định: 'vinai/phobert-base-v2').
        cache_dir: Thư mục lưu trữ cache cục bộ (tùy chọn).

    Returns:
        Instance của PreTrainedTokenizerBase.
    """
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is None:
        try:
            _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_fast=False
            )
        except Exception as exc:
            logger.warning(
                "Không thể nạp với use_fast=False (%s), thử lại với cấu hình mặc định...", exc
            )
            _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
    return _CACHED_TOKENIZER


def encode_reviews(
    texts: Union[str, Iterable[str]],
    max_len: int = MAX_SEQUENCE_LENGTH,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    apply_preprocessing: bool = False,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """Mã hóa văn bản review tiếng Việt thành PyTorch tensors tương thích PhoBERT.

    Args:
        texts: Một chuỗi đơn hoặc danh sách chuỗi đánh giá của người dùng.
        max_len: Độ dài chuỗi tối đa sau khi padding / truncation (mặc định: 64).
        tokenizer: Tokenizer tùy biến, nếu None sẽ dùng get_phobert_tokenizer().
        apply_preprocessing: Nếu True, sẽ tự động chạy qua preprocess_pipeline trước khi tokenize.
        return_tensors: Kiểu tensor trả về, mặc định 'pt' (PyTorch).

    Returns:
        Dict[str, torch.Tensor] chứa:
            - 'input_ids': torch.LongTensor hình dạng [batch_size, max_len]
            - 'attention_mask': torch.LongTensor hình dạng [batch_size, max_len]
    """
    if tokenizer is None:
        tokenizer = get_phobert_tokenizer()

    # Chuẩn hóa đầu vào thành danh sách chuỗi
    if isinstance(texts, str):
        text_list = [texts]
    else:
        text_list = [str(t) if t is not None else "" for t in texts]

    if not text_list:
        text_list = [""]

    # Áp dụng tiền xử lý nếu được yêu cầu
    if apply_preprocessing:
        text_list = [preprocess_pipeline(t) for t in text_list]

    encoded = tokenizer(
        text_list,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors=return_tensors
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }


def batch_encode(
    texts: List[str],
    batch_size: int = 32,
    max_len: int = MAX_SEQUENCE_LENGTH,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    apply_preprocessing: bool = False
) -> Iterator[Dict[str, torch.Tensor]]:
    """Bộ tạo (generator) sinh tensor theo từng lô phục vụ suy luận hàng loạt.

    Args:
        texts: Danh sách chuỗi văn bản.
        batch_size: Kích thước lô (mặc định: 32).
        max_len: Độ dài chuỗi tối đa (mặc định: 64).
        tokenizer: Tokenizer sử dụng.
        apply_preprocessing: Có áp dụng tiền xử lý trước không.

    Yields:
        Dict[str, torch.Tensor] cho từng lô.
    """
    if tokenizer is None:
        tokenizer = get_phobert_tokenizer()

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        yield encode_reviews(
            texts=chunk,
            max_len=max_len,
            tokenizer=tokenizer,
            apply_preprocessing=apply_preprocessing,
            return_tensors="pt"
        )


def decode_tokens(
    input_ids: Union[torch.Tensor, List[int], List[List[int]]],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    skip_special_tokens: bool = True
) -> Union[str, List[str]]:
    """Giải mã token IDs ngược lại chuỗi từ để kiểm tra và trực quan hóa.

    Args:
        input_ids: Tensor hoặc danh sách ID tokens (1D hoặc 2D).
        tokenizer: Tokenizer sử dụng.
        skip_special_tokens: Có bỏ qua các token đặc biệt (<s>, </s>, <pad>) không.

    Returns:
        Chuỗi văn bản (nếu đầu vào 1D) hoặc danh sách chuỗi (nếu đầu vào 2D).
    """
    if tokenizer is None:
        tokenizer = get_phobert_tokenizer()

    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 1:
            return tokenizer.decode(input_ids.tolist(), skip_special_tokens=skip_special_tokens)
        elif input_ids.dim() == 2:
            return [
                tokenizer.decode(ids.tolist(), skip_special_tokens=skip_special_tokens)
                for ids in input_ids
            ]
        else:
            raise ValueError(f"input_ids tensor phải có số chiều 1 hoặc 2, nhận được {input_ids.dim()}")

    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return [
                tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
                for ids in input_ids
            ]
        return tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)

    raise TypeError(f"Không hỗ trợ kiểu dữ liệu input_ids: {type(input_ids)}")
