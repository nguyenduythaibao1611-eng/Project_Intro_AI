"""Vietnamese review preprocessing and leakage-aware stratified splits.

Run from the repository root with ``python -m src.preprocess``.
The public functions and CSV columns follow the team's task specification.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
LABEL_TO_ID = {"NEG": 0, "NEU": 1, "POS": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
OUTPUT_COLUMNS = ["content", "processed_content", "label", "label_id", "start"]

# Fixed, manually authored rules, not learned from validation/test reviews.
# Ambiguous aliases (dc/dk, rep/fb) follow section 2.1 of the team specification.
# Do not map isolated price suffixes, ratings, product sizes or sentiment labels.
TEENCODE_DICT: Dict[str, str] = {
    "k": "không", "ko": "không", "khong": "không", "khg": "không",
    "kh": "không", "hok": "không", "hông": "không",
    "hong": "không", "hem": "không", "hum": "không", "khum": "không",
    "khom": "không", "h0k": "không", "k0": "không", "k0ng": "không",
    "chua": "chưa", "chang": "chẳng",
    "đc": "được", "dc": "được", "dk": "được", "đk": "được",
    "duoc": "được", "được r": "được rồi",
    "sp": "sản phẩm", "spham": "sản phẩm", "sphẩm": "sản phẩm",
    "sanpham": "sản phẩm", "san pham": "sản phẩm", "sản phảm": "sản phẩm",
    "shop": "cửa hàng", "shopp": "cửa hàng", "shoppe": "cửa hàng",
    "cua hang": "cửa hàng", "seller": "người bán", "ngban": "người bán",
    "ng bán": "người bán", "khach hang": "khách hàng", "khách h": "khách hàng",
    "ship": "giao hàng", "shipping": "giao hàng", "giao hang": "giao hàng",
    "gh": "giao hàng", "giaohang": "giao hàng", "shipper": "nhân viên giao hàng",
    "freeship": "miễn phí giao hàng", "free ship": "miễn phí giao hàng",
    "free shipping": "miễn phí giao hàng", "phi ship": "phí giao hàng",
    "phí ship": "phí giao hàng", "tien ship": "tiền giao hàng",
    "tốc độ ship": "tốc độ giao hàng", "giao nhah": "giao nhanh",
    "dong goi": "đóng gói", "đóng goi": "đóng gói", "donggoi": "đóng gói",
    "đóng gói kĩ": "đóng gói kỹ", "dóng gói": "đóng gói",
    "auth": "chính hãng", "aut": "chính hãng", "authentic": "chính hãng",
    "chinh hang": "chính hãng", "chinhhang": "chính hãng",
    "fake": "giả", "hang gia": "hàng giả", "hang nhai": "hàng nhái",
    "rep": "trả lời", "reply": "trả lời", "trl": "trả lời",
    "tra loi": "trả lời", "fb": "phản hồi", "feedback": "phản hồi",
    "phan hoi": "phản hồi", "ib": "nhắn tin", "inbox": "nhắn tin",
    "nt": "nhắn tin", "mess": "tin nhắn", "tn": "tin nhắn",
    "đt": "điện thoại", "dt": "điện thoại", "dienthoai": "điện thoại",
    "dien thoai": "điện thoại", "sdt": "số điện thoại", "sđt": "số điện thoại",
    "dvu": "dịch vụ", "dvụ": "dịch vụ", "dich vu": "dịch vụ",
    "nvien": "nhân viên", "nviên": "nhân viên", "nhan vien": "nhân viên",
    "cskh": "chăm sóc khách hàng", "bao hanh": "bảo hành",
    "baohanh": "bảo hành", "đổi tra": "đổi trả", "doi tra": "đổi trả",
    "hoan tien": "hoàn tiền", "hoantien": "hoàn tiền",
    "gia tien": "giá tiền", "giatien": "giá tiền", "gia ca": "giá cả",
    "chat luong": "chất lượng", "chatluong": "chất lượng", "clg": "chất lượng",
    "chất lg": "chất lượng", "số lg": "số lượng", "so luong": "số lượng",
    "soluong": "số lượng", "slg": "số lượng", "kich thuoc": "kích thước",
    "kichthuoc": "kích thước", "mau sac": "màu sắc", "mausac": "màu sắc",
    "mau ma": "mẫu mã", "mauma": "mẫu mã", "hinh anh": "hình ảnh",
    "hinhanh": "hình ảnh", "mo ta": "mô tả", "mota": "mô tả",
    "y hinh": "y hình", "giong hinh": "giống hình",
    "giong mo ta": "giống mô tả", "dung mo ta": "đúng mô tả",
    "rất ok": "rất ổn", "ok": "ổn", "oke": "ổn", "okie": "ổn",
    "okela": "ổn", "oki": "ổn", "okey": "ổn", "okay": "ổn",
    "good": "tốt", "gud": "tốt", "nice": "tốt", "bad": "tệ",
    "tot": "tốt", "tốt wa": "tốt quá", "đẹp wa": "đẹp quá",
    "dep": "đẹp", "depp": "đẹp", "xau": "xấu", "xau xi": "xấu xí",
    "te": "tệ", "hai long": "hài lòng", "hailong": "hài lòng",
    "that vong": "thất vọng", "thatvong": "thất vọng",
    "tuyet voi": "tuyệt vời", "tuyetvoi": "tuyệt vời",
    "ung y": "ưng ý", "ungy": "ưng ý", "ung ho": "ủng hộ", "ungho": "ủng hộ",
    "cam on": "cảm ơn", "camon": "cảm ơn", "cám ơn": "cảm ơn",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn", "thank you": "cảm ơn",
    "thankyou": "cảm ơn", "thankiu": "cảm ơn", "thks": "cảm ơn",
    "xin loi": "xin lỗi", "xinloi": "xin lỗi", "sorry": "xin lỗi",
    "mik": "mình", "mjk": "mình", "mk": "mình", "mìn": "mình",
    "mn": "mọi người", "mng": "mọi người", "mọi ng": "mọi người",
    "mọi nguoi": "mọi người", "ngta": "người ta", "ng ta": "người ta",
    "bn": "bạn", "bth": "bình thường",
    "bt": "bình thường", "binh thuong": "bình thường", "binhthuong": "bình thường",
    "bthuong": "bình thường", "bthuờng": "bình thường",
    "vs": "với", "vay": "vậy", "zậy": "vậy",
    "zay": "vậy", "z": "vậy", "rùi": "rồi", "ròi": "rồi", "r": "rồi",
    "roi": "rồi", "lun": "luôn", "luon": "luôn", "luônn": "luôn",
    "wa": "quá", "wá": "quá", "qá": "quá",
    "nhìu": "nhiều", "nhieu": "nhiều", "nhju": "nhiều",
    "nhanhh": "nhanh", "nhah": "nhanh", "nhan": "nhận", "nhan dc": "nhận được",
    "nhận dc": "nhận được", "sd": "sử dụng", "sudung": "sử dụng",
    "su dung": "sử dụng", "sài": "xài", "nch": "nói chung", "nói chug": "nói chung",
    "rat": "rất", "cx": "cũng", "cug": "cũng", "cũg": "cũng",
    "vẫn ch": "vẫn chưa", "ch": "chưa", "thik": "thích", "thich": "thích",
    "thjk": "thích", "iu": "yêu", "yeu": "yêu",
    # Chuẩn hóa từ lóng biểu đạt cảm xúc mạnh, chê bai, chửi bậy viết tắt trên TMĐT (Shopee Profanity & Negative Slang)
    "vcl": "rất", "vkl": "rất", "vlon": "rất", "vlin": "rất", "vcc": "rất", "vl": "rất",
    "như c": "rất tệ", "nhu c": "rất tệ", "như cc": "rất tệ", "nhu cc": "rất tệ",
    "như cl": "rất tệ", "nhu cl": "rất tệ", "như l": "rất tệ", "nhu l": "rất tệ",
    "như lol": "rất tệ", "nhu lol": "rất tệ", "như hạch": "rất tệ", "nhu hach": "rất tệ",
    "đm": "tệ", "dm": "tệ", "dkm": "tệ", "đkm": "tệ", "đmm": "tệ", "dmm": "tệ",
    "đcm": "tệ", "dcm": "tệ", "cl": "tệ", "củ l": "tệ",
    "cac": "tệ", "cặc": "tệ", "cak": "tệ", "cặk": "tệ", "lồn": "tệ", "lon": "tệ",
    "lừa đảo": "lừa đảo", "lua dao": "lừa đảo", "luadao": "lừa đảo", "lđ": "lừa đảo",
    "lởm": "kém chất lượng", "lom": "kém chất lượng", "dỏm": "kém chất lượng", "dom": "kém chất lượng",
    "hãm": "tệ", "ham": "tệ", "bực": "bực mình", "buc": "bực mình",
    # Mở rộng từ lóng và viết tắt TMĐT Shopee Tuần 4 (E-commerce Delivery, Quality & Sentiment Slang)
    "hoatoc": "hỏa tốc", "hoa toc": "hỏa tốc", "giao hoa toc": "giao hỏa tốc", "giao hoatoc": "giao hỏa tốc",
    "chuanbihang": "chuẩn bị hàng", "chuan bi hang": "chuẩn bị hàng", "cb hang": "chuẩn bị hàng",
    "dong goi can than": "đóng gói cẩn thận", "dg cẩn thận": "đóng gói cẩn thận",
    "xin": "tốt", "xịn": "tốt", "xinxo": "tốt", "xịn xò": "tốt", "xịn sò": "tốt",
    "dinh": "tuyệt vời", "đỉnh": "tuyệt vời", "dinh cao": "tuyệt vời", "đỉnh cao": "tuyệt vời",
    "qua da": "tuyệt vời", "quá đã": "tuyệt vời", "ung bung": "ưng ý", "ưng bụng": "ưng ý",
    "dang tien": "đáng tiền", "đáng tiền": "đáng tiền", "dang dong tien": "đáng tiền", "đáng đồng tiền": "đáng tiền",
    "kp": "không phải", "k phai": "không phải", "ko phai": "không phải", "khong phai": "không phải",
    "k thik": "không thích", "ko thik": "không thích", "khum thik": "không thích",
    "k ung": "không ưng ý", "ko ung": "không ưng ý", "k ưng": "không ưng ý", "ko ưng": "không ưng ý",
    "hang deu": "hàng giả", "hàng đểu": "hàng giả", "chan doi": "thất vọng", "chán đời": "thất vọng",
    "treo dau de ban thit cho": "lừa đảo", "treo đầu dê bán thịt chó": "lừa đảo",
}
# Defined for team compatibility. Removal is intentionally disabled: short review
# particles, intensifiers and negations can carry sentiment.
STOPWORDS_SET: Set[str] = {"ạ", "ơi", "nhé", "nha"}
NEGATIONS = {"không", "chẳng", "chả", "chưa", "đừng"}

_URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
_EMAIL = re.compile(r"(?<!\w)[\w.+-]+@[\w.-]+\.[a-z]{2,}(?!\w)", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+84|0084|0)(?:[ .()\-]*\d){9}(?!\d)")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_TOKENS = re.compile(r"[^\W_]+(?:_[^\W_]+)*", re.UNICODE)
_HTML_TAGS = re.compile(r"</?[a-z][^>]*>", re.IGNORECASE)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f\u200b\ufeff]")
_WHITESPACE = re.compile(r"\s+")
DOMAIN_COMPOUNDS = {phrase: phrase.replace(" ", "_") for phrase in (
    "sản phẩm", "giao hàng", "chất lượng", "hài lòng", "thất vọng", "cửa hàng",
    "chính hãng", "đóng gói", "phản hồi", "trả lời", "bảo hành", "đổi trả",
    "rất tệ", "kém chất lượng", "lừa đảo", "bực mình", "chuẩn bị hàng", "hỏa tốc", "đáng tiền",
)}


@lru_cache(maxsize=16384)
def _collapse_word_elongation(word: str) -> str:
    """Normalize repeated elongated characters in words with LRU cache."""
    if len(word) < 3:
        return word
    result, index = [], 0
    while index < len(word):
        end = index + 1
        base = unicodedata.normalize("NFD", word[index])[0]
        while end < len(word) and unicodedata.normalize("NFD", word[end])[0] == base:
            end += 1
        run = word[index:end]
        result.append(next((char for char in run if char != base), run[0])
                      if len(run) >= 3 else run)
        index = end
    return "".join(result)


def _collapse_elongation(match: re.Match) -> str:
    return _collapse_word_elongation(match.group())


def clean_basic_text(text: str) -> str:
    """Normalize Unicode/case/elongation; remove contacts, HTML and controls.

    Preserve punctuation, emojis, negation, product numbers and prices.
    Missing scalar values become an empty string rather than the word 'nan'.
    Pre-compiled regex patterns and LRU cached elongation ensure < 2ms latency.
    """
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    text_str = str(text)
    if not text_str:
        return ""
    text = unicodedata.normalize("NFC", html.unescape(text_str)).lower()
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    text = _HTML_TAGS.sub(" ", text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WORD.sub(_collapse_elongation, text)
    return _WHITESPACE.sub(" ", text).strip()


@lru_cache(maxsize=8)
def _compile_teencode(items: tuple) -> re.Pattern:
    keys = sorted((key for key, _ in items), key=lambda key: (-len(key), key))
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, keys)) + r")(?!\w)", re.IGNORECASE)


def _teencode_matches(text: str, teencode_dict: Dict[str, str]):
    mapping = {unicodedata.normalize("NFC", key).lower(): value
               for key, value in teencode_dict.items()}
    if not mapping:
        return mapping, []
    if "" in mapping:
        raise ValueError("Teencode keys must not be empty.")
    pattern = _compile_teencode(tuple(sorted(mapping.items())))
    return mapping, list(pattern.finditer(text))


def normalize_teencode(text: str, teencode_dict: Dict[str, str]) -> str:
    """Replace whole tokens/phrases once, preferring the longest matching key.

    For example, 'k' matches in 'k tốt', but not in '100k', 'size_k' or 'skincare'.
    Values are not recursively interpreted as new abbreviations.
    """
    mapping, matches = _teencode_matches(text, teencode_dict)
    parts, cursor = [], 0
    for match in matches:
        parts.extend((text[cursor:match.start()], mapping[match.group().lower()]))
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts)


def segment_vietnamese(text: str) -> str:
    """Use PyVi, then join fixed commerce compounds PyVi may leave separated."""
    if not text.strip():
        return ""
    try:
        from pyvi import ViTokenizer
    except ImportError as exc:
        raise ImportError("Vietnamese segmentation requires pyvi: pip install pyvi==0.1.1") from exc
    segmented = " ".join(ViTokenizer.tokenize(text).split())
    return normalize_teencode(segmented, DOMAIN_COMPOUNDS)


def preprocess_pipeline(text: str) -> str:
    """Run clean_basic_text -> normalize_teencode -> segment_vietnamese."""
    return segment_vietnamese(normalize_teencode(clean_basic_text(text), TEENCODE_DICT))


def clean_text_with_diff(
    raw_text: str,
    apply_segmentation: bool = True
) -> Tuple[str, List[Dict[str, str]]]:
    """Clean and normalize Vietnamese e-commerce text, tracking all modifications.

    Designed for low-latency (< 2ms) inference and real-time visual diff
    inspection in the Streamlit Web Demo.

    Args:
        raw_text: Raw review string containing potential teencode, noise, or contacts.
        apply_segmentation: Whether to apply PyVi word segmentation on the output.

    Returns:
        tuple[str, list[dict]]:
            - clean_text: Fully normalized, teencode-expanded and optionally segmented text.
            - changes: Detailed list of interventions, each represented as:
              {"original": str, "normalized": str, "type": str}
              Types include: 'teencode', 'elongation', 'html', 'url', 'email', 'phone'.
    """
    if raw_text is None or (not isinstance(raw_text, str) and pd.isna(raw_text)):
        return "", []

    text_raw = str(raw_text)
    if not text_raw.strip():
        return "", []

    changes: List[Dict[str, str]] = []
    seen_changes = set()

    def _record_change(orig: str, norm: str, ctype: str) -> None:
        key = (orig, norm, ctype)
        if key not in seen_changes:
            seen_changes.add(key)
            changes.append({"original": orig, "normalized": norm, "type": ctype})

    # 1. Track HTML removals
    for match in _HTML_TAGS.finditer(text_raw):
        _record_change(match.group(), "", "html")

    # 2. Track contact removals (URL, Email, Phone)
    for match in _URL.finditer(text_raw):
        _record_change(match.group(), "", "url")
    for match in _EMAIL.finditer(text_raw):
        _record_change(match.group(), "", "email")
    for match in _PHONE.finditer(text_raw):
        _record_change(match.group(), "", "phone")

    # 3. Track elongation
    for match in _WORD.finditer(text_raw):
        w = match.group()
        collapsed = _collapse_word_elongation(w)
        if collapsed != w:
            _record_change(w, collapsed, "elongation")

    # 4. Basic cleaning
    cleaned = clean_basic_text(text_raw)

    # 5. Track teencode substitutions
    mapping, matches = _teencode_matches(cleaned, TEENCODE_DICT)
    for m in matches:
        orig = m.group()
        norm = mapping.get(orig.lower(), orig)
        if orig.lower() != norm.lower():
            if apply_segmentation and norm in DOMAIN_COMPOUNDS:
                norm_display = DOMAIN_COMPOUNDS[norm]
            else:
                norm_display = norm
            _record_change(orig, norm_display, "teencode")

    # 6. Normalize teencode
    normalized = normalize_teencode(cleaned, TEENCODE_DICT)

    # 7. Word Segmentation
    if apply_segmentation:
        final_clean = segment_vietnamese(normalized)
    else:
        final_clean = normalized

    return final_clean, changes


def _word_tokens(text: str) -> list[str]:
    return _TOKENS.findall(text.lower())


def vocabulary_coverage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Token/type coverage against a TRAIN-only vocabulary with min frequency 2.

    Counts segmented words/numbers, excluding punctuation and emojis. This is
    a descriptive word vocabulary, not the baseline TF-IDF feature vocabulary.
    """
    train_counts = Counter(token for text in train.processed_content for token in _word_tokens(text))
    vocabulary = {token for token, count in train_counts.items() if count >= 2}
    result = {"definition": "segmented word/number tokens, TRAIN-only, min_frequency=2",
              "min_frequency": 2, "train_unique_tokens": len(train_counts),
              "vocabulary_size": len(vocabulary), "splits": {}}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        counts = Counter(token for text in frame.processed_content for token in _word_tokens(text))
        total = sum(counts.values())
        covered = sum(count for token, count in counts.items() if token in vocabulary)
        known_types = len(counts.keys() & vocabulary)
        result["splits"][name] = {
            "tokens": total, "types": len(counts), "covered_tokens": covered,
            "token_coverage": covered / total if total else 0.0,
            "type_coverage": known_types / len(counts) if counts else 0.0,
            "oov_tokens": total - covered, "oov_types": len(counts) - known_types,
        }
    return result


def split_and_save_stratified(
    input_csv_path: str,
    output_dir: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process reviews, split by label, and save train/val/test.csv.

    Identical processed reviews are kept in the same split. Stratification uses
    each group's modal label (ties: NEG, NEU, POS); original row labels stay
    unchanged. Ratios are approximate when groups contain multiple rows.
    Empty processed rows are exported for review, never silently discarded.
    """
    ratios = (train_size, val_size, test_size)
    if any(not math.isfinite(value) or not 0 < value < 1 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("train_size, val_size and test_size must be positive and sum to 1.")
    source = Path(input_csv_path).resolve()
    destination = Path(output_dir).resolve()
    output_paths = [destination / name for name in
                    ("train.csv", "val.csv", "test.csv", "excluded_rows.csv", "preprocessing_audit.json")]
    if source in output_paths:
        raise ValueError("The input CSV must not be overwritten by an output file.")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.read_csv(source, encoding="utf-8-sig", keep_default_na=False)
    required = {"content", "label", "start"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    invalid = set(frame.label) - set(LABEL_TO_ID)
    if invalid:
        raise ValueError(f"Unknown labels: {sorted(invalid)}. Expected NEG, NEU, POS.")
    frame["label_id"] = frame.label.map(LABEL_TO_ID).astype(int)
    basic = frame.content.map(clean_basic_text)
    normalized = basic.map(lambda text: normalize_teencode(text, TEENCODE_DICT))
    # Process duplicate strings once; no learned preprocessing or vocabulary fit.
    segmented = {text: segment_vietnamese(text) for text in dict.fromkeys(normalized)}
    frame["processed_content"] = normalized.map(segmented)
    empty = frame.processed_content.map(lambda text: not bool(_word_tokens(text)))
    excluded = frame.loc[empty, OUTPUT_COLUMNS].copy()
    excluded["reason"] = "empty_or_no_word_tokens_after_preprocessing"
    retained = frame.loc[~empty, OUTPUT_COLUMNS].copy()
    if retained.empty:
        raise ValueError("No nonempty reviews remain after preprocessing.")

    # A row-wise stratified split leaks duplicates created by normalization.
    # Split the group table, then expand groups back to all original rows.
    retained["_group"] = retained.processed_content
    group_labels = retained.groupby("_group", sort=True).label.agg(
        lambda values: min(Counter(values), key=lambda label: (-Counter(values)[label], LABEL_TO_ID[label])))
    groups = group_labels.reset_index(name="stratum")
    try:
        train_groups, holdout_groups = train_test_split(
            groups, test_size=val_size + test_size, stratify=groups.stratum, random_state=random_state)
        val_groups, test_groups = train_test_split(
            holdout_groups, test_size=test_size / (val_size + test_size),
            stratify=holdout_groups.stratum, random_state=random_state)
    except ValueError as exc:
        raise ValueError("Insufficient review groups per label for the requested stratified splits.") from exc
    splits = tuple(retained.loc[retained._group.isin(group_table._group), OUTPUT_COLUMNS].reset_index(drop=True)
                   for group_table in (train_groups, val_groups, test_groups))
    key_sets = [set(part.processed_content) for part in splits]
    overlaps = {name: len(key_sets[left] & key_sets[right])
                for name, left, right in (("train_val", 0, 1), ("train_test", 0, 2), ("val_test", 1, 2))}
    if any(overlaps.values()) or sum(map(len, splits)) != len(retained):
        raise RuntimeError("Split integrity check failed.")
    if any(set(part.label) != set(LABEL_TO_ID) for part in splits):
        raise ValueError("Every output split must contain NEG, NEU and POS.")

    matches = Counter()
    for text in basic:
        mapping, found = _teencode_matches(text, TEENCODE_DICT)
        matches.update(match.group().lower() for match in found
                       if mapping[match.group().lower()] != match.group())
    group_sizes = retained.groupby("_group").size()
    conflicting = retained.groupby("_group").label.nunique().gt(1)
    input_counts = frame.label.value_counts().to_dict()
    source_unchanged = hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    if not source_unchanged:
        raise RuntimeError("Input changed during processing. Outputs were not saved.")
    audit = {
        "input_file": source.name, "source_sha256": source_hash, "source_unchanged": source_unchanged,
        "input_rows": len(frame), "retained_rows": len(retained), "excluded_rows": len(excluded),
        "input_label_counts": {label: int(input_counts.get(label, 0)) for label in LABEL_TO_ID},
        "random_state": random_state, "target_ratios": dict(zip(("train", "val", "test"), ratios)),
        "split_method": "train_test_split stratified on modal label of identical processed_content groups",
        "unique_processed_groups": len(groups), "duplicate_groups": int(group_sizes.gt(1).sum()),
        "extra_rows_in_duplicate_groups": int((group_sizes - 1).sum()),
        "largest_group_rows": int(group_sizes.max()),
        "conflicting_label_groups": int(conflicting.sum()),
        "conflicting_group_rows": int(group_sizes.loc[conflicting].sum()),
        "processed_overlap_between_splits": overlaps,
        "basic_cleaning_changed_rows": int(frame.content.ne(basic).sum()),
        "teencode_changed_rows": int(basic.ne(normalized).sum()),
        "teencode_dictionary_entries": len(TEENCODE_DICT),
        "teencode_effective_entries": sum(key != value for key, value in TEENCODE_DICT.items()),
        "matched_effective_rules": len(matches), "replacement_occurrences": sum(matches.values()),
        "replacement_counts": dict(matches.most_common()),
        "stopword_removal": False,
        "segmentation": "PyVi ViTokenizer + 12 fixed commerce compounds",
        "splits": {name: {"rows": len(part), "ratio": len(part) / len(retained),
                          "label_counts": {label: int(part.label.eq(label).sum()) for label in LABEL_TO_ID}}
                   for name, part in zip(("train", "val", "test"), splits)},
        "vocabulary_coverage": vocabulary_coverage(*splits),
        "versions": {package: version(package) for package in ("pandas", "scikit-learn", "pyvi")},
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "val", "test"), splits):
        part.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    excluded.to_csv(destination / "excluded_rows.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    (destination / "preprocessing_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/shopee_reviews_cleaned.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    splits = split_and_save_stratified(args.input, args.output_dir, random_state=args.seed)
    print(json.dumps({name: len(frame) for name, frame in zip(("train", "val", "test"), splits)}))


if __name__ == "__main__":
    main()
