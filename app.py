"""Streamlit demo: frozen baseline and locally trained BiLSTM/PhoBERT checkpoints.

Run from any directory: python -m streamlit run /absolute/path/to/app.py
Missing checkpoints fail explicitly; weights are never silently initialized.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.preprocess import clean_text_with_diff, preprocess_pipeline
from demo_models import (
    MODEL_NAMES, available_models, checkpoint_signature, load_neural_runtime,
    predict_neural_cleaned,
)

ROOT = Path(__file__).resolve().parent
MAX_ROWS = 5_000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LABELS = {0: "NEG", 1: "NEU", 2: "POS"}
NAMES = {"NEG": "Tiêu cực", "NEU": "Trung tính", "POS": "Tích cực"}
COLORS = {"NEG": "#e76f71", "NEU": "#e5b554", "POS": "#159a84"}
CATEGORIES = {
    "sarcasm": "Mỉa mai",
    "implicit_praise": "Khen ngầm",
    "teencode_slang": "Teencode",
}
DEFAULT_REVIEW = "Sp đẹp lắmmm, đóng gói cẩn thận, giao hàng nhanh. Sẽ ủng hộ shop tiếp!"
SAMPLE_CSV = pd.DataFrame({"comment": [
    DEFAULT_REVIEW,
    "Hàng quá tệ, chất lượng kém, không giống hình.",
    "Tạm được, chất lượng bình thường so với giá.",
    "Ko hài lòng, giao hàng chậm mà đóng gói ẩu.",
    "Áo đẹp, vải mát, shop tư vấn nhiệt tình.",
]}).to_csv(index=False).encode("utf-8-sig")


@st.cache_resource(show_spinner=False)
def load_baseline():
    """Keep the team's frozen baseline artifacts unchanged."""
    vectorizer = joblib.load(ROOT / "models/baseline_tfidf_vectorizer.pkl")
    classifier = joblib.load(ROOT / "models/baseline_linear_svc.pkl")
    return vectorizer, classifier


@st.cache_data(show_spinner=False)
def load_presets() -> list[dict]:
    data = json.loads((ROOT / "docs/demo_challenge_cases.json").read_text(encoding="utf-8"))
    return data["cases"]


@st.cache_resource(show_spinner="Đang nạp mô hình đã huấn luyện…", max_entries=2)
def _load_neural_cached(model_type: str, signature: tuple):
    return load_neural_runtime(model_type, signature)


def neural_runtime(model_type: str):
    return _load_neural_cached(model_type, checkpoint_signature(model_type))


def _has_content(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _predict_cleaned(texts: list[str], baseline=None) -> tuple[list[str], np.ndarray]:
    vectorizer, classifier = baseline if baseline is not None else load_baseline()
    features = vectorizer.transform(texts)
    labels = [LABELS[int(value)] for value in classifier.predict(features)]
    margins = np.asarray(classifier.decision_function(features))
    # These normalized support scores are NOT calibrated probabilities.
    exp_scores = np.exp(margins - margins.max(axis=1, keepdims=True))
    support = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    order = [list(classifier.classes_).index(class_id) for class_id in LABELS]
    return labels, support[:, order]


def analyze_review(text: str, model_type: str = "baseline") -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Bạn hãy nhập một bình luận trước khi phân tích.")
    if model_type == "baseline":
        engine, metadata = load_baseline(), {}
    else:
        engine = neural_runtime(model_type)
        metadata = {**engine.metadata, "device": str(engine.device)}
    started = time.perf_counter()
    cleaned, changes = clean_text_with_diff(text)
    if not _has_content(cleaned):
        raise ValueError("Bình luận cần có chữ hoặc số sau khi làm sạch.")
    labels, support = (_predict_cleaned([cleaned], engine) if model_type == "baseline"
                       else predict_neural_cleaned(engine, [cleaned]))
    return {
        "text": text, "cleaned_text": cleaned, "changes": changes,
        "label": labels[0],
        "scores": dict(zip(LABELS.values(), map(float, support[0]))),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "token_count": len(cleaned.split()),
        "model_type": model_type, "model_name": MODEL_NAMES[model_type],
        "score_type": "support" if model_type == "baseline" else "probability",
        "metadata": metadata,
    }


def read_csv_upload(payload: bytes) -> pd.DataFrame:
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Tệp vượt quá 10 MB. Bạn hãy chia thành tệp nhỏ hơn.")
    try:
        frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", keep_default_na=False)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError("Không đọc được CSV. Bạn hãy lưu tệp ở định dạng CSV UTF-8.") from error
    if not frame.columns.is_unique or "comment" not in frame.columns:
        raise ValueError("Tệp cần có cột tên chính xác là comment. Bạn có thể tải CSV mẫu bên dưới.")
    if frame.empty:
        raise ValueError("Tệp mới có tiêu đề, chưa có bình luận để phân tích.")
    if len(frame) > MAX_ROWS:
        raise ValueError("Mỗi lượt hỗ trợ tối đa 5.000 dòng. Bạn hãy chia tệp thành các phần nhỏ hơn.")
    return frame


def analyze_dataframe(frame: pd.DataFrame, progress: Callable[[float], None] | None = None,
                      model_type: str = "baseline") -> pd.DataFrame:
    """Preserve row order, original columns and empty rows; never fabricate labels."""
    if "comment" not in frame or frame.empty:
        raise ValueError("Dữ liệu cần có cột comment và ít nhất một dòng.")
    if len(frame) > MAX_ROWS:
        raise ValueError("Mỗi lượt hỗ trợ tối đa 5.000 dòng.")
    engine = load_baseline() if model_type == "baseline" else neural_runtime(model_type)
    source = frame["comment"].fillna("").astype(str).tolist()
    records = []
    for start in range(0, len(source), 128):
        batch = source[start:start + 128]
        cleaned = [preprocess_pipeline(text) for text in batch]
        valid = [index for index, text in enumerate(cleaned) if _has_content(text)]
        predictions = {}
        if valid:
            inputs = [cleaned[index] for index in valid]
            labels, scores = (_predict_cleaned(inputs, engine) if model_type == "baseline"
                             else predict_neural_cleaned(engine, inputs))
            predictions = {index: (label, score) for index, label, score in zip(valid, labels, scores)}
        for index, text in enumerate(cleaned):
            item = {"predicted_label": "", "predicted_sentiment": "", "processed_comment": text,
                    "analysis_status": "skipped_empty", "score_NEG": None, "score_NEU": None, "score_POS": None,
                    "model_name": MODEL_NAMES[model_type],
                    "score_type": "support" if model_type == "baseline" else "probability"}
            if index in predictions:
                label, score = predictions[index]
                item.update(predicted_label=label, predicted_sentiment=NAMES[label], analysis_status="ok")
                item.update({f"score_{name}": float(value) for name, value in zip(LABELS.values(), score)})
            records.append(item)
        if progress:
            progress(min((start + len(batch)) / len(source), 1.0))
    # Avoid overwriting user-provided columns with the same names.
    names = {name: name for name in records[0]}
    for name in names:
        while names[name] in frame.columns or names[name] in [v for k, v in names.items() if k != name]:
            names[name] = "ai_" + names[name]
    result = frame.copy()
    predicted = pd.DataFrame(records).rename(columns=names)
    for name in predicted:
        result[name] = predicted[name].to_numpy()
    result.attrs["result_columns"] = names
    result.attrs["model_type"] = model_type
    return result


def highlighted_original(text: str, changes: list[dict]) -> str:
    """Escape all user text before adding our own highlighting markup."""
    terms = sorted({change["original"] for change in changes if change.get("original")}, key=len, reverse=True)
    if not terms:
        return html.escape(text)
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, terms)) + r")(?!\w)", re.IGNORECASE)
    pieces, cursor = [], 0
    for match in pattern.finditer(text):
        pieces.extend([html.escape(text[cursor:match.start()]), "<mark>" + html.escape(match.group()) + "</mark>"])
        cursor = match.end()
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def support_chart(scores: dict, key: str) -> None:
    order = ["NEG", "NEU", "POS"]
    fig = go.Figure(go.Bar(
        x=[scores[name] * 100 for name in order], y=[NAMES[name] for name in order],
        orientation="h", marker_color=[COLORS[name] for name in order],
        text=[f"{scores[name] * 100:.1f}%" for name in order], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(height=190, margin=dict(l=5, r=48, t=8, b=8), bargap=0.56,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial", color="#51636c", size=12),
                      xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
                      yaxis=dict(autorange="reversed", fixedrange=True, tickfont=dict(size=13)))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def result_card(result: dict, chart_key: str) -> None:
    label = result["label"]
    st.markdown(f'<div class="section-label">KẾT QUẢ PHÂN TÍCH</div><div class="result-heading">'
                f'<span class="sentiment-dot" style="background:{COLORS[label]}"></span>'
                f'{NAMES[label]}<span class="label-code">{label}</span></div>'
                f'<div class="result-description">Dự đoán từ {html.escape(result["model_name"])}.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-meta"><span>◷ {result["latency_ms"]:.2f} ms</span>'
                f'<span>{result["token_count"]} token sau xử lý</span>'
                f'<span>{"TF-IDF" if result["model_type"] == "baseline" else "GPU" if result["metadata"].get("device", "").startswith("cuda") else "CPU"}</span></div>', unsafe_allow_html=True)
    support_chart(result["scores"], chart_key)
    if result["score_type"] == "support":
        st.caption("Điểm ủng hộ được chuẩn hóa từ LinearSVC; không phải xác suất đã hiệu chỉnh.")
    else:
        st.caption("Phân phối xác suất softmax từ checkpoint đã huấn luyện.")
        metadata = result["metadata"]
        val_f1 = (metadata.get("metrics") or {}).get("val_macro_f1")
        if val_f1 is not None:
            st.caption(f"Checkpoint epoch {metadata.get('epoch')} · Validation Macro F1: {val_f1 * 100:.2f}% (số lưu trong file).")


def choose_preset(case: dict) -> None:
    st.session_state["review_text"] = case["text"]
    st.session_state["selected_case"] = case["id"]
    submit_review()


def submit_review() -> None:
    try:
        st.session_state["single_result"] = analyze_review(st.session_state["review_text"], st.session_state.get("review_model", "baseline"))
        st.session_state.pop("single_error", None)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        st.session_state["single_result"] = None
        st.session_state["single_error"] = str(error)


def pipeline_panel(result: dict) -> None:
    st.markdown('<div class="section-label pipeline-heading">BÊN TRONG QUÁ TRÌNH XỬ LÝ</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown('<div class="text-panel"><span class="panel-kicker">01 / BÌNH LUẬN BAN ĐẦU</span>'
                    f'<p>{highlighted_original(result["text"], result["changes"])}</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="text-panel cleaned"><span class="panel-kicker">02 / SAU KHI LÀM SẠCH</span>'
                    f'<p>{html.escape(result["cleaned_text"])}</p></div>', unsafe_allow_html=True)
    with st.expander(f"Xem {len(result['changes'])} thay đổi được ghi nhận"):
        if result["changes"]:
            table = pd.DataFrame(result["changes"]).rename(columns={"original": "Ban đầu", "normalized": "Sau xử lý", "type": "Loại thay đổi"})
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.caption("Không có thay đổi teencode, ký tự kéo dài hoặc thông tin liên hệ cần ghi nhận.")


def batch_panel() -> None:
    intro, action = st.columns([3, 1])
    with intro:
        st.subheader("Từ một bình luận đến cả tập dữ liệu")
        st.caption("Tải CSV có cột comment · Định dạng UTF-8 · Tối đa 5.000 dòng / 10 MB")
    with action:
        st.download_button("↓ Tải CSV mẫu", SAMPLE_CSV, "sample_reviews.csv", "text/csv", width="stretch")
    mode = st.radio("Nguồn dữ liệu", ["Tải tệp CSV", "Thử dữ liệu mẫu"], horizontal=True, key="csv_mode")
    model_type = st.selectbox("Mô hình phân tích tệp", available_models(), format_func=MODEL_NAMES.get, key="csv_model")
    payload = None
    if mode == "Tải tệp CSV":
        uploaded = st.file_uploader("Chọn tệp đánh giá", type=["csv"], key="csv_upload")
        if uploaded is not None:
            payload = uploaded.getvalue()
    else:
        payload = SAMPLE_CSV
    version = checkpoint_signature(model_type) if model_type != "baseline" else ()
    fingerprint = (hashlib.sha256(payload).hexdigest(), model_type, version) if payload is not None else None
    if st.session_state.get("csv_fingerprint") != fingerprint:
        st.session_state["csv_fingerprint"] = fingerprint
        st.session_state.pop("batch_result", None)
    frame = None
    if payload is not None:
        try:
            frame = read_csv_upload(payload)
            st.caption(f"Đã đọc {len(frame):,} dòng. Xem trước tối đa 5 dòng:")
            st.dataframe(frame.head(5), width="stretch", hide_index=True)
        except ValueError as error:
            st.error(str(error))
    if st.button("Phân tích toàn bộ tệp", type="primary", disabled=frame is None, key="analyze_csv"):
        progress = st.progress(0.0, text="Đang làm sạch và phân loại bình luận…")
        try:
            with st.spinner(f"Đang phân tích bằng {MODEL_NAMES[model_type]}…"):
                st.session_state["batch_result"] = analyze_dataframe(frame, lambda value: progress.progress(value), model_type)
        except (ValueError, OSError, RuntimeError, ImportError) as error:
            st.error(f"Không thể chạy mô hình đã chọn: {error}")
        progress.empty()
    result = st.session_state.get("batch_result")
    if result is None:
        st.markdown('<div class="batch-empty"><div class="empty-glyph">≋</div>'
                    '<strong>Một góc nhìn tổng thể về khách hàng</strong>'
                    '<p>Phân loại từng bình luận, xem phân phối cảm xúc và tải kết quả về.</p></div>', unsafe_allow_html=True)
        return
    names = result.attrs["result_columns"]
    status, label = names["analysis_status"], names["predicted_label"]
    valid = result.loc[result[status] == "ok"]
    counts = valid[label].value_counts()
    st.success(f"Hoàn tất: {len(valid):,}/{len(result):,} bình luận đã được phân loại.")
    skipped = len(result) - len(valid)
    if skipped:
        st.info(f"{skipped} dòng không còn chữ hoặc số sau xử lý. Các dòng này vẫn được giữ trong CSV, với trạng thái skipped_empty.")
    chart, summary = st.columns([1, 1.5], gap="large")
    with chart:
        if not valid.empty:
            order = ["POS", "NEU", "NEG"]
            fig = go.Figure(go.Pie(labels=[NAMES[n] for n in order], values=[int(counts.get(n, 0)) for n in order],
                                   hole=0.72, marker_colors=[COLORS[n] for n in order], sort=False,
                                   textinfo="percent", hovertemplate="%{label}: %{value}<extra></extra>"))
            fig.update_layout(height=270, margin=dict(l=15, r=15, t=10, b=15),
                              paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
                              annotations=[dict(text=f"<b>{len(valid)}</b><br>bình luận", x=0.5, y=0.5, showarrow=False, font_size=18)])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="batch_distribution")
    with summary:
        for name in ("POS", "NEU", "NEG"):
            st.metric(NAMES[name], f"{counts.get(name, 0):,} bình luận")
        st.download_button("↓ Tải kết quả phân tích", result.to_csv(index=False).encode("utf-8-sig"),
                           "download_predictions.csv", "text/csv", type="primary", key="download_predictions")
    st.dataframe(result, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Shopee Insight · Sentiment Lab", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(f'<style>{(ROOT / "static/style.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)
    st.markdown(f'''<div class="topbar"><div class="brand"><span class="brand-icon">s</span>
        <span>Shopee<span class="brand-light"> Insight</span></span></div>
        <div class="topbar-right"><span class="live-dot"></span> {len(available_models())} mô hình sẵn sàng <span class="project-tag">AI CAPSTONE</span></div></div>
        <div class="hero"><div class="hero-content"><div class="eyebrow">VIETNAMESE SENTIMENT LAB / WEEK 04</div>
        <h1>Hiểu cảm xúc trong<br>từng đánh giá.</h1>
        <p>Khám phá cách AI đọc, làm sạch và phân tích bình luận Shopee.</p></div>
        <div class="hero-visual"><div class="orbit orbit-one"></div><div class="orbit orbit-two"></div>
        <div class="signal-bubble positive">Tích cực <span>↗</span></div><div class="signal-bubble neutral">Trung tính <span>→</span></div>
        <div class="signal-bubble negative">Tiêu cực <span>↘</span></div><div class="visual-caption">Từ ngôn ngữ đến thấu hiểu</div></div></div>''', unsafe_allow_html=True)
    try:
        load_baseline()
        presets = load_presets()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Không nạp được tài nguyên Baseline hoặc bộ câu mẫu: {error}")
        st.stop()
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = DEFAULT_REVIEW
    if "review_model" not in st.session_state:
        st.session_state["review_model"] = "baseline"
    if "single_result" not in st.session_state:
        st.session_state["single_result"] = analyze_review(DEFAULT_REVIEW)
    previous = st.session_state.get("single_result")
    if previous and "model_type" not in previous:
        st.session_state["single_result"] = analyze_review(previous["text"])
    analyze_tab, compare_tab, csv_tab = st.tabs(["01  Phân tích đánh giá", "02  So sánh mô hình", "03  Phân tích tệp CSV"])
    with analyze_tab:
        left, right = st.columns([1.12, 1], gap="large")
        with left:
            with st.container(border=True):
                st.markdown('<div class="section-label">THỬ MỘT BÌNH LUẬN</div>', unsafe_allow_html=True)
                st.selectbox("Mô hình phân tích", available_models(), format_func=MODEL_NAMES.get,
                             key="review_model", on_change=submit_review)
                st.text_area("Khách hàng đang nói gì?", key="review_text", height=146, max_chars=5_000,
                             placeholder="Nhập một đánh giá Shopee bằng tiếng Việt…")
                button, note = st.columns([1, 1.1])
                with button:
                    st.button("Phân tích đánh giá  →", type="primary", on_click=submit_review, key="analyze_review", width="stretch")
                with note:
                    st.caption("Hỗ trợ teencode, từ viết tắt và tiếng Việt.")
            with st.expander("15 tình huống để thử mô hình", expanded=False):
                group_tabs = st.tabs([f"{name} · 5" for name in CATEGORIES.values()])
                for category, group_tab in zip(CATEGORIES, group_tabs):
                    with group_tab:
                        for case in presets:
                            if case["category"] == category:
                                title = case["title"].split(": ", 1)[-1]
                                st.button(title, key=f"preset_{case['id']}", on_click=choose_preset,
                                          args=(case,), width="stretch")
        result = st.session_state.get("single_result")
        with right:
            with st.container(border=True):
                if result:
                    result_card(result, "single_support")
                else:
                    st.info(st.session_state.get("single_error", "Nhập bình luận để xem kết quả."))
        if result:
            if st.session_state["review_text"] != result["text"]:
                st.caption("Bạn đã thay đổi nội dung. Nhấn Phân tích đánh giá để cập nhật kết quả.")
            selected = next((case for case in presets if case["text"] == result["text"]), None)
            if selected:
                st.info(f"Câu mẫu {CATEGORIES[selected['category']].lower()} · Nhãn tham chiếu: {NAMES[selected['expected_label']]} · {result['model_name']} dự đoán: {NAMES[result['label']]}")
                with st.expander("Vì sao câu này là một tình huống thử thách?"):
                    st.write(selected["explanation"])
            pipeline_panel(result)
    with compare_tab:
        st.subheader("Ba mô hình, cùng một bình luận")
        st.caption("Dự đoán thực tế từ từng mô hình. Nội dung đồng bộ từ tab Phân tích đánh giá.")
        if st.button("Chạy lại so sánh", key="refresh_comparison"):
            st.session_state.pop("comparison_key", None)
        result = st.session_state.get("single_result")
        if result:
            st.markdown(f'<div class="comparison-quote">“{html.escape(result["text"])}”</div>', unsafe_allow_html=True)
        versions = tuple((kind, checkpoint_signature(kind)) for kind in available_models() if kind != "baseline")
        comparison_key = (result["text"], versions) if result else None
        if result and st.session_state.get("comparison_key") != comparison_key:
            comparisons, failures = {}, {}
            for kind in available_models():
                try:
                    comparisons[kind] = analyze_review(result["text"], kind)
                except (ValueError, OSError, RuntimeError, ImportError) as error:
                    failures[kind] = str(error)
            st.session_state["comparison_key"] = comparison_key
            st.session_state["comparison_results"] = comparisons
            st.session_state["comparison_failures"] = failures
        comparisons = st.session_state.get("comparison_results", {}) if result else {}
        columns = st.columns(3, gap="medium")
        for kind, column in zip(("baseline", "phobert", "bilstm"), columns):
            with column, st.container(border=True):
                st.markdown(f"#### {MODEL_NAMES[kind]}")
                if kind in comparisons:
                    result_card(comparisons[kind], f"comparison_{kind}")
                elif result:
                    message = st.session_state.get("comparison_failures", {}).get(kind, "Chưa có checkpoint của mô hình này.")
                    st.info(message)
                else:
                    st.info("Nhập bình luận ở tab đầu để xem kết quả.")
        if "phobert" in comparisons:
            val_f1 = (comparisons["phobert"]["metadata"].get("metrics") or {}).get("val_macro_f1")
            if val_f1 is not None and val_f1 < 0.75:
                st.warning(f"Checkpoint PhoBERT hiện ghi Validation Macro F1 {val_f1 * 100:.2f}%. Đây là chỉ số validation kèm file, chưa phải kết quả trên tập test; các dự đoán bên trên chạy trực tiếp từ checkpoint đã nhận.")
        case = next((item for item in presets if result and item["text"] == result["text"]), None)
        if case and comparisons.get("baseline", {}).get("label") != case["expected_label"]:
            corrected = [MODEL_NAMES[kind] for kind in ("phobert", "bilstm") if comparisons.get(kind, {}).get("label") == case["expected_label"]]
            if corrected:
                st.success(f"Với nhãn tham chiếu {NAMES[case['expected_label']]}, {' và '.join(corrected)} sửa đúng trường hợp Baseline dự đoán sai.")
    with csv_tab:
        batch_panel()
    st.markdown('<div class="app-footer"><span>Shopee Insight · AI Capstone / VNUK</span>'
                '<span>Web demo: Đặng Đức Minh</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
