"""Exercise the real baseline and the Streamlit UI, including uploaded CSV files."""
import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import analyze_dataframe, analyze_review, highlighted_original, load_presets, read_csv_upload
from demo_models import available_models

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("text,label", [
    ("Sản phẩm rất tốt, giao hàng nhanh, đóng gói cẩn thận.", "POS"),
    ("Hàng quá tệ không giống hình, chất lượng kém, shop lừa đảo.", "NEG"),
])
def test_real_baseline_predictions(text, label):
    result = analyze_review(text)
    assert result["label"] == label
    assert sum(result["scores"].values()) == pytest.approx(1.0)
    assert result["cleaned_text"]


def test_page_load_and_blank_input_do_not_fabricate_prediction():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert len([button for button in at.button if (button.key or "").startswith("preset_")]) == 15
    assert set(at.session_state["comparison_results"]) == set(available_models())
    at.text_area(key="review_text").set_value("   ")
    at.button(key="analyze_review").click().run()
    assert not at.exception
    assert at.session_state["single_result"] is None
    assert any("nhập một bình luận" in info.value for info in at.info)


def test_preset_callback_fills_text_and_uses_actual_baseline():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    case = load_presets()[0]
    at.button(key=f"preset_{case['id']}").click().run()
    assert not at.exception
    assert at.text_area(key="review_text").value == case["text"]
    assert at.session_state["single_result"]["label"] == case["baseline_pred"]
    assert at.session_state["single_result"]["label"] != case["expected_label"]


def test_uploaded_csv_runs_end_to_end_and_keeps_empty_rows():
    frame = pd.DataFrame({"row_id": [10, 11, 12], "comment": [
        "Sản phẩm rất tốt giao hàng nhanh", "", "Hàng quá tệ, chất lượng kém không giống hình",
    ]})
    payload = frame.to_csv(index=False).encode("utf-8-sig")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.file_uploader(key="csv_upload").upload("reviews.csv", payload, "text/csv").run()
    assert not at.exception
    assert not at.button(key="analyze_csv").disabled
    at.button(key="analyze_csv").click().run()
    assert not at.exception
    result = at.session_state["batch_result"]
    assert result["row_id"].tolist() == [10, 11, 12]
    assert result["analysis_status"].tolist() == ["ok", "skipped_empty", "ok"]
    assert result["predicted_label"].tolist() == ["POS", "", "NEG"]
    exported = result.to_csv(index=False).encode("utf-8-sig")
    recovered = pd.read_csv(io.BytesIO(exported), keep_default_na=False)
    assert recovered["comment"].tolist() == frame["comment"].tolist()
    assert any("Hoàn tất: 2/3" in element.value for element in at.success)


def test_invalid_upload_is_explained_and_previous_result_is_cleared():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.radio(key="csv_mode").set_value("Thử dữ liệu mẫu").run()
    at.button(key="analyze_csv").click().run()
    assert "batch_result" in at.session_state
    at.radio(key="csv_mode").set_value("Tải tệp CSV").run()
    at.file_uploader(key="csv_upload").upload("bad.csv", b"wrong_column\nhello\n", "text/csv").run()
    assert not at.exception
    assert any("cột tên chính xác là comment" in error.value for error in at.error)
    assert at.button(key="analyze_csv").disabled
    assert "batch_result" not in at.session_state


def test_csv_preserves_existing_output_columns_and_emoji_only_rows():
    original = pd.DataFrame({"comment": ["😍😍", "Giao hàng nhanh sản phẩm rất tốt"], "predicted_label": ["original", "keep"]})
    result = analyze_dataframe(original)
    assert result["predicted_label"].tolist() == ["original", "keep"]
    assert result["ai_predicted_label"].tolist() == ["", "POS"]
    assert result["analysis_status"].tolist() == ["skipped_empty", "ok"]


@pytest.mark.parametrize("payload", [b"", b"comment\n", b"other\ntext\n"])
def test_unusable_csv_is_rejected(payload):
    with pytest.raises(ValueError):
        read_csv_upload(payload)


def test_highlight_never_executes_user_html():
    rendered = highlighted_original('<script>alert("x")</script> sp đẹp', [{"original": "sp"}])
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<mark>sp</mark>" in rendered
