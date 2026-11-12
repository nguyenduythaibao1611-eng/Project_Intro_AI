"""Checkpoint validation, cache isolation and real neural model integration."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import demo_models
from app import _load_neural_cached, analyze_dataframe, analyze_review, neural_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_missing_checkpoint_fails_without_creating_random_model(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_models, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        demo_models.checkpoint_signature("phobert")
    assert demo_models.available_models() == ["baseline"]


def test_changed_checkpoint_rejected_before_deserialization(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "example.pt"
    path.write_bytes(b"old")
    old_signature = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
    path.write_bytes(b"new file")
    with pytest.raises(ValueError, match="đã thay đổi"):
        demo_models.load_neural_runtime("bilstm", old_signature)


def test_switching_checkpoint_paths_cannot_reuse_wrong_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from src.dataset import Vocabulary
    from src.models import BiLSTMClassifier

    monkeypatch.setattr(demo_models, "ROOT", tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(["sản_phẩm tốt tệ"])
    vocab.save(models / "vocab.json")
    net = BiLSTMClassifier(len(vocab), embedding_dim=8, hidden_dim=8, num_layers=1)
    metadata = dict(model_type="bilstm", vocab_size=len(vocab), embedding_dim=8, hidden_dim=8,
                    num_layers=1, dropout_rate=0.3, epoch=1, metrics={})
    signatures = []
    for name, bias in [("force_neg.pt", [20.0, 0, 0]), ("force_pos.pt", [0, 0, 20.0])]:
        with torch.no_grad():
            net.fc.weight.zero_()
            net.fc.bias.copy_(torch.tensor(bias))
        path = models / name
        torch.save({**metadata, "model_state_dict": net.state_dict()}, path)
        signatures.append((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns))
    first = _load_neural_cached("bilstm", signatures[0])
    second = _load_neural_cached("bilstm", signatures[1])
    assert first is not second
    assert demo_models.predict_neural_cleaned(first, ["sản_phẩm tốt"])[0] == ["NEG"]
    assert demo_models.predict_neural_cleaned(second, ["sản_phẩm tốt"])[0] == ["POS"]


def _has_checkpoint(kind):
    return (ROOT / "models" / demo_models.CHECKPOINT_NAMES[kind]).exists()


@pytest.mark.parametrize("kind", ["bilstm", "phobert"])
def test_real_checkpoint_loaded_strictly_and_used_for_predictions(kind):
    if not _has_checkpoint(kind):
        pytest.skip("Local model checkpoint is not part of Git.")
    torch = pytest.importorskip("torch")
    runtime = neural_runtime(kind)
    path = ROOT / "models" / demo_models.CHECKPOINT_NAMES[kind]
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    head = "classifier.weight" if kind == "phobert" else "fc.weight"
    assert torch.equal(runtime.model.state_dict()[head].detach().cpu(), payload["model_state_dict"][head])
    assert not runtime.model.training
    result = analyze_review("Sản phẩm tốt, giao hàng nhanh", kind)
    assert result["model_type"] == kind
    assert result["score_type"] == "probability"
    assert result["metadata"]["epoch"] == payload["epoch"]
    assert sum(result["scores"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["bilstm", "phobert"])
def test_neural_csv_preserves_rows_and_uses_selected_model(kind):
    if not _has_checkpoint(kind):
        pytest.skip("Local model checkpoint is not part of Git.")
    frame = pd.DataFrame({"comment": ["Sản phẩm tốt", "", "Hàng quá tệ"], "row_id": [2, 4, 8]})
    result = analyze_dataframe(frame, model_type=kind)
    assert result.row_id.tolist() == [2, 4, 8]
    assert result.analysis_status.tolist() == ["ok", "skipped_empty", "ok"]
    assert result.model_name.tolist() == [demo_models.MODEL_NAMES[kind]] * 3
    valid = result[result.analysis_status == "ok"]
    np.testing.assert_allclose(valid[["score_NEG", "score_NEU", "score_POS"]].sum(axis=1), 1.0, atol=1e-5)


def test_ui_can_select_both_trained_models():
    if not all(_has_checkpoint(kind) for kind in ("bilstm", "phobert")):
        pytest.skip("Local model checkpoints are not part of Git.")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    for kind in ("phobert", "bilstm"):
        at.selectbox(key="review_model").set_value(kind).run()
        assert not at.exception
        assert at.session_state["single_result"]["model_type"] == kind
        assert at.session_state["single_result"]["metadata"]["epoch"] in (1, 4)
    assert any("12.85%" in message.value for message in at.warning)
