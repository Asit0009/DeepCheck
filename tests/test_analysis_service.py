import io

import pytest
from PIL import Image

from src.data.imageops import InvalidImageError
from src.models.predictor import Predictor
from src.services.analysis import AnalysisResult, InferenceError, analyze_upload
from src.storage.history import HistoryStore
from tests.conftest import make_jpeg, make_png


@pytest.fixture()
def predictor(tiny_artifact_factory):
    return Predictor(tiny_artifact_factory(model_id="svc", final_bias=3.0, calibrated=True))


def test_successful_analysis_is_saved_and_reopenable(predictor, tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    result = analyze_upload(make_png(), "my photo.png", predictor, store)
    assert result.saved_to_history and result.id and result.decision in {"likely_ai", "likely_real", "inconclusive"}
    assert result.score_is_calibrated and 0 <= result.score <= 1 and result.model_id == "svc"
    reopened = AnalysisResult.from_record(store.get(result.id))
    assert (reopened.decision, reopened.score, reopened.filename) == (result.decision, result.score, "my photo.png")
    assert reopened.image_sha256 == result.image_sha256 and reopened.width == 300


def test_thumbnail_is_small_jpeg_and_original_is_not_stored(predictor, tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    big = make_jpeg(size=(1200, 900))
    result = analyze_upload(big, "big.jpg", predictor, store)
    thumb = Image.open(io.BytesIO(store.get(result.id).thumbnail))
    assert thumb.format == "JPEG" and max(thumb.size) <= 160
    db_bytes = (tmp_path / "h.db").read_bytes()
    assert big not in db_bytes and len(db_bytes) < len(big)                    # the upload itself never lands in the database


def test_thumbnail_can_be_disabled(predictor, tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    result = analyze_upload(make_png(), "a.png", predictor, store, keep_thumbnail=False)
    assert store.get(result.id).thumbnail is None


@pytest.mark.parametrize("payload", [b"", b"junk" * 100, make_png()[:300]])
def test_invalid_uploads_raise_and_leave_history_untouched(predictor, tmp_path, payload):
    store = HistoryStore(tmp_path / "h.db")
    with pytest.raises(InvalidImageError):
        analyze_upload(payload, "bad.png", predictor, store)
    assert store.count() == 0


def test_model_failure_never_produces_a_result_or_history_row(predictor, tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / "h.db")

    def boom(_img):
        raise RuntimeError("weights exploded")

    monkeypatch.setattr(predictor, "predict_image", boom)
    with pytest.raises(InferenceError, match="could not analyse"):
        analyze_upload(make_png(), "a.png", predictor, store)
    assert store.count() == 0


def test_without_a_store_nothing_is_saved(predictor):
    result = analyze_upload(make_png(), "a.png", predictor, store=None)
    assert result.id is None and result.saved_to_history is False


def test_same_image_twice_gives_same_score_and_two_history_rows(predictor, tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    data = make_png(seed=7)
    a, b = analyze_upload(data, "x.png", predictor, store), analyze_upload(data, "x.png", predictor, store)
    assert a.score == pytest.approx(b.score) and a.image_sha256 == b.image_sha256 and store.count() == 2
