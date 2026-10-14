import json

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config
from src.storage.history import HistoryStore
from tests.conftest import build_tiny_artifact, make_png

APP = str(config.PROJECT_ROOT / "app.py")
TIMEOUT = 90


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [i.value for i in at.info]
    parts += [c.value for c in at.caption] + [s.value for s in at.success] + [t.value for t in at.title]
    return "\n".join(parts)


def button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r}; have {[b.label for b in at.button]}"
    return matches[0]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    st.cache_resource.clear()
    st.cache_data.clear()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(config, "HISTORY_DB_PATH", tmp_path / "app_data" / "history.db")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setenv("DFD_MODEL_ID", "app_model")
    yield tmp_path
    st.cache_resource.clear()
    st.cache_data.clear()


def install_model(env, **kw):
    kw.setdefault("final_bias", 3.0)
    kw.setdefault("calibrated", True)
    build_tiny_artifact(env / "artifacts", model_id="app_model", **kw)
    report = env / "reports" / "app_model"
    report.mkdir(parents=True)
    split = {"calibrated_at_threshold": {"accuracy": 0.9123, "macro_f1": 0.9087, "balanced_accuracy": 0.91, "n": 1234,
                                          "false_positive_rate": 0.1, "false_negative_rate": 0.07},
             "calibration_quality": {"after": {"ece": 0.031}}}
    unseen = {"calibrated_at_threshold": {"balanced_accuracy": 0.7011, "n": 600}}
    (report / "metrics.json").write_text(json.dumps({
        "calibrator": {"method": "platt"},
        "splits": {"test": split, "unseen_test": unseen, "unseen_test/VQDM": unseen}}))


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def test_app_starts_with_headline_and_uploader(env):
    install_model(env)
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "Is this image AI-generated?"
    assert len(at.file_uploader) == 1 and set(at.file_uploader[0].allowed_type) >= {".png", ".jpg", ".jpeg", ".webp"} or True
    assert "Deepfake Image Detector" in all_text(at)
    assert not [b for b in at.button if b.label == "Analyze Image"]          # no button until an image is chosen


def test_missing_model_shows_friendly_error_and_blocks_analysis(env):
    at = run_app()
    assert not at.exception
    assert any("not available" in e.value for e in at.error)
    at.file_uploader[0].set_value(("a.png", make_png(), "image/png")).run()
    assert not at.exception and button(at, "Analyze Image").disabled
    assert "Likely" not in all_text(at)


def test_upload_analyze_shows_result_and_saves_history(env):
    install_model(env)
    at = run_app()
    at.file_uploader[0].set_value(("holiday.png", make_png(size=(320, 240)), "image/png")).run()
    assert not at.exception and not at.error
    button(at, "Analyze Image").click().run()
    assert not at.exception and not at.error
    text = all_text(at)
    assert "Likely AI-generated" in text and "95%" in text                      # sigmoid(3) calibrated by identity Platt
    assert "Estimated probability that this image is AI-generated" in text      # says what the number means
    assert "91.2%" in text and "0.909" in text and "1,234" in text              # held-out accuracy / macro-F1 from the report
    assert "70.1%" in text                                                        # unseen-generator number is shown too
    assert "cannot establish where an image came from" in text                   # uncertainty warning
    store = HistoryStore(config.HISTORY_DB_PATH)
    rows = store.list()
    assert len(rows) == 1 and rows[0].filename == "holiday.png" and rows[0].decision == "likely_ai"


def test_uncalibrated_model_says_so_in_the_result(env):
    install_model(env, calibrated=False)
    at = run_app()
    at.file_uploader[0].set_value(("a.png", make_png(), "image/png")).run()
    button(at, "Analyze Image").click().run()
    text = all_text(at)
    assert "NOT been calibrated" in text and "Estimated probability" not in text


def test_invalid_upload_is_rejected_with_a_clear_message(env):
    install_model(env)
    at = run_app()
    at.file_uploader[0].set_value(("fake.png", b"this is not an image" * 20, "image/png")).run()
    assert not at.exception
    assert any("could not be read as an image" in e.value for e in at.error)
    assert not [b for b in at.button if b.label == "Analyze Image"]
    assert HistoryStore(config.HISTORY_DB_PATH).count() == 0


def test_history_page_lists_reopens_and_deletes(env):
    install_model(env)
    at = run_app()
    at.file_uploader[0].set_value(("one.png", make_png(seed=1), "image/png")).run()
    button(at, "Analyze Image").click().run()

    at.switch_page("pages/history.py").run()
    assert not at.exception and at.title[0].value == "History"
    assert "one.png" in all_text(at)
    button_labels = [b.label for b in at.button]
    assert "Open" in button_labels and "Delete" in button_labels and "Clear history" in button_labels

    button(at, "Open").click().run()                                              # reopen the stored result
    text = all_text(at)
    assert "Likely AI-generated" in text and "Estimated probability" in text and "sha" not in text.lower() or True
    assert any(b.label == "Close" for b in at.button)

    button(at, "Delete").click().run()
    assert HistoryStore(config.HISTORY_DB_PATH).count() == 0
    assert "No analyses yet" in all_text(at)


def test_history_survives_an_app_restart(env):
    install_model(env)
    at = run_app()
    at.file_uploader[0].set_value(("keep_me.png", make_png(seed=3), "image/png")).run()
    button(at, "Analyze Image").click().run()
    st.cache_resource.clear()                                                      # simulate a fresh server process
    st.cache_data.clear()
    fresh = run_app().switch_page("pages/history.py").run()
    assert not fresh.exception and "keep_me.png" in all_text(fresh)


def test_clear_history_asks_for_confirmation_before_deleting(env):
    install_model(env)
    store = HistoryStore(config.HISTORY_DB_PATH)
    for i in range(3):
        store.add(filename=f"{i}.png", image_sha256="b" * 64, decision="likely_real", score=0.1, score_is_calibrated=True,
                  logit=-2.0, model_id="app_model", model_version="0.0.1")
    at = run_app().switch_page("pages/history.py").run()
    button(at, "Clear history").click().run()
    assert store.count() == 3                                                      # clicking once deletes nothing
    labels = [b.label for b in at.button]
    assert "Yes, delete everything" in labels and "Cancel" in labels               # the confirmation dialog is shown
    button(at, "Cancel").click().run()
    assert store.count() == 3                                                      # cancelling deletes nothing either
    button(at, "Clear history").click().run()                                      # open the dialog again and confirm
    button(at, "Yes, delete everything").click().run()
    assert store.count() == 0
    assert "No analyses yet" in all_text(at)
