import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import imageops
from src.models import preprocess
from src.models.calibration import (Calibrator, DecisionRule, choose_calibrator, expected_calibration_error,
                                    fit_decision_rule, nll)
from src.models.factory import (METADATA_FILE, MODEL_FILE, ModelArtifactError, load_artifact, read_metadata)
from src.models.predictor import Predictor, default_artifact_dir
from tests.conftest import make_png


# ---- preprocessing ------------------------------------------------------------------------------
def test_eval_tensor_shape_and_normalisation():
    canvas = Image.new("RGB", (256, 256), (124, 116, 104))      # close to the ImageNet mean colour
    t = preprocess.eval_tensor(canvas)
    assert t.shape == (3, 224, 224) and t.dtype == torch.float32
    assert float(t.abs().max()) < 0.2                           # mean colour normalises to ~0


def test_eval_tensor_rejects_non_canonical_input():
    with pytest.raises(AssertionError):
        preprocess.eval_tensor(Image.new("RGB", (300, 300)))


def test_train_and_eval_pipelines_produce_same_geometry():
    canvas = Image.new("RGB", (256, 256), (50, 60, 70))
    train = preprocess.TrainTransform()(canvas)
    assert train.shape == preprocess.eval_tensor(canvas).shape


def test_training_augmentation_is_applied_to_pixels():
    rng = np.random.default_rng(0)
    canvas = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    aug = preprocess.TrainTransform(rescale_prob=1.0, jpeg_prob=1.0, blur_prob=1.0)
    outs = {tuple(aug(canvas).flatten()[:5].tolist()) for _ in range(5)}
    assert len(outs) > 1                                        # random, not constant


def test_predictor_preprocessing_matches_manual_pipeline(tiny_artifact):
    predictor = Predictor(tiny_artifact)
    img, _ = imageops.inspect_bytes(make_png(size=(333, 222)))
    canvas = imageops.canonicalize(img)
    manual = preprocess.eval_tensor(canvas, predictor.mean, predictor.std).unsqueeze(0)
    with torch.inference_mode():
        expected = float(predictor.model(manual.to(memory_format=torch.channels_last)).squeeze())
    assert predictor.predict_image(img).logit == pytest.approx(expected, abs=1e-5)


# ---- artifacts ------------------------------------------------------------------------------------
def test_artifact_roundtrip_loads_and_runs(tiny_artifact):
    model, meta = load_artifact(tiny_artifact)
    assert meta["model_id"] == "tiny_test" and meta["class_names"] == ["real", "ai_generated"]
    with torch.inference_mode():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_missing_weights_gives_friendly_error(tiny_artifact):
    (tiny_artifact / MODEL_FILE).unlink()
    with pytest.raises(ModelArtifactError, match="weights not found"):
        Predictor(tiny_artifact)


def test_missing_artifact_folder_gives_friendly_error(tmp_path):
    with pytest.raises(ModelArtifactError):
        Predictor(tmp_path / "does_not_exist")


def test_corrupt_weights_give_friendly_error(tiny_artifact):
    (tiny_artifact / MODEL_FILE).write_bytes(b"not a torch file")
    with pytest.raises(ModelArtifactError, match="Could not load"):
        load_artifact(tiny_artifact)


def test_invalid_metadata_is_rejected(tiny_artifact):
    path = tiny_artifact / METADATA_FILE
    meta = json.loads(path.read_text())
    del meta["timm_name"]
    path.write_text(json.dumps(meta))
    with pytest.raises(ModelArtifactError, match="missing fields"):
        read_metadata(tiny_artifact)
    path.write_text("{ not json")
    with pytest.raises(ModelArtifactError, match="not valid JSON"):
        read_metadata(tiny_artifact)


def test_wrong_class_order_is_rejected(tiny_artifact):
    path = tiny_artifact / METADATA_FILE
    meta = json.loads(path.read_text())
    meta["class_names"] = ["ai_generated", "real"]
    path.write_text(json.dumps(meta))
    with pytest.raises(ModelArtifactError, match="class order"):
        read_metadata(tiny_artifact)


def test_no_default_model_pointer_is_a_friendly_error(tmp_path, monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(cfg, "ARTIFACTS_DIR", tmp_path)
    with pytest.raises(ModelArtifactError, match="No default model"):
        default_artifact_dir()


# ---- inference + class mapping ----------------------------------------------------------------------
def test_positive_logit_means_ai_generated(tiny_artifact_factory):
    pred = Predictor(tiny_artifact_factory(model_id="always_ai", final_bias=8.0))
    img, _ = imageops.inspect_bytes(make_png())
    p = pred.predict_image(img)
    assert p.logit > 0 and p.score > 0.99 and p.decision == "likely_ai" and p.label == "Likely AI-generated"


def test_negative_logit_means_real(tiny_artifact_factory):
    pred = Predictor(tiny_artifact_factory(model_id="always_real", final_bias=-8.0))
    img, _ = imageops.inspect_bytes(make_png())
    p = pred.predict_image(img)
    assert p.logit < 0 and p.score < 0.01 and p.decision == "likely_real" and p.label == "Likely real"


def test_uncalibrated_model_is_labelled_uncalibrated(tiny_artifact):
    pred = Predictor(tiny_artifact)
    img, _ = imageops.inspect_bytes(make_png())
    result = pred.predict_image(img)
    assert result.score_is_calibrated is False and 0.0 <= result.score <= 1.0


def test_calibrated_model_uses_calibration_file_and_three_way_rule(tiny_artifact_factory):
    art = tiny_artifact_factory(model_id="cal", final_bias=0.0, calibrated=True)
    pred = Predictor(art)
    assert pred.is_calibrated
    img, _ = imageops.inspect_bytes(make_png())
    r = pred.predict_image(img)
    assert r.score_is_calibrated and r.decision == "inconclusive"      # score 0.5 sits between 0.2 and 0.8


def test_batch_logits_are_independent_of_batch_size(tiny_artifact):
    pred = Predictor(tiny_artifact)
    canvases = [imageops.canonicalize(imageops.inspect_bytes(make_png(seed=s))[0]) for s in range(5)]
    a = pred.logits_for_canvases(canvases, batch_size=1)
    b = pred.logits_for_canvases(canvases, batch_size=5)
    assert np.allclose(a, b, atol=1e-4)


# ---- calibration ----------------------------------------------------------------------------------------
def _overconfident_logits(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    z = (2 * y - 1) * 1.0 + rng.normal(0, 1.5, n)             # informative but noisy
    return z * 4, y                                            # blown up -> overconfident


def test_temperature_scaling_improves_log_loss_and_ece():
    z, y = _overconfident_logits()
    raw = 1 / (1 + np.exp(-z))
    cal = Calibrator.fit("temperature", z, y)
    assert cal.params["temperature"] > 1.5
    after = cal.apply(z)
    assert nll(y, after) < nll(y, raw) and expected_calibration_error(y, after) < expected_calibration_error(y, raw)


def test_platt_and_isotonic_are_valid_probabilities_and_monotone():
    z, y = _overconfident_logits()
    grid = np.linspace(-20, 20, 200)
    for method in ("platt", "isotonic"):
        p = Calibrator.fit(method, z, y).apply(grid)
        assert p.min() >= 0 and p.max() <= 1 and np.all(np.diff(p) >= -1e-9)


def test_calibrator_survives_json_roundtrip():
    z, y = _overconfident_logits()
    for method in ("temperature", "platt", "isotonic"):
        cal = Calibrator.fit(method, z, y)
        again = Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
        assert np.allclose(cal.apply(z[:50]), again.apply(z[:50]))


def test_choose_calibrator_returns_one_of_the_methods():
    z, y = _overconfident_logits(n=1500)
    cal, info = choose_calibrator(z, y)
    assert cal.method in {"temperature", "platt", "isotonic"} and info["chosen"] == cal.method


def test_decision_rule_hits_target_precision_on_fit_data():
    z, y = _overconfident_logits()
    p = Calibrator.fit("platt", z, y).apply(z)
    rule = fit_decision_rule(p, y, target_precision=0.9)
    assert rule.real_threshold <= rule.ai_threshold
    decisions = rule.decide_many(p)
    ai, real = decisions == "likely_ai", decisions == "likely_real"
    assert y[ai].mean() >= 0.9 - 1e-9 and (1 - y[real]).mean() >= 0.9 - 1e-9
    assert (decisions == "inconclusive").any()


def test_decision_rule_boundaries():
    rule = DecisionRule(0.8, 0.2, 0.9)
    assert [rule.decide(x) for x in (0.0, 0.2, 0.5, 0.8, 1.0)] == [
        "likely_real", "likely_real", "inconclusive", "likely_ai", "likely_ai"]
