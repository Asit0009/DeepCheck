import io
import json

import numpy as np
import pytest
import timm
import torch
from PIL import Image

from src.models.factory import (CALIBRATION_FILE, load_models_config, save_artifact)

TINY_ARCH = "mobilenetv3_small_050"


def make_png(size=(300, 200), color=(120, 80, 200), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.clip(np.array(color) + rng.normal(0, 20, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg(size=(300, 200), seed=1, quality=85) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_webp(size=(128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((size[1], size[0], 3), 90, dtype=np.uint8)).save(buf, format="WEBP")
    return buf.getvalue()


@pytest.fixture()
def png_bytes():
    return make_png()


def build_tiny_artifact(root, model_id="tiny_test", final_bias=None, calibrated=False, seed=0):
    torch.manual_seed(seed)
    model = timm.create_model(TINY_ARCH, pretrained=False, num_classes=1)
    if final_bias is not None:
        with torch.no_grad():
            model.get_classifier().bias.fill_(final_bias)
            model.get_classifier().weight.zero_()
    cfg = load_models_config()
    meta = {
        "model_id": model_id, "architecture": "tiny", "timm_name": TINY_ARCH, "version": "0.0.1",
        "class_names": cfg["common"]["class_names"],
        "input": {"canvas_size": 256, "crop_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    art = root / model_id
    save_artifact(model, art, meta)
    if calibrated:
        (art / CALIBRATION_FILE).write_text(json.dumps({
            "calibrator": {"method": "platt", "params": {"a": 1.0, "b": 0.0}},
            "decision_rule": {"ai_threshold": 0.8, "real_threshold": 0.2, "target_precision": 0.9, "note": ""},
            "binary_threshold": 0.5, "fitted_on": {"split": "calibration", "n": 10}}), encoding="utf-8")
    return art


@pytest.fixture()
def tiny_artifact(tmp_path):
    return build_tiny_artifact(tmp_path)


@pytest.fixture()
def tiny_artifact_factory(tmp_path):
    def factory(**kwargs):
        return build_tiny_artifact(tmp_path, **kwargs)
    return factory
