from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import config
from src.data import imageops
from src.models import preprocess
from src.models.calibration import Calibrator, DecisionRule
from src.models.factory import CALIBRATION_FILE, ModelArtifactError, load_artifact

LABELS = {"likely_ai": "Likely AI-generated", "likely_real": "Likely real", "inconclusive": "Inconclusive"}


@dataclass
class Prediction:
    decision: str  # likely_ai, likely_real, or inconclusive
    score: float  # probability or raw score between 0 and 1
    score_is_calibrated: bool
    logit: float
    model_id: str
    model_version: str
    latency_ms: float

    @property
    def label(self) -> str:
        return LABELS[self.decision]


class Predictor:
    def __init__(self, artifact_dir: Path, threads: int | None = None, device: str = "cpu"):
        self.artifact_dir = Path(artifact_dir)  # folder where weights and configs live
        self.device = torch.device(device)
        self.model, self.meta = load_artifact(self.artifact_dir)  # load pytorch model and meta
        self.model = self.model.to(self.device).to(memory_format=torch.channels_last)

        if threads:
            torch.set_num_threads(threads)  # limit cpu thread usage

        self.mean = tuple(self.meta["input"]["mean"])  # normalization values
        self.std = tuple(self.meta["input"]["std"])
        self.calibrator = Calibrator("none")
        self.rule = DecisionRule(0.5, 0.5, 0.0, "uncalibrated fallback")
        self.calibration_info: dict = {}

        cal_path = self.artifact_dir / CALIBRATION_FILE  # check for calibration parameters
        if cal_path.is_file():
            try:
                data = json.loads(cal_path.read_text(encoding="utf-8"))
                self.calibrator = Calibrator.from_dict(data["calibrator"])
                self.rule = DecisionRule.from_dict(data["decision_rule"])
                self.calibration_info = data
            except (KeyError, json.JSONDecodeError) as exc:
                raise ModelArtifactError(f"calibration.json is invalid: {exc}") from exc

        self._lock = threading.Lock()  # lock for thread safety

    @property
    def model_id(self) -> str:
        return self.meta["model_id"]

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator.is_calibrated

    def logits_for_canvases(self, canvases: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        out = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(canvases), batch_size):
                chunk = canvases[i:i + batch_size]
                x = torch.stack([preprocess.eval_tensor(c, self.mean, self.std) for c in chunk])
                x = x.to(self.device).to(memory_format=torch.channels_last)
                out.append(self.model(x).squeeze(1).float().cpu().numpy())  # run forward pass
        return np.concatenate(out) if out else np.zeros(0)

    def predict_image(self, img: Image.Image) -> Prediction:
        t0 = time.perf_counter()  # start timer
        canvas = imageops.canonicalize(img)  # preprocess input image to 256x256
        logit = float(self.logits_for_canvases([canvas])[0])  # get network output
        score = float(self.calibrator.apply(np.array([logit]))[0])  # apply calibration

        return Prediction(
            decision=self.rule.decide(score),
            score=score,
            score_is_calibrated=self.is_calibrated,
            logit=logit,
            model_id=self.model_id,
            model_version=self.meta.get("version", "?"),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def default_artifact_dir() -> Path:
    pointer = config.ARTIFACTS_DIR / "DEFAULT_MODEL"  # read pointer file
    if pointer.is_file():
        return config.ARTIFACTS_DIR / pointer.read_text(encoding="utf-8").strip()
    raise ModelArtifactError("No default model has been selected yet (artifacts/DEFAULT_MODEL is missing).")
