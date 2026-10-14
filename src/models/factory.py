from __future__ import annotations

import json
import time
from pathlib import Path

import timm
import torch

from src import config

MODEL_FILE = "model.pt"
METADATA_FILE = "metadata.json"
CALIBRATION_FILE = "calibration.json"


class ModelArtifactError(RuntimeError):
    pass


def load_models_config() -> dict:
    return config.load_yaml(config.CONFIGS_DIR / "models.yaml")  # load yaml configuration


def create_model(timm_name: str, pretrained: bool = True) -> torch.nn.Module:
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=1)  # create backbone with 1 output logit
    return model


def pretrained_normalisation(model: torch.nn.Module) -> tuple[tuple, tuple]:
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))


def save_artifact(model: torch.nn.Module, out_dir: Path, metadata: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure destination directory exists
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out_dir / MODEL_FILE)  # save weights
    metadata = {**metadata, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")  # save metadata
    return out_dir


def read_metadata(artifact_dir: Path) -> dict:
    path = Path(artifact_dir) / METADATA_FILE
    if not path.is_file():
        raise ModelArtifactError(f"Model metadata not found at {path}")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelArtifactError(f"Model metadata at {path} is not valid JSON") from exc

    required = {"model_id", "architecture", "timm_name", "class_names", "input"}
    missing = required - meta.keys()
    if missing:
        raise ModelArtifactError(f"Model metadata is missing fields: {sorted(missing)}")
    if list(meta["class_names"]) != list(load_models_config()["common"]["class_names"]):
        raise ModelArtifactError("Model class order does not match expected configuration.")
    return meta


def load_artifact(artifact_dir: Path) -> tuple[torch.nn.Module, dict]:
    artifact_dir = Path(artifact_dir)
    meta = read_metadata(artifact_dir)  # read config metadata
    weights = artifact_dir / MODEL_FILE  # check for weights file
    if not weights.is_file():
        raise ModelArtifactError(f"Model weights not found at {weights}")
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)  # load tensor dictionary
        model = create_model(meta["timm_name"], pretrained=False)  # instantiate architecture
        model.load_state_dict(state, strict=True)  # apply weights
    except Exception as exc:
        raise ModelArtifactError(f"Could not load model weights: {exc}") from exc
    model.eval()  # set model to evaluation mode
    return model, meta
