from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from src import config
from src.evaluation.metrics import classification_metrics
from src.models.preprocess import jpeg_roundtrip
from src.models.predictor import Predictor
from src.training.dataset import load_splits, stratified_sample

LOG = logging.getLogger("robustness")


def downscale_then_restore(scale: float):
    def apply(canvas: Image.Image) -> Image.Image:
        small = canvas.resize((max(64, round(256 * scale)),) * 2, Image.BICUBIC)  # downscale
        return small.resize((256, 256), Image.BICUBIC)  # restore back
    return apply


CONDITIONS = {
    "clean": lambda c: c,
    "jpeg_q90": lambda c: jpeg_roundtrip(c, 90),
    "jpeg_q70": lambda c: jpeg_roundtrip(c, 70),
    "jpeg_q50": lambda c: jpeg_roundtrip(c, 50),
    "resize_0.75x": downscale_then_restore(0.75),
    "resize_0.5x": downscale_then_restore(0.5),
    "blur_sigma1": lambda c: c.filter(ImageFilter.GaussianBlur(1.0)),
    "blur_sigma2": lambda c: c.filter(ImageFilter.GaussianBlur(2.0)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--per-cell", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    predictor = Predictor(config.ARTIFACTS_DIR / args.model_id, threads=args.threads, device=args.device)
    cal = json.loads((config.ARTIFACTS_DIR / args.model_id / "calibration.json").read_text())
    threshold = cal["binary_threshold"]
    frame = load_splits()
    pool = frame[frame["split"].isin(["test", "unseen_test"])]
    subset = stratified_sample(pool, ["generator", "label"], args.per_cell, seed=0)
    LOG.info("%d images, %d conditions", len(subset), len(CONDITIONS))

    canvases = []
    for rel in subset["processed_file"]:
        with Image.open(config.PROCESSED_ROOT / rel) as im:
            canvases.append(im.convert("RGB"))  # load images
    y = subset["label"].to_numpy()
    familiar = (subset["split"] == "test").to_numpy()

    results = {}
    for name, fn in CONDITIONS.items():
        logits = predictor.logits_for_canvases([fn(c) for c in canvases])  # score degraded images
        p = predictor.calibrator.apply(logits)
        entry = {
            "all": classification_metrics(y, p, threshold),
            "familiar_generators": classification_metrics(y[familiar], p[familiar], threshold),
            "unseen_generators": classification_metrics(y[~familiar], p[~familiar], threshold),
        }
        results[name] = entry
        LOG.info("%-14s bAcc %.3f AUC %.3f", name, entry["all"]["balanced_accuracy"], entry["all"]["roc_auc"])

    out_dir = config.REPORTS_DIR / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": args.model_id, "n_images": int(len(subset)), "per_cell": args.per_cell, "threshold": threshold, "conditions": results}
    (out_dir / "robustness.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")  # save results
    LOG.info("robustness report saved")


if __name__ == "__main__":
    main()
