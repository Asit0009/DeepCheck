from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import psutil
from PIL import Image

from src import config
from src.data import imageops
from src.models.predictor import Predictor
from src.training.dataset import load_splits

LOG = logging.getLogger("run_eval")


def prediction_path(model_id: str):
    return config.REPORTS_DIR / "predictions" / f"{model_id}.csv"  # path to output predictions file


def load_predictions(model_id: str) -> pd.DataFrame:
    path = prediction_path(model_id)
    if path.is_file():
        return pd.read_csv(path)  # load predictions table
    return pd.DataFrame(columns=["record_id", "split", "generator", "label", "logit"])


def run_splits(predictor: Predictor, splits: list[str], limit: int | None = None, batch: int = 16) -> None:
    frame = load_splits()
    existing = load_predictions(predictor.model_id)
    done = set(existing["record_id"])  # skip already evaluated rows
    path = prediction_path(predictor.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    for split in splits:
        rows = frame[(frame["split"] == split) & ~frame["record_id"].isin(done)]
        if limit:
            rows = rows.sample(n=min(limit, len(rows)), random_state=0)
        LOG.info("%s: %d images to score (%d already done)", split, len(rows), int((frame["split"] == split).sum()) - len(rows))
        t0 = time.time()
        for start in range(0, len(rows), batch):
            chunk = rows.iloc[start : start + batch]
            canvases = []
            for rel in chunk["processed_file"]:
                with Image.open(config.PROCESSED_ROOT / rel) as im:
                    canvases.append(im.convert("RGB"))
            logits = predictor.logits_for_canvases(canvases, batch_size=batch)  # compute logits
            out = chunk[["record_id", "split", "generator", "label"]].copy()
            out["logit"] = logits
            out.to_csv(path, mode="a", header=not path.is_file(), index=False)  # append to csv
            if (start // batch) % 10 == 9 or start + batch >= len(rows):
                n = min(start + batch, len(rows))
                LOG.info("  %s %d/%d (%.1f img/s)", split, n, len(rows), n / (time.time() - t0))


def measure_latency(predictor: Predictor, n: int = 30) -> dict:
    frame = load_splits()
    sample = frame[frame["split"] == "test"].sample(n=n, random_state=0)
    blobs = [(config.DATA_DIR / r).read_bytes() for r in sample["raw_file"]]
    predictor.predict_image(imageops.inspect_bytes(blobs[0])[0])  # warmup inference call
    proc = psutil.Process()
    times = []
    for data in blobs:
        t0 = time.perf_counter()
        img, _ = imageops.inspect_bytes(data)
        predictor.predict_image(img)  # measure single image latency
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "model_id": predictor.model_id,
        "n_images": n,
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "process_rss_mb": round(proc.memory_info().rss / 1e6, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--splits", nargs="*", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--latency", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    predictor = Predictor(config.ARTIFACTS_DIR / args.model_id, threads=args.threads, device=args.device)
    if args.splits:
        run_splits(predictor, args.splits, args.limit)  # score splits
    if args.latency:
        result = measure_latency(predictor)  # record inference speed
        out = config.REPORTS_DIR / args.model_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "latency.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOG.info("latency: %s", result)


if __name__ == "__main__":
    main()
