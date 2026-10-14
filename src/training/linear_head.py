from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src import config
from src.models import preprocess
from src.models.factory import create_model, load_models_config, pretrained_normalisation, save_artifact
from src.training.dataset import load_splits, stratified_sample

LOG = logging.getLogger("linear_head")


def extract(
    model,
    mean,
    std,
    frame: pd.DataFrame,
    cache: "config.Path",
    batch: int = 16,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    if cache.is_file():
        feats = np.load(cache)  # load pre-extracted features if present
        if len(feats) == len(frame):
            return feats

    out = []
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, len(frame), batch):
            xs = []
            for rel in frame["processed_file"].iloc[start : start + batch]:
                with Image.open(config.PROCESSED_ROOT / rel) as im:
                    xs.append(preprocess.eval_tensor(im.convert("RGB"), mean, std))
            x = torch.stack(xs).to(device).to(memory_format=torch.channels_last)
            out.append(model.forward_head(model.forward_features(x), pre_logits=True).float().cpu().numpy())
            if (start // batch) % 50 == 0:
                LOG.info("  %s: %d/%d (%.1f img/s)", cache.stem, start, len(frame), (start + len(xs)) / max(time.time() - t0, 1e-6))

    feats = np.concatenate(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, feats)  # cache extracted features to disk
    return feats


def fold_linear_head(model: torch.nn.Module, coef: np.ndarray, intercept: float, mu: np.ndarray, sigma: np.ndarray):
    w = coef / sigma  # scale weights by feature std
    b0 = float(intercept - (coef * mu / sigma).sum())  # adjust bias with mean offset
    fc = model.get_classifier()
    assert tuple(fc.weight.shape) == (1, len(w)), f"unexpected head shape {tuple(fc.weight.shape)}"

    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(w).float().view(1, -1))
        fc.bias.copy_(torch.tensor([b0]).float())  # set folded linear weights
    return w, b0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-per-cell", type=int, default=500, help="train images per (generator, class)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    LOG.info("device %s", device)

    cfg = load_models_config()
    spec = cfg["models"][args.model]
    frame = load_splits()  # read splits manifest
    data_cfg = config.load_yaml(config.DATA_CONFIG_PATH)
    train = stratified_sample(frame[frame["split"] == "train"], ["generator", "label"], args.train_per_cell, seed=0)
    assert set(train["generator"]).isdisjoint(set(data_cfg["generators"]["unseen"])), "unseen generator in training data"

    sets = {
        "train": train,
        "validation": (lambda v: v.sample(n=min(900, len(v)), random_state=0))(frame[frame["split"] == "validation"]),
        "calibration": frame[frame["split"] == "calibration"],
        "test": frame[frame["split"] == "test"],
        "unseen_test": frame[frame["split"] == "unseen_test"],
    }

    model = create_model(spec["timm_name"], pretrained=True).eval().to(device).to(memory_format=torch.channels_last)
    mean, std = pretrained_normalisation(model)
    cache_dir = config.PROCESSED_ROOT / "features" / args.model_id
    feats = {name: extract(model, mean, std, df, cache_dir / f"{name}.npy", args.batch_size, device) for name, df in sets.items()}

    # feature standardisation
    mu, sigma = feats["train"].mean(0), feats["train"].std(0) + 1e-6
    def z(f): return (f - mu) / sigma

    y_train, y_val = sets["train"]["label"].to_numpy(), sets["validation"]["label"].to_numpy()
    best = None
    for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        clf = LogisticRegression(C=C, max_iter=5000).fit(z(feats["train"]), y_train)  # train classifier
        loss = log_loss(y_val, clf.predict_proba(z(feats["validation"]))[:, 1])
        LOG.info("C=%-6g validation log-loss %.4f", C, loss)
        if best is None or loss < best[0]:
            best = (loss, C, clf)

    loss, C, clf = best
    LOG.info("chosen C=%g (validation log-loss %.4f)", C, loss)

    # fold head into model architecture
    w, b0 = fold_linear_head(model, clf.coef_[0], float(clf.intercept_[0]), mu, sigma)

    # export split prediction tables
    pred_rows = []
    for name in ("validation", "calibration", "test", "unseen_test"):
        logits = feats[name] @ w + b0
        out = sets[name][["record_id", "split", "generator", "label"]].copy()
        out["logit"] = logits
        pred_rows.append(out)

    pred_path = config.REPORTS_DIR / "predictions" / f"{args.model_id}.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_rows).to_csv(pred_path, index=False)

    check = sets["validation"].head(48)
    with torch.inference_mode():
        x = torch.stack([preprocess.eval_tensor(Image.open(config.PROCESSED_ROOT / r).convert("RGB"), mean, std) for r in check["processed_file"]])
        folded = model(x.to(device).to(memory_format=torch.channels_last)).squeeze(1).float().cpu().numpy()
    expected = feats["validation"][:48] @ w + b0
    max_err = float(np.abs(folded - expected).max())
    LOG.info("folded-network vs feature-space logits: max abs difference %.2e", max_err)
    assert max_err < 1e-3, "folded head does not reproduce the logistic regression"

    train_acc = float((clf.predict(z(feats["train"])) == y_train).mean())
    metadata = {
        "model_id": args.model_id,
        "architecture": args.model + "_linear",
        "timm_name": spec["timm_name"],
        "version": "1.0.0",
        "description": spec["description"] + " (frozen backbone, linear head fitted by logistic regression)",
        "class_names": cfg["common"]["class_names"],
        "positive_class": cfg["common"]["class_names"][1],
        "output": "one logit; sigmoid(logit) is the (uncalibrated) AI-generated score",
        "input": {
            "canvas_size": cfg["common"]["canvas_size"],
            "crop_size": cfg["common"]["input_size"],
            "mean": list(mean),
            "std": list(std),
            "preprocessing": "EXIF-transpose, RGB, centre square crop, bicubic resize to 256, centre crop 224, normalise",
        },
        "training": {
            "method": "frozen backbone + logistic regression (standardised features), head folded into the final layer",
            "backbone_weights": spec["timm_name"],
            "train_images": int(len(train)),
            "regularisation_C": C,
            "C_chosen_on": "validation split log-loss",
            "validation_log_loss": loss,
            "train_accuracy": train_acc,
            "feature_dim": int(len(w)),
            "augmentation": "none",
            "device": device.type,
        },
        "data": {
            "source": "GenImage (nebula/GenImage-arrow mirror)",
            "license": data_cfg["source"]["license"],
            "familiar_generators": data_cfg["generators"]["familiar"],
            "unseen_generators_never_trained_on": data_cfg["generators"]["unseen"],
        },
    }
    save_artifact(model.cpu(), config.ARTIFACTS_DIR / args.model_id, metadata)  # persist model weights and metadata
    LOG.info("artifact saved; predictions written to %s", pred_path)


if __name__ == "__main__":
    main()
