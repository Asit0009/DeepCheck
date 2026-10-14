from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.models import preprocess
from src.models.factory import create_model, load_models_config, pretrained_normalisation
from src.training.dataset import load_splits

LOG = logging.getLogger("linear_probe")


def extract(model, mean, std, frame: pd.DataFrame, batch: int = 16) -> np.ndarray:
    feats = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch):
            xs = []
            for rel in frame["processed_file"].iloc[start : start + batch]:
                with Image.open(config.PROCESSED_ROOT / rel) as im:
                    xs.append(preprocess.eval_tensor(im.convert("RGB"), mean, std))
            x = torch.stack(xs).to(memory_format=torch.channels_last)
            feats.append(model.forward_head(model.forward_features(x), pre_logits=True).float().numpy())  # extract features
            if (start // batch) % 40 == 0:
                LOG.info("  features %d/%d", start, len(frame))
    return np.concatenate(feats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--train-n", type=int, default=1800)
    parser.add_argument("--val-n", type=int, default=600)
    parser.add_argument("--unseen-n", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(args.threads)

    cfg = load_models_config()
    sel_gen = cfg["selection"]["selection_unseen_generator"]
    frame = load_splits()
    train = frame[frame["split"] == "train"].groupby("label", group_keys=False).sample(n=args.train_n // 2, random_state=0)
    val = frame[frame["split"] == "validation"].groupby("label", group_keys=False).sample(n=args.val_n // 2, random_state=0)
    unseen = frame[(frame["split"] == "unseen_test") & (frame["generator"] == sel_gen)]
    unseen = unseen.groupby("label", group_keys=False).sample(
        n=min(args.unseen_n // 2, int(unseen["label"].value_counts().min())), random_state=0
    )
    assert set(unseen["generator"]) == {sel_gen}

    results = {}
    for key in args.models:
        spec = cfg["models"][key]
        LOG.info("== %s (%s)", key, spec["timm_name"])
        model = create_model(spec["timm_name"], pretrained=True).eval().to(memory_format=torch.channels_last)
        mean, std = pretrained_normalisation(model)
        t0 = time.time()
        f_train, f_val, f_unseen = (extract(model, mean, std, df) for df in (train, val, unseen))

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        clf.fit(f_train, train["label"])  # fit probe classifier

        row = {"timm_name": spec["timm_name"], "feature_dim": int(f_train.shape[1]), "minutes": round((time.time() - t0) / 60, 1)}
        for name, feats, df in (("validation", f_val, val), (f"unseen_{sel_gen}", f_unseen, unseen)):
            score = clf.predict_proba(feats)[:, 1]
            row[name] = {
                "balanced_accuracy": float(balanced_accuracy_score(df["label"], score > 0.5)),
                "roc_auc": float(roc_auc_score(df["label"], score)),
                "n": int(len(df)),
            }
        row["mean_balanced_accuracy"] = 0.5 * (row["validation"]["balanced_accuracy"] + row[f"unseen_{sel_gen}"]["balanced_accuracy"])
        results[key] = row
        LOG.info("%s -> %s", key, json.dumps(row))
        del model

    out = {
        "protocol": f"frozen features + logistic regression, evaluated on validation and {sel_gen}",
        "train_images": int(len(train)),
        "results": results,
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "linear_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")  # save probe results


if __name__ == "__main__":
    main()
