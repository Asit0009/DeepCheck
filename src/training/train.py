from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src import config
from src.models import preprocess
from src.models.factory import create_model, load_models_config, pretrained_normalisation, save_artifact
from src.training.dataset import SplitDataset, load_splits

LOG = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix random seed for reproducibility


def make_batches(n_items: int, batch_size: int, total_steps: int, seed: int) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    pass_id, pool = 0, np.array([], dtype=int)
    while len(batches) < total_steps:
        if len(pool) < batch_size:
            pool = np.concatenate([pool, np.random.default_rng(seed + pass_id).permutation(n_items)])
            pass_id += 1
        batches.append(pool[:batch_size])
        pool = pool[batch_size:]
    return batches


def evaluate(model, loader, device) -> dict:
    model.eval()  # validation evaluation loop
    logits, labels = [], []
    with torch.inference_mode():
        for x, y, _ in loader:
            logits.append(model(x.to(device).to(memory_format=torch.channels_last)).squeeze(1).float().cpu())
            labels.append(y)
    z, y = torch.cat(logits).numpy(), torch.cat(labels).numpy()
    loss = float(torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(z), torch.from_numpy(y)))
    pred = (z > 0).astype(int)
    tpr = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    tnr = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return {
        "val_loss": loss,
        "val_auc": float(roc_auc_score(y, z)),
        "val_balanced_acc": (tpr + tnr) / 2,
        "val_tpr": tpr,
        "val_tnr": tnr,
    }


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup  # linear warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key in configs/models.yaml")
    parser.add_argument("--model-id", required=True, help="artifact folder name")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_models_config()
    spec = cfg["models"][args.model]
    hp = {**cfg["training"]["default"], **cfg["training"].get(args.model, {})}
    for key, value in (
        ("epochs", args.epochs),
        ("samples_per_epoch", args.samples_per_epoch),
        ("batch_size", args.batch_size),
        ("lr", args.lr),
        ("head_lr", args.head_lr),
        ("eval_every_steps", args.eval_every),
    ):
        if value is not None:
            hp[key] = value
    if args.smoke:
        hp.update(epochs=1, samples_per_epoch=32, batch_size=8, eval_every_steps=2, val_subset=16)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    seed = cfg["common"]["seed"]
    set_seed(seed)
    torch.set_num_threads(args.threads)
    out_dir = config.ARTIFACTS_DIR / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.load_yaml(config.DATA_CONFIG_PATH)
    unseen = set(data_cfg["generators"]["unseen"])
    frame = load_splits()
    train_df = frame[frame["split"] == "train"].reset_index(drop=True)
    leaked = unseen & set(train_df["generator"])
    assert not leaked, f"unseen generators leaked into training: {leaked}"
    val_df = frame[frame["split"] == "validation"]
    val_df = val_df.sample(n=min(hp["val_subset"], len(val_df)), random_state=seed)
    LOG.info("device %s | train images: %d | validation subset: %d", device, len(train_df), len(val_df))

    model = create_model(spec["timm_name"], pretrained=True)
    mean, std = pretrained_normalisation(model)
    model = model.to(device).to(memory_format=torch.channels_last)
    if args.grad_ckpt:
        model.set_grad_checkpointing(True)

    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": body_params, "lr": hp["lr"]}, {"params": head_params, "lr": hp["head_lr"]}],
        weight_decay=hp["weight_decay"],
    )
    base_lrs = [hp["lr"], hp["head_lr"]]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    aug = hp["augmentation"]
    train_tf = preprocess.TrainTransform(
        rescale_prob=aug["rescale_prob"],
        jpeg_prob=aug["jpeg_prob"],
        blur_prob=aug["blur_prob"],
        flip=aug["flip"],
        mean=mean,
        std=std,
    )
    def val_tf(canvas): return preprocess.eval_tensor(canvas, mean, std)
    val_loader = DataLoader(SplitDataset(val_df, val_tf), batch_size=16, shuffle=False, num_workers=0)

    steps_per_epoch = hp["samples_per_epoch"] // hp["batch_size"]
    total_steps = steps_per_epoch * hp["epochs"]
    schedule = make_batches(len(train_df), hp["batch_size"], total_steps, seed)
    state = {"step": 0, "best_auc": -1.0, "history": [], "elapsed": 0.0}
    last_path = out_dir / "last.pt"
    if args.resume and last_path.is_file():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        LOG.info("resumed from step %d", state["step"])

    loader = DataLoader(
        SplitDataset(train_df, train_tf),
        batch_sampler=schedule[state["step"]:],
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    log_file = open(out_dir / "train_log.jsonl", "a", encoding="utf-8")
    t_start = time.time() - state["elapsed"]
    running: list[float] = []
    model.train()  # set training mode

    for x, y, _ in loader:
        if state["step"] >= total_steps:
            break
        x = x.to(device).to(memory_format=torch.channels_last)
        y = y.to(device)
        factor = lr_factor(state["step"], hp["warmup_steps"], total_steps)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        state["step"] += 1
        running.append(float(loss))

        if state["step"] % 10 == 0:
            el = time.time() - t_start
            LOG.info("step %d/%d loss %.4f | %.2f img/s", state["step"], total_steps, np.mean(running[-10:]), state["step"] * hp["batch_size"] / max(el, 1e-6))

        if state["step"] % hp["eval_every_steps"] == 0 or state["step"] == total_steps:
            metrics = evaluate(model, val_loader, device)  # run validation check
            metrics.update(step=state["step"], train_loss=float(np.mean(running[-hp["eval_every_steps"]:])), elapsed_min=round((time.time() - t_start) / 60, 1))
            state["history"].append(metrics)
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()
            if metrics["val_auc"] > state["best_auc"]:
                state["best_auc"] = metrics["val_auc"]
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / "best_state.pt")  # checkpoint best model
            state["elapsed"] = time.time() - t_start
            if not args.smoke:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, last_path)
            model.train()

    model.load_state_dict(torch.load(out_dir / "best_state.pt", map_location=device, weights_only=True))
    split_counts = frame[frame["split"].isin(["train", "validation"])].groupby(["split", "label"]).size()
    metadata = {
        "model_id": args.model_id,
        "architecture": args.model,
        "timm_name": spec["timm_name"],
        "version": "1.0.0",
        "description": spec["description"],
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
            **hp,
            "device": str(device),
            "amp": use_amp,
            "threads": args.threads,
            "seed": seed,
            "total_steps": total_steps,
            "best_validation_auc": state["best_auc"],
            "training_minutes": round(state["elapsed"] / 60, 1),
            "pretrained_weights": spec["timm_name"],
            "train_images_available": int(len(train_df)),
            "images_seen": int(state["step"] * hp["batch_size"]),
        },
        "data": {
            "source": "GenImage (nebula/GenImage-arrow mirror)",
            "license": data_cfg["source"]["license"],
            "familiar_generators": data_cfg["generators"]["familiar"],
            "unseen_generators_never_trained_on": data_cfg["generators"]["unseen"],
            "split_counts": {f"{s}/{'fake' if l else 'real'}": int(c) for (s, l), c in split_counts.items()},
        },
        "validation_history": state["history"],
    }
    if args.smoke:
        return
    save_artifact(model, out_dir, metadata)  # save trained checkpoint
    LOG.info("training complete, artifact saved to %s", out_dir)


if __name__ == "__main__":
    main()
