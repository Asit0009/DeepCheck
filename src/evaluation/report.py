from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src import config
from src.evaluation.metrics import classification_metrics, three_way_summary
from src.evaluation.run_eval import load_predictions
from src.models.calibration import (
    Calibrator,
    DecisionRule,
    brier,
    expected_calibration_error,
    nll,
    reliability_bins,
)
from src.models.factory import CALIBRATION_FILE, load_models_config, read_metadata

LOG = logging.getLogger("report")
EVAL_SPLITS = ["validation", "test", "unseen_test"]


def per_generator(df, p: np.ndarray, threshold: float) -> dict:
    out = {}
    y = df["label"].to_numpy()
    gens = df["generator"].to_numpy()
    for g in sorted(set(gens)):
        mask = gens == g
        fake, real = mask & (y == 1), mask & (y == 0)
        entry = {"n_ai": int(fake.sum()), "n_real": int(real.sum())}
        if fake.any():
            entry["ai_recall"] = float((p[fake] >= threshold).mean())  # detection recall
        if real.any():
            entry["real_specificity"] = float((p[real] < threshold).mean())
        if fake.any() and real.any():
            entry["balanced_accuracy"] = 0.5 * (entry["ai_recall"] + entry["real_specificity"])
            entry["roc_auc"] = float(roc_auc_score(y[mask], p[mask]))
        out[g] = entry
    return out


def split_report(df, calibrator: Calibrator, rule: DecisionRule, threshold: float) -> dict:
    z, y = df["logit"].to_numpy(), df["label"].to_numpy()
    raw_p, p = expit(z), calibrator.apply(z)
    return {
        "uncalibrated_at_0.5": classification_metrics(y, raw_p, 0.5),
        "calibrated_at_threshold": classification_metrics(y, p, threshold),
        "calibration_quality": {
            "before": {"ece": expected_calibration_error(y, raw_p), "brier": brier(y, raw_p), "log_loss": nll(y, raw_p)},
            "after": {"ece": expected_calibration_error(y, p), "brier": brier(y, p), "log_loss": nll(y, p)},
        },
        "three_way_decision": three_way_summary(y, rule.decide_many(p)),
        "per_generator": per_generator(df, p, threshold),
        "reliability_bins": reliability_bins(y, p),
    }


def draw_confusion(ax, cm: dict, title: str) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(mat, cmap="Blues")  # confusion matrix plot
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(mat[i, j]),
                ha="center",
                va="center",
                color="white" if mat[i, j] > mat.max() / 2 else "black",
                fontsize=13,
            )
    ax.set_xticks([0, 1], ["pred real", "pred AI"])
    ax.set_yticks([0, 1], ["true real", "true AI"])
    ax.set_title(title, fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    art = config.ARTIFACTS_DIR / args.model_id
    meta = read_metadata(art)
    cal_data = json.loads((art / CALIBRATION_FILE).read_text(encoding="utf-8"))
    calibrator, rule = Calibrator.from_dict(cal_data["calibrator"]), DecisionRule.from_dict(cal_data["decision_rule"])
    threshold = cal_data["binary_threshold"]
    preds = load_predictions(args.model_id)
    out_dir = config.REPORTS_DIR / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = load_models_config()["selection"]

    results: dict[str, dict] = {}
    for split in EVAL_SPLITS:
        df = preds[preds["split"] == split]
        if len(df):
            results[split] = split_report(df, calibrator, rule, threshold)  # evaluate each split
    unseen = preds[preds["split"] == "unseen_test"]
    for g in sorted(unseen["generator"].unique()):
        results[f"unseen_test/{g}"] = split_report(unseen[unseen["generator"] == g], calibrator, rule, threshold)

    latency = json.loads((out_dir / "latency.json").read_text()) if (out_dir / "latency.json").is_file() else None
    robustness = json.loads((out_dir / "robustness.json").read_text()) if (out_dir / "robustness.json").is_file() else None
    metrics = {
        "model_id": args.model_id,
        "architecture": meta["architecture"],
        "timm_name": meta["timm_name"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision_threshold_binary": threshold,
        "calibrator": cal_data["calibrator"],
        "decision_rule": cal_data["decision_rule"],
        "calibration_fit": cal_data["fitted_on"],
        "selection_roles": {
            "selection_unseen_generator": sel["selection_unseen_generator"],
            "final_report_unseen_generator": sel["final_report_unseen_generator"],
        },
        "splits": results,
        "latency": latency,
        "robustness_summary_file": "robustness.json" if robustness else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")  # save metrics json

    shown = [s for s in ("test", "unseen_test") if s in results]
    if shown:
        fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 3.6))
        axes = np.atleast_1d(axes)
        for ax, s in zip(axes, shown):
            m = results[s]["calibrated_at_threshold"]
            draw_confusion(ax, m["confusion_matrix"], f"{s.replace('_', ' ')} (n={m['n']})\nbalanced acc {m['balanced_accuracy']:.3f}")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrices.png", dpi=130)  # save confusion matrices plot
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.4, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="ideal")
        for s in shown:
            bins = [b for b in results[s]["reliability_bins"] if b["count"]]
            ax.plot([b["mean_score"] for b in bins], [b["fraction_ai"] for b in bins], "o-", label=s.replace("_", " "))
        ax.set_xlabel("calibrated score")
        ax.set_ylabel("fraction AI")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "reliability.png", dpi=130)  # save reliability diagram
        plt.close(fig)

        gens = {}
        for s in shown:
            for g, e in results[s]["per_generator"].items():
                if "ai_recall" in e:
                    gens[g] = (e["ai_recall"], s)
        if gens:
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            names = sorted(gens)
            ax.bar(names, [gens[g][0] for g in names], color=["#4c8bf5" if gens[g][1] == "test" else "#e0883a" for g in names])
            ax.set_ylim(0, 1)
            ax.set_ylabel("AI recall")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "per_generator_recall.png", dpi=130)  # save bar chart
            plt.close(fig)

    LOG.info("metrics saved to %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
