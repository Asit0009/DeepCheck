from __future__ import annotations

import argparse
import json
import logging

import numpy as np
from scipy.special import expit

from src import config
from src.evaluation.metrics import best_balanced_threshold, classification_metrics, three_way_summary
from src.evaluation.run_eval import load_predictions
from src.models.calibration import (
    Calibrator,
    DecisionRule,
    brier,
    choose_calibrator,
    expected_calibration_error,
    fit_decision_rule,
    nll,
    reliability_bins,
)
from src.models.factory import CALIBRATION_FILE, load_models_config

LOG = logging.getLogger("calibrate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    preds = load_predictions(args.model_id)
    cal = preds[preds["split"] == "calibration"]
    if len(cal) < 200:
        raise SystemExit(f"only {len(cal)} calibration predictions found; run run_eval.py --splits calibration first")

    z, y = cal["logit"].to_numpy(), cal["label"].to_numpy()
    target = load_models_config()["calibration"]["target_precision"]

    raw_p = expit(z)  # raw sigmoid probabilities
    calibrator, selection = choose_calibrator(z, y)  # pick optimal calibrator
    p = calibrator.apply(z)  # apply calibrator to logits
    rule = fit_decision_rule(p, y, target)  # compute decision cutoffs
    threshold = best_balanced_threshold(y, p)  # find optimal binary threshold

    summary = {
        "model_id": args.model_id,
        "fitted_on": {
            "split": "calibration",
            "n": int(len(y)),
            "n_real": int((y == 0).sum()),
            "n_ai": int((y == 1).sum()),
            "generators": sorted(cal["generator"].unique().tolist()),
        },
        "calibrator": calibrator.to_dict(),
        "calibrator_selection": selection,
        "binary_threshold": threshold,
        "decision_rule": rule.to_dict(),
        "calibration_split_metrics": {
            "before": {"ece": expected_calibration_error(y, raw_p), "brier": brier(y, raw_p), "log_loss": nll(y, raw_p)},
            "after": {"ece": expected_calibration_error(y, p), "brier": brier(y, p), "log_loss": nll(y, p)},
        },
        "calibration_split_at_threshold": classification_metrics(y, p, threshold),
        "calibration_split_three_way": three_way_summary(y, rule.decide_many(p)),
        "reliability_bins_after": reliability_bins(y, p),
    }

    out = config.ARTIFACTS_DIR / args.model_id / CALIBRATION_FILE
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")  # save calibration configuration
    LOG.info("calibrator=%s ECE %.4f -> %.4f | binary threshold %.3f", calibrator.method, summary["calibration_split_metrics"]["before"]["ece"], summary["calibration_split_metrics"]["after"]["ece"], threshold)


if __name__ == "__main__":
    main()
