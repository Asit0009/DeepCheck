from __future__ import annotations

import argparse
import json
import logging

from src import config
from src.models.factory import load_models_config

LOG = logging.getLogger("compare")


def collect() -> dict[str, dict]:
    out = {}
    for path in sorted(config.REPORTS_DIR.glob("*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))  # load candidate evaluation report
        if "splits" in data and "validation" in data["splits"]:
            out[data["model_id"]] = data
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true", help="write the winner to artifacts/DEFAULT_MODEL")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    sel = load_models_config()["selection"]
    sel_gen, final_gen = sel["selection_unseen_generator"], sel["final_report_unseen_generator"]
    limits = sel["constraints"]

    rows = []
    for model_id, m in collect().items():
        s = m["splits"]
        def get(key, field="balanced_accuracy"):
            return s[key]["calibrated_at_threshold"][field] if key in s else float("nan")

        lat = (m.get("latency") or {}).get("median_ms")
        row = {
            "model_id": model_id,
            "timm_name": m["timm_name"],
            "validation_bacc": get("validation"),
            f"unseen_{sel_gen}_bacc": get(f"unseen_test/{sel_gen}"),
            "selection_score": 0.5 * (get("validation") + get(f"unseen_test/{sel_gen}")),
            "test_accuracy": get("test", "accuracy"),
            "test_macro_f1": get("test", "macro_f1"),
            "test_bacc": get("test"),
            f"unseen_{final_gen}_bacc": get(f"unseen_test/{final_gen}"),
            "unseen_fpr": get("unseen_test", "false_positive_rate"),
            "test_fpr": get("test", "false_positive_rate"),
            "test_fnr": get("test", "false_negative_rate"),
            "ece_test": s["test"]["calibration_quality"]["after"]["ece"] if "test" in s else float("nan"),
            "latency_ms": lat,
        }
        problems = []
        if lat is not None and lat > limits["max_single_image_latency_seconds"] * 1000:
            problems.append("too slow")
        if row["unseen_fpr"] > limits["max_false_positive_rate_unseen"]:
            problems.append("flags unseen real photos")
        row["constraint_violations"] = problems
        rows.append(row)

    ranked = sorted([r for r in rows if not r["constraint_violations"]], key=lambda r: -r["selection_score"])
    winner = ranked[0]["model_id"] if ranked else None

    payload = {"rule": sel, "candidates": rows, "ranking": [r["model_id"] for r in ranked], "selected": winner}
    (config.REPORTS_DIR / "model_comparison.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")  # save comparison json

    if args.select and winner:
        (config.ARTIFACTS_DIR / "DEFAULT_MODEL").write_text(winner + "\n", encoding="utf-8")  # update default model
        LOG.info("selected default model: %s", winner)


if __name__ == "__main__":
    main()
