from __future__ import annotations

import json

from src import config


def load_model_card(model_id: str) -> dict:
    path = config.REPORTS_DIR / model_id / "metrics.json"  # path to evaluation metrics
    if not path.is_file():
        return {"available": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        test = data["splits"]["test"]["calibrated_at_threshold"]
        card = {
            "available": True,
            "model_id": model_id,
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "test_balanced_accuracy": test["balanced_accuracy"],
            "test_n": test["n"],
            "test_fpr": test["false_positive_rate"],
            "test_fnr": test["false_negative_rate"],
            "calibrator": data["calibrator"]["method"],
            "ece_test": data["splits"]["test"]["calibration_quality"]["after"]["ece"],
        }
        if "unseen_test" in data["splits"]:
            unseen = data["splits"]["unseen_test"]["calibrated_at_threshold"]
            card.update(
                unseen_balanced_accuracy=unseen["balanced_accuracy"],
                unseen_n=unseen["n"],
                unseen_generators=sorted(g.split("/", 1)[1] for g in data["splits"] if g.startswith("unseen_test/")),
            )
        return card
    except (KeyError, json.JSONDecodeError):
        return {"available": False}
