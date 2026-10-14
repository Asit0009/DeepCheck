from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))  # 95% confidence interval


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")  # avoid zero division


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)  # threshold predictions
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_ai, n_real = tp + fn, tn + fp
    recall_ai, recall_real = _safe_div(tp, n_ai), _safe_div(tn, n_real)
    precision_ai, precision_real = _safe_div(tp, tp + fp), _safe_div(tn, tn + fn)

    def f1(prec: float, rec: float) -> float:
        return _safe_div(2 * prec * rec, prec + rec) if not (math.isnan(prec) or math.isnan(rec)) else float("nan")

    f1_ai, f1_real = f1(precision_ai, recall_ai), f1(precision_real, recall_real)
    both = n_ai > 0 and n_real > 0
    lo, hi = wilson_interval(tp + tn, tp + tn + fp + fn)

    return {
        "n": int(len(y)),
        "n_real": int(n_real),
        "n_ai": int(n_ai),
        "threshold": float(threshold),
        "accuracy": _safe_div(tp + tn, len(y)),
        "accuracy_ci95": [lo, hi],
        "balanced_accuracy": float(np.nanmean([recall_ai, recall_real])),
        "precision_ai": precision_ai,
        "recall_ai": recall_ai,
        "f1_ai": f1_ai,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "f1_real": f1_real,
        "macro_f1": float(np.nanmean([f1_ai, f1_real])),
        "false_positive_rate": _safe_div(fp, n_real),
        "false_negative_rate": _safe_div(fn, n_ai),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if both else float("nan"),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def best_balanced_threshold(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    best_t, best = 0.5, -1.0
    for t in np.unique(np.round(p, 4)):
        pred = p >= t
        bal = 0.5 * (pred[y == 1].mean() + (~pred)[y == 0].mean())
        if bal > best + 1e-12:
            best, best_t = bal, float(t)  # update best threshold
    return best_t


def three_way_summary(y: np.ndarray, decisions: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    decided = decisions != "inconclusive"
    correct = ((decisions == "likely_ai") & (y == 1)) | ((decisions == "likely_real") & (y == 0))
    return {
        "coverage": float(decided.mean()),
        "inconclusive_rate": float(1 - decided.mean()),
        "inconclusive_rate_real": float((decisions[y == 0] == "inconclusive").mean()) if (y == 0).any() else float("nan"),
        "inconclusive_rate_ai": float((decisions[y == 1] == "inconclusive").mean()) if (y == 1).any() else float("nan"),
        "accuracy_when_decided": _safe_div(int(correct[decided].sum()), int(decided.sum())),
        "ai_decisions": int((decisions == "likely_ai").sum()),
        "ai_decision_precision": _safe_div(int(((decisions == "likely_ai") & (y == 1)).sum()), int((decisions == "likely_ai").sum())),
        "real_decisions": int((decisions == "likely_real").sum()),
        "real_decision_precision": _safe_div(int(((decisions == "likely_real") & (y == 0)).sum()), int((decisions == "likely_real").sum())),
    }
