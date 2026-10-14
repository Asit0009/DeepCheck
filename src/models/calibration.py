from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6  # small epsilon to avoid log(0)


def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))  # binary cross entropy loss


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))  # mean squared error


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = ids == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())  # weighted bin difference
    return float(ece)


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = ids == b
        out.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "count": int(mask.sum()),
                "mean_score": float(p[mask].mean()) if mask.any() else None,
                "fraction_ai": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return out


@dataclass
class Calibrator:
    method: str  # none, temperature, platt, or isotonic
    params: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, method: str, logits: np.ndarray, y: np.ndarray) -> "Calibrator":
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if method == "none":
            return cls("none", {})
        if method == "temperature":
            res = minimize(lambda t: nll(y, expit(logits / np.exp(t[0]))), x0=[0.0], method="Nelder-Mead")
            return cls("temperature", {"temperature": float(np.exp(res.x[0]))})  # temperature scaling
        if method == "platt":
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(logits.reshape(-1, 1), y.astype(int))
            return cls("platt", {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])})  # platt scaling
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(logits, y)
            return cls("isotonic", {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()})
        raise ValueError(f"unknown calibration method: {method}")

    def apply(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if self.method == "none":
            return expit(logits)  # raw sigmoid
        if self.method == "temperature":
            return expit(logits / self.params["temperature"])
        if self.method == "platt":
            return expit(self.params["a"] * logits + self.params["b"])
        if self.method == "isotonic":
            return np.interp(logits, self.params["x"], self.params["y"])
        raise ValueError(f"unknown calibration method: {self.method}")

    @property
    def is_calibrated(self) -> bool:
        return self.method != "none"

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, data: dict) -> "Calibrator":
        return cls(data["method"], data.get("params", {}))


def choose_calibrator(logits: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> tuple[Calibrator, dict]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    chunks = np.array_split(order, folds)
    scores: dict[str, float] = {}

    for method in ("temperature", "platt", "isotonic"):
        losses = []
        for k in range(folds):
            test_idx = chunks[k]
            train_idx = np.concatenate([chunks[j] for j in range(folds) if j != k])
            cal = Calibrator.fit(method, logits[train_idx], y[train_idx])
            losses.append(nll(y[test_idx], cal.apply(logits[test_idx])))
        scores[method] = float(np.mean(losses))  # average loss across folds

    best = min(scores, key=scores.get)
    return Calibrator.fit(best, logits, y), {"cv_log_loss": scores, "chosen": best}


@dataclass
class DecisionRule:
    ai_threshold: float
    real_threshold: float
    target_precision: float
    note: str = ""

    def decide(self, p: float) -> str:
        if p >= self.ai_threshold:
            return "likely_ai"
        if p <= self.real_threshold:
            return "likely_real"
        return "inconclusive"

    def decide_many(self, p: np.ndarray) -> np.ndarray:
        out = np.full(len(p), "inconclusive", dtype=object)
        out[p >= self.ai_threshold] = "likely_ai"
        out[p <= self.real_threshold] = "likely_real"
        return out

    def to_dict(self) -> dict:
        return {
            "ai_threshold": self.ai_threshold,
            "real_threshold": self.real_threshold,
            "target_precision": self.target_precision,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionRule":
        return cls(data["ai_threshold"], data["real_threshold"], data["target_precision"], data.get("note", ""))


def fit_decision_rule(p: np.ndarray, y: np.ndarray, target_precision: float = 0.90) -> DecisionRule:
    candidates = np.unique(np.round(p, 4))
    ai_t, real_t, note = None, None, []

    for t in candidates:  # find lower bound threshold for ai
        sel = p >= t
        if sel.sum() >= 20 and y[sel].mean() >= target_precision:
            ai_t = float(t)
            break

    for t in candidates[::-1]:  # find upper bound threshold for real
        sel = p <= t
        if sel.sum() >= 20 and (1 - y[sel]).mean() >= target_precision:
            real_t = float(t)
            break

    if ai_t is None:
        ai_t = 0.5
        note.append("defaulted ai threshold to 0.5")
    if real_t is None:
        real_t = 0.5
        note.append("defaulted real threshold to 0.5")

    if real_t >= ai_t:
        mid = (real_t + ai_t) / 2
        real_t = ai_t = float(mid)  # collapse thresholds if overlapping

    return DecisionRule(ai_t, real_t, target_precision, "; ".join(note))
