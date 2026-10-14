import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression

from src.training.linear_head import fold_linear_head


def test_folded_head_reproduces_logistic_regression():
    torch.manual_seed(0)
    model = timm.create_model("convnext_atto", pretrained=False, num_classes=1).eval()
    x = torch.randn(40, 3, 64, 64)
    with torch.inference_mode():
        feats = model.forward_head(model.forward_features(x), pre_logits=True).numpy()
    y = (feats[:, 0] + 0.3 * np.random.default_rng(0).normal(size=40) > feats[:, 0].mean()).astype(int)
    mu, sigma = feats.mean(0), feats.std(0) + 1e-6
    clf = LogisticRegression(C=0.1, max_iter=2000).fit((feats - mu) / sigma, y)
    w, b0 = fold_linear_head(model, clf.coef_[0], float(clf.intercept_[0]), mu, sigma)
    with torch.inference_mode():
        folded = model(x).squeeze(1).numpy()
    assert np.allclose(folded, clf.decision_function((feats - mu) / sigma), atol=1e-4)
    assert np.allclose(folded, feats @ w + b0, atol=1e-4)
