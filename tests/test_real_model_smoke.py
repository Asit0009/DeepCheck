import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import imageops
from src.models.factory import ModelArtifactError
from src.models.predictor import Predictor, default_artifact_dir


@pytest.fixture(scope="module")
def real_setup():
    try:
        artifact = default_artifact_dir()
    except ModelArtifactError:
        pytest.skip("no default model selected on this machine")
    splits_csv = config.MANIFEST_DIR / "splits.csv"
    pred_csv = config.REPORTS_DIR / "predictions" / f"{artifact.name}.csv"
    if not artifact.is_dir() or not splits_csv.is_file() or not pred_csv.is_file():
        pytest.skip("real model or dataset manifests not available")
    splits = pd.read_csv(splits_csv).set_index("record_id")
    preds = pd.read_csv(pred_csv).drop_duplicates("record_id").set_index("record_id")
    ids = preds[preds["split"] == "test"].groupby("label").head(3).index.tolist()     # 3 real + 3 AI test images
    if not all((config.DATA_DIR / splits.at[i, "raw_file"]).is_file() for i in ids):
        pytest.skip("raw dataset files not available")
    return Predictor(artifact, threads=2), splits, preds, ids


def test_app_path_matches_stored_evaluation_logits(real_setup):
    predictor, splits, preds, ids = real_setup
    for record_id in ids:
        data = (config.DATA_DIR / splits.at[record_id, "raw_file"]).read_bytes()
        image, _ = imageops.inspect_bytes(data)
        result = predictor.predict_image(image)
        assert result.logit == pytest.approx(preds.at[record_id, "logit"], abs=5e-3), record_id


def test_real_model_outputs_are_valid_and_calibrated(real_setup):
    predictor, splits, _, ids = real_setup
    assert predictor.is_calibrated and predictor.meta["class_names"] == ["real", "ai_generated"]
    for record_id in ids:
        image, _ = imageops.inspect_bytes((config.DATA_DIR / splits.at[record_id, "raw_file"]).read_bytes())
        result = predictor.predict_image(image)
        assert 0.0 <= result.score <= 1.0 and result.decision in {"likely_ai", "likely_real", "inconclusive"}
        assert result.score_is_calibrated and result.model_id == predictor.model_id


def test_real_model_separates_the_samples_better_than_chance(real_setup):
    predictor, splits, preds, ids = real_setup
    scores = []
    for record_id in ids:
        image, _ = imageops.inspect_bytes((config.DATA_DIR / splits.at[record_id, "raw_file"]).read_bytes())
        scores.append(predictor.predict_image(image).score)
    labels = preds.loc[ids, "label"].to_numpy()
    scores = np.array(scores)
    # six hand-picked-by-position images: the mean AI score of the AI images should exceed that of the real ones
    assert scores[labels == 1].mean() > scores[labels == 0].mean()
