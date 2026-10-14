from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src import config
from src.models.factory import ModelArtifactError
from src.models.predictor import Predictor, default_artifact_dir
from src.services.model_card import load_model_card
from src.storage.history import HistoryStore


def artifact_dir() -> Path:
    override = os.environ.get("DFD_MODEL_ID")  # check for manual model override
    return config.ARTIFACTS_DIR / override if override else default_artifact_dir()


@st.cache_resource(show_spinner="Loading detection model...")
def _load_predictor(path: str) -> Predictor:
    return Predictor(Path(path), threads=min(4, os.cpu_count() or 2))  # load predictor with thread limit


def get_predictor() -> tuple[Predictor | None, str | None]:
    try:
        return _load_predictor(str(artifact_dir())), None  # fetch cached predictor instance
    except ModelArtifactError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Could not load model: {exc}"


@st.cache_resource
def get_store() -> HistoryStore:
    return HistoryStore(config.HISTORY_DB_PATH)  # initialize history db connection


@st.cache_data(show_spinner=False)
def get_model_card(model_id: str) -> dict:
    return load_model_card(model_id)  # read model performance metrics
