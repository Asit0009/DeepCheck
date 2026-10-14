from __future__ import annotations

from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src import config


def stratified_sample(frame: pd.DataFrame, keys: list[str], n: int, seed: int = 0) -> pd.DataFrame:
    parts = [group.sample(n=min(n, len(group)), random_state=seed) for _, group in frame.groupby(keys)]
    return pd.concat(parts).reset_index(drop=True)  # sample balanced subset


def load_splits() -> pd.DataFrame:
    path = config.MANIFEST_DIR / "splits.csv"  # splits manifest
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run python -m src.data.splits first.")
    return pd.read_csv(path)


class SplitDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Callable[[Image.Image], torch.Tensor]):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform  # image transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        with Image.open(config.PROCESSED_ROOT / row["processed_file"]) as im:
            canvas = im.convert("RGB")  # load processed canvas
        return self.transform(canvas), torch.tensor(float(row["label"])), i  # return transformed tensor and label
