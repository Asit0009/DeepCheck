import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.splits import build_splits

FAMILIAR = ["ADM", "BigGAN", "glide", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "wukong"]
UNSEEN = ["Midjourney", "VQDM"]


def _hex(rng) -> str:
    return f"{int(rng.integers(0, 2**63, dtype=np.uint64)) * 2 + int(rng.integers(0, 2)):016x}"


def synthetic(n_train=40, n_val_folder=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def add(split, gen, label, k):
        rid = f"{split}/{gen}/{'fake' if label else 'real'}/img_{k}.png"
        rows.append(dict(record_id=rid, official_split=split, generator=gen, label=label,
                         sha256=f"{len(rows):064x}", pixel_sha256=f"{len(rows) + 10**6:064x}", dhash=_hex(rng), phash=_hex(rng),
                         processed_file=rid))

    for g in FAMILIAR:
        for label in (0, 1):
            for k in range(n_train):
                add("train", g, label, k)
    for g in FAMILIAR + UNSEEN:
        for label in (0, 1):
            for k in range(n_val_folder if g in FAMILIAR else 30):
                add("test", g, label, k)
    return pd.DataFrame(rows)


@pytest.fixture()
def cfg():
    c = config.load_yaml(config.CONFIGS_DIR / "data.yaml")
    c["sampling"]["calibration"] = {"fake_per_generator": 10, "real_per_generator": 10}
    c["sampling"]["test"] = {"fake_per_generator": 30, "real_per_generator": 30}
    return c


def test_unseen_generators_never_reach_train_validation_or_calibration(cfg):
    df, _ = build_splits(synthetic(), cfg)
    used = df[df["split"].isin(["train", "validation", "calibration"])]
    assert set(used["generator"]).isdisjoint(UNSEEN)
    assert set(df[df["split"] == "unseen_test"]["generator"]) == set(UNSEEN)


def test_train_and_validation_come_only_from_official_train_folders(cfg):
    df, _ = build_splits(synthetic(), cfg)
    assert set(df[df["split"].isin(["train", "validation"])]["official_split"]) == {"train"}
    assert set(df[df["split"].isin(["calibration", "test", "unseen_test"])]["official_split"]) == {"test"}


def test_calibration_and_test_are_disjoint_and_sized(cfg):
    df, _ = build_splits(synthetic(), cfg)
    cal, test = df[df["split"] == "calibration"], df[df["split"] == "test"]
    assert set(cal["record_id"]).isdisjoint(set(test["record_id"]))
    per = cal.groupby(["generator", "label"]).size()
    assert (per == 10).all() and len(per) == len(FAMILIAR) * 2          # 10 per generator per class
    assert (test.groupby(["generator", "label"]).size() == 30).all()


def test_validation_fraction_is_respected(cfg):
    df, _ = build_splits(synthetic(n_train=100), cfg)
    per = df[df["split"].isin(["train", "validation"])].groupby(["generator", "label", "split"]).size().unstack()
    assert (per["validation"] == 15).all() and (per["train"] == 85).all()


def test_exact_duplicate_between_train_and_test_is_removed_from_train(cfg):
    df = synthetic()
    train_idx = df.index[(df["official_split"] == "train") & (df["generator"] == "ADM") & (df["label"] == 0)][0]
    test_idx = df.index[(df["official_split"] == "test") & (df["generator"] == "wukong") & (df["label"] == 0)][0]
    df.loc[train_idx, "sha256"] = df.loc[test_idx, "sha256"]               # same file bytes
    out, report = build_splits(df, cfg)
    train_rid, test_rid = df.at[train_idx, "record_id"], df.at[test_idx, "record_id"]
    assert out.set_index("record_id").at[train_rid, "split"] == "dropped_duplicate"
    assert out.set_index("record_id").at[test_rid, "split"] in {"test", "calibration"}
    assert report["exact_duplicate_pairs_found"] >= 1


def test_same_pixels_in_different_encodings_count_as_duplicates(cfg):
    df = synthetic()
    a = df.index[(df["official_split"] == "train") & (df["generator"] == "glide") & (df["label"] == 1)][0]
    b = df.index[(df["official_split"] == "train") & (df["generator"] == "BigGAN") & (df["label"] == 1)][0]
    df.loc[b, "pixel_sha256"] = df.loc[a, "pixel_sha256"]
    out, _ = build_splits(df, cfg)
    assert (out["split"] == "dropped_duplicate").sum() == 1


def test_near_duplicate_is_removed_from_the_less_protected_source(cfg):
    df = synthetic()
    t = df.index[(df["official_split"] == "train") & (df["generator"] == "ADM") & (df["label"] == 1)][3]
    u = df.index[(df["generator"] == "Midjourney") & (df["label"] == 1)][2]
    flip = (1 << 3) | (1 << 40)                                              # two bits differ: well inside the threshold
    df.loc[t, "phash"] = f"{int(df.at[u, 'phash'], 16) ^ flip:016x}"
    df.loc[t, "dhash"] = df.at[u, "dhash"]
    out, report = build_splits(df, cfg)
    out = out.set_index("record_id")
    assert out.at[df.at[t, "record_id"], "split"] == "dropped_duplicate"       # the train copy goes
    assert out.at[df.at[u, "record_id"], "split"] == "unseen_test"             # the protected unseen image stays
    assert report["near_duplicate_pairs_found"] >= 1


def test_unseen_test_image_wins_over_familiar_test_pool_duplicate(cfg):
    df = synthetic()
    u = df.index[(df["generator"] == "VQDM") & (df["label"] == 0)][0]
    f = df.index[(df["official_split"] == "test") & (df["generator"] == "ADM") & (df["label"] == 0)][0]
    df.loc[f, "sha256"] = df.loc[u, "sha256"]
    out, _ = build_splits(df, cfg)
    out = out.set_index("record_id")
    assert out.at[df.at[u, "record_id"], "split"] == "unseen_test"
    assert out.at[df.at[f, "record_id"], "split"] == "dropped_duplicate"


def test_no_duplicate_group_spans_two_splits(cfg):
    df = synthetic()
    rng = np.random.default_rng(5)
    for _ in range(25):                                                       # sprinkle exact duplicates around
        a, b = rng.choice(len(df), 2, replace=False)
        df.loc[b, "sha256"] = df.loc[a, "sha256"]
    out, _ = build_splits(df, cfg)
    kept = out[~out["split"].str.startswith("dropped")]
    assert (kept.groupby("group")["split"].nunique() == 1).all()
    assert kept["group"].is_unique                                              # exactly one survivor per group


def test_splits_are_deterministic(cfg):
    a, _ = build_splits(synthetic(), cfg)
    b, _ = build_splits(synthetic(), cfg)
    assert a["split"].tolist() == b["split"].tolist()
    cfg2 = {**cfg, "splits": {**cfg["splits"], "seed": cfg["splits"]["seed"] + 1}}
    c, _ = build_splits(synthetic(), cfg2)
    assert a["split"].tolist() != c["split"].tolist()


def test_unseen_generator_in_train_pool_is_refused(cfg):
    df = synthetic()
    df.loc[0, "generator"] = "Midjourney"                                       # a Midjourney image in the train pool
    with pytest.raises(AssertionError, match="unseen generator"):
        build_splits(df, cfg)
