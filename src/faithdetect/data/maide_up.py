"""MAiDE-up English in-domain data: loading, leakage-free splits, torch Dataset.

MAiDE-up (Ignat, Xu & Mihalcea, Findings of NAACL 2025) contains human and GPT-4-generated
hotel reviews. We use the English subset (2,000 reviews = 1,000 human + 1,000 AI across
100 hotels in 10 cities, exactly 20 reviews per hotel, every hotel present in both classes).

Because every hotel appears in BOTH classes, a naive random split lets the model memorise
hotel-specific content that leaks across train/test. We therefore default to a
GROUPED split by hotel; the random split is still available so the paper can quantify the
leakage gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset

HUMAN_LABEL = 0
AI_LABEL = 1


def load_maide_up_english(csv_path: str) -> pd.DataFrame:
    """Load the English MAiDE-up subset into a tidy frame.

    Returns columns: ``text`` (Upside + Downside combined), ``label`` (0 human / 1 AI),
    ``hotel``, ``city``. Empty reviews are dropped.
    """
    df = pd.read_csv(csv_path)
    en = df[df["Review_Language"] == "English"].copy()
    up = en["Upside_Review"].fillna("").astype(str)
    down = en["Downside_Review"].fillna("").astype(str)
    en["text"] = (up + " " + down).str.replace(r"\s+", " ", regex=True).str.strip()
    en = en[en["text"] != ""].copy()
    out = pd.DataFrame(
        {
            "text": en["text"].values,
            "label": en["source"].astype(int).values,
            "hotel": en["Hotel Name"].astype(str).values,
            "city": en["City Name"].astype(str).values,
        }
    )
    return out.reset_index(drop=True)


@dataclass
class SplitBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    mode: str
    seed: int

    def describe(self) -> dict:
        def stats(d: pd.DataFrame) -> dict:
            return {
                "n": int(len(d)),
                "n_ai": int((d["label"] == AI_LABEL).sum()),
                "n_human": int((d["label"] == HUMAN_LABEL).sum()),
                "n_hotels": int(d["hotel"].nunique()),
            }

        return {
            "mode": self.mode,
            "seed": self.seed,
            "train": stats(self.train),
            "val": stats(self.val),
            "test": stats(self.test),
            "hotel_leakage_train_test": int(
                len(set(self.train["hotel"]) & set(self.test["hotel"]))
            ),
        }


def make_splits(
    df: pd.DataFrame,
    mode: str = "grouped",
    seed: int = 0,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> SplitBundle:
    """Build train/val/test.

    mode="grouped": split by hotel so no hotel appears in two folds (no leakage).
    mode="random" : stratified random split by label, ignoring hotels (leaky baseline).
    `val_size` is a fraction of the WHOLE dataset (carved from the train+val portion).
    """
    if mode == "grouped":
        groups = df["hotel"].values
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(gss1.split(df, groups=groups))
        trainval = df.iloc[trainval_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        rel_val = val_size / (1 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        tr_idx, val_idx = next(gss2.split(trainval, groups=trainval["hotel"].values))
        train = trainval.iloc[tr_idx].reset_index(drop=True)
        val = trainval.iloc[val_idx].reset_index(drop=True)
    elif mode == "random":
        trainval, test = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["label"]
        )
        rel_val = val_size / (1 - test_size)
        train, val = train_test_split(
            trainval, test_size=rel_val, random_state=seed, stratify=trainval["label"]
        )
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)
    else:
        raise ValueError(f"Unknown split mode: {mode!r}")
    return SplitBundle(train, val, test, mode=mode, seed=seed)


class TextLabelDataset(Dataset):
    """Holds raw text + label (+ optional group). Tokenisation happens in the Collator so
    function-word masking is applied in one place for all three model variants."""

    def __init__(self, texts, labels, groups=None):
        self.texts = list(texts)
        self.labels = [int(x) for x in labels]
        self.groups = list(groups) if groups is not None else [None] * len(self.texts)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "TextLabelDataset":
        return cls(df["text"].tolist(), df["label"].tolist(), df.get("hotel"))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {"text": self.texts[idx], "label": self.labels[idx], "index": idx}
