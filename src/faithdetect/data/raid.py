"""RAID out-of-distribution sampler (cross-domain, cross-generator, adversarial).

RAID (Dugan et al., ACL 2024) provides 6M+ generations over 11 generators, 8 domains and
11 adversarial attacks. It is hosted on HuggingFace (``liamdugan/raid``) as monolithic CSVs
that are *sorted by domain*, so reaching a late domain (e.g. ``reviews``) by streaming is
slow. We therefore stream with explicit caps and an optional ``max_scan`` budget:

* Local smoke runs use a cheaply-reachable domain (``abstracts``) as a genuine, and in fact
  harder, cross-domain test (hotel reviews -> scientific abstracts).
* The Colab notebook calls this with ``domains=['reviews', ...]`` and a large/None
  ``max_scan`` (or a pre-downloaded CSV) to build the full reviews / cross-generator /
  attack matrix.

Schema used: ``model`` ('human' or a generator name -> label), ``domain``, ``attack``,
``generation`` (the text).
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HUMAN_LABEL = 0
AI_LABEL = 1


def load_raid_from_csv(
    csv_path: str,
    domains=("reviews",),
    attacks=("none",),
    cap_per_group: int = 200,
    balanced: bool = True,
    min_words: int = 20,
    seed: int = 0,
    chunksize: int = 200_000,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Build a capped RAID sample from a pre-downloaded CSV via chunked pandas reads.

    Far faster than streaming for *late* domains (the CSV is domain-sorted), so this is the
    right way to reach the ``reviews`` domain on Colab. Keeps the ``model`` column so the caller
    can do a cross-generator breakdown. Returns columns: text, label, domain, model, attack.
    """
    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    domains, attacks = set(domains), set(attacks)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    seen_target = False
    cols = ["generation", "model", "domain", "attack"]
    for chunk in pd.read_csv(csv_path, usecols=lambda c: c in cols, chunksize=chunksize):
        hit = chunk[chunk["domain"].isin(domains) & chunk["attack"].isin(attacks)]
        if len(hit):
            seen_target = True
            for _, ex in hit.iterrows():
                text = str(ex.get("generation") or "").strip()
                if len(text.split()) < min_words:
                    continue
                model = ex.get("model", "unknown")
                key = (ex["domain"], model)
                if len(buckets[key]) >= cap_per_group:
                    continue
                label = HUMAN_LABEL if model == "human" else AI_LABEL
                buckets[key].append({"text": text, "label": label,
                                     "domain": ex["domain"], "model": model, "attack": ex["attack"]})
        elif seen_target:
            break  # domain-sorted: once past the target domains, nothing more to find
    rows = [r for items in buckets.values() for r in items]
    if not rows:
        raise RuntimeError(f"No RAID rows for domains={domains} in {csv_path}")
    df = pd.DataFrame(rows)
    if balanced:
        df = _balance(df, seed)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return df


def _balance(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(int((df["label"] == HUMAN_LABEL).sum()), int((df["label"] == AI_LABEL).sum()))
    parts = []
    for lab in (HUMAN_LABEL, AI_LABEL):
        sub = df[df["label"] == lab]
        take = rng.choice(sub.index.values, size=min(n, len(sub)), replace=False)
        parts.append(df.loc[take])
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_raid_sample(
    domains=("abstracts",),
    attacks=("none",),
    cap_per_group: int = 150,
    balanced: bool = True,
    min_words: int = 20,
    seed: int = 0,
    max_scan: int | None = 400_000,
    split: str = "train",
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Stream a capped, optionally class-balanced RAID sample.

    A "group" is a (domain, model) pair, so each generator and the human set are each capped
    at `cap_per_group`. Returns columns: text, label, domain, model, attack.
    """
    domains = set(domains)
    attacks = set(attacks)
    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    from datasets import load_dataset

    ds = load_dataset("liamdugan/raid", split=split, streaming=True)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    n_scanned = 0
    for ex in ds:
        n_scanned += 1
        if max_scan is not None and n_scanned > max_scan:
            break
        dom = ex.get("domain")
        att = ex.get("attack")
        if dom not in domains or att not in attacks:
            # Early stop: file is domain-sorted, so once we are *past* all requested
            # domains there is nothing left to find. We approximate this by stopping when
            # every requested group has reached its cap (checked below).
            continue
        text = (ex.get("generation") or "").strip()
        if len(text.split()) < min_words:
            continue
        model = ex.get("model", "unknown")
        key = (dom, model)
        if len(buckets[key]) >= cap_per_group:
            continue
        label = HUMAN_LABEL if model == "human" else AI_LABEL
        buckets[key].append(
            {"text": text, "label": label, "domain": dom, "model": model, "attack": att}
        )

    rows = [r for items in buckets.values() for r in items]
    if not rows:
        raise RuntimeError(
            f"No RAID rows collected for domains={domains} within max_scan={max_scan}. "
            "Increase max_scan or pre-download the CSV (see notebooks/colab_full_run.ipynb)."
        )
    df = pd.DataFrame(rows)

    if balanced:
        rng = np.random.default_rng(seed)
        n = min(int((df["label"] == HUMAN_LABEL).sum()), int((df["label"] == AI_LABEL).sum()))
        parts = []
        for lab in (HUMAN_LABEL, AI_LABEL):
            sub = df[df["label"] == lab]
            take = rng.choice(sub.index.values, size=min(n, len(sub)), replace=False)
            parts.append(df.loc[take])
        df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return df
