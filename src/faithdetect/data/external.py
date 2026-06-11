"""External AI-text corpora for cross-domain transfer experiments.

FAITH-Detect trains on MAiDE-up hotel reviews; these loaders supply *out-of-domain*
human/AI pairs so the paper can ask whether function-word invariance transfers beyond
reviews. Each loader returns a tidy ``DataFrame[text, label]`` (label 1 = AI, 0 = human),
class-balanced, capped, deterministically shuffled, and cached to parquet so repeated
(possibly niced) runs never re-hit the network.

Sources
-------
* ``load_hc3``               : HC3 (Guo et al., 2023) — paired human / ChatGPT answers
                               across QA domains (config ``all``). Answer lists are
                               exploded to one row per answer.
* ``load_gpt_wiki``          : GPT-wiki-intro (Aaditya Bhat, 2023) — paired Wikipedia
                               intros (``wiki_intro`` = human, ``generated_intro`` = AI).
* ``load_raid_domain_duckdb``: one domain of RAID (Dugan et al., ACL 2024) pulled via
                               duckdb httpfs over the HuggingFace parquet conversion.
                               The conversion is *partial* and contains only the
                               ``abstracts``/``books``/``news``/``poetry`` domains; use
                               :mod:`faithdetect.data.raid` for the other domains.

Caching: a cache file is keyed by its *path only* — changing ``cap_per_class`` (or other
filters) does not invalidate an existing parquet, so use a distinct ``cache_path`` per
configuration or delete the stale file.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

HUMAN_LABEL = 0
AI_LABEL = 1

_RAID_PARQUET_INDEX = "https://datasets-server.huggingface.co/parquet?dataset=liamdugan/raid"
#: Domains present in the (partial) HF parquet conversion of liamdugan/raid, config
#: ``raid``, split ``train``. Other RAID domains never appear in these shards.
RAID_DUCKDB_DOMAINS = ("abstracts", "books", "news", "poetry")


# --------------------------------------------------------------------------- utilities
def _read_cache(cache_path: str | None) -> pd.DataFrame | None:
    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return None


def _write_cache(df: pd.DataFrame, cache_path: str | None) -> None:
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)


def _clean(text: object) -> str:
    return " ".join(str(text or "").split())


def _maybe_add(
    bucket: list[str], text: object, cap: int, min_words: int, seen: set[str]
) -> None:
    """Append a cleaned, deduplicated, long-enough text to ``bucket`` (up to ``cap``)."""
    if len(bucket) >= cap:
        return
    t = _clean(text)
    if len(t.split()) < min_words or t in seen:
        return
    seen.add(t)
    bucket.append(t)


def _finalise(
    human: list[str], ai: list[str], cap_per_class: int, seed: int, source: str
) -> pd.DataFrame:
    """Balance the two class buckets, shuffle deterministically, return text/label frame."""
    n = min(len(human), len(ai), cap_per_class)
    if n == 0:
        raise RuntimeError(
            f"{source}: collected {len(human)} human / {len(ai)} AI usable texts — "
            "cannot build a balanced frame. Check network access / dataset availability."
        )
    df = pd.DataFrame(
        {
            "text": human[:n] + ai[:n],
            "label": [HUMAN_LABEL] * n + [AI_LABEL] * n,
        }
    )
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------------- HC3
def _hc3_examples() -> Iterator[dict]:
    """Yield raw HC3 'all' examples (question, human_answers, chatgpt_answers).

    Prefers ``datasets`` streaming; HC3 is a *script* dataset, which ``datasets>=3``
    refuses to load, so we fall back to streaming the repo's ``all.jsonl`` directly
    (downloaded once into the HuggingFace hub cache).
    """
    try:
        from datasets import load_dataset

        return iter(
            load_dataset("Hello-SimpleAI/HC3", "all", split="train", streaming=True)
        )
    except Exception:
        pass
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")

    def gen() -> Iterator[dict]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    return gen()


def load_hc3(
    cap_per_class: int = 600,
    cache_path: str | None = "results/cache/hc3.parquet",
    seed: int = 0,
    min_words: int = 20,
) -> pd.DataFrame:
    """Balanced human-vs-ChatGPT answers from HC3 (config ``all``).

    Streams the corpus, explodes the ``human_answers`` / ``chatgpt_answers`` lists to one
    row per answer (dropping answers under ``min_words`` words and exact duplicates), and
    stops as soon as both class caps are met. Returns ``DataFrame[text, label]`` with
    exactly ``min(cap_per_class, available)`` rows per class, shuffled with ``seed``.
    """
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    human: list[str] = []
    ai: list[str] = []
    seen: set[str] = set()
    try:
        for ex in _hc3_examples():
            for ans in ex.get("human_answers") or []:
                _maybe_add(human, ans, cap_per_class, min_words, seen)
            for ans in ex.get("chatgpt_answers") or []:
                _maybe_add(ai, ans, cap_per_class, min_words, seen)
            if len(human) >= cap_per_class and len(ai) >= cap_per_class:
                break
    except Exception as e:
        raise RuntimeError(
            "Failed to stream HC3 (Hello-SimpleAI/HC3, config 'all'). "
            "Check network access and that the dataset is still hosted on the Hub."
        ) from e
    df = _finalise(human, ai, cap_per_class, seed, source="HC3")
    _write_cache(df, cache_path)
    return df


# ------------------------------------------------------------------------ GPT-wiki-intro
def load_gpt_wiki(
    cap_per_class: int = 600,
    cache_path: str | None = "results/cache/gptwiki.parquet",
    seed: int = 0,
    min_words: int = 20,
) -> pd.DataFrame:
    """Balanced Wikipedia-intro pairs from ``aadityaubhat/GPT-wiki-intro``.

    Each source row pairs a human intro (``wiki_intro``) with a GPT continuation
    (``generated_intro``); both sides are collected independently under the same
    ``min_words`` / dedup discipline, and streaming stops once both caps are met.
    Returns ``DataFrame[text, label]`` shuffled with ``seed``.
    """
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    human: list[str] = []
    ai: list[str] = []
    seen: set[str] = set()
    try:
        from datasets import load_dataset

        ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train", streaming=True)
        for ex in ds:
            _maybe_add(human, ex.get("wiki_intro"), cap_per_class, min_words, seen)
            _maybe_add(ai, ex.get("generated_intro"), cap_per_class, min_words, seen)
            if len(human) >= cap_per_class and len(ai) >= cap_per_class:
                break
    except Exception as e:
        raise RuntimeError(
            "Failed to stream aadityaubhat/GPT-wiki-intro. "
            "Check network access and that the dataset is still hosted on the Hub."
        ) from e
    df = _finalise(human, ai, cap_per_class, seed, source="GPT-wiki-intro")
    _write_cache(df, cache_path)
    return df


# ------------------------------------------------------------------------- RAID (duckdb)
def _raid_train_shard_urls() -> list[str]:
    """Return the parquet shard URLs for liamdugan/raid, config ``raid``, split ``train``."""
    import requests

    try:
        resp = requests.get(_RAID_PARQUET_INDEX, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        raise RuntimeError(
            f"Could not query the HF datasets-server parquet index ({_RAID_PARQUET_INDEX})."
        ) from e
    urls = [
        f["url"]
        for f in payload.get("parquet_files", [])
        if f.get("config") == "raid" and f.get("split") == "train" and f.get("url")
    ]
    if not urls:
        raise RuntimeError(
            "No parquet shards with config='raid', split='train' in the datasets-server "
            f"response (keys: {sorted(payload)}). The conversion may have moved."
        )
    return urls


def load_raid_domain_duckdb(
    domain: str,
    cap_per_class: int = 300,
    cache_path: str | None = None,
    min_words: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Balanced human-vs-AI sample for one RAID domain via duckdb httpfs.

    Runs a single remote scan over the HF parquet conversion of ``liamdugan/raid``
    (config ``raid``, split ``train`` — covers only :data:`RAID_DUCKDB_DOMAINS`),
    keeping rows with ``attack='none'``. The subsample is deterministic: per class,
    rows are ranked by ``md5(generation)`` and the top ranks fetched (2x the cap, to
    survive the ``min_words`` / dedup filters), so re-runs select identical rows with
    no seed-dependent network traffic. Returns ``DataFrame[text, label, model]``
    balanced to ``min(cap_per_class, available)`` per class, shuffled with ``seed``.
    """
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    if domain not in RAID_DUCKDB_DOMAINS:
        raise ValueError(
            f"domain must be one of {RAID_DUCKDB_DOMAINS} (the HF parquet conversion of "
            f"liamdugan/raid contains only these); got {domain!r}. For other domains use "
            "faithdetect.data.raid.load_raid_sample / load_raid_from_csv."
        )
    try:
        import duckdb
    except ImportError as e:
        raise RuntimeError(
            "load_raid_domain_duckdb requires the 'duckdb' package (pip install duckdb)."
        ) from e

    urls = _raid_train_shard_urls()
    urls_sql = ", ".join("'" + u.replace("'", "''") + "'" for u in urls)
    fetch_per_class = 2 * int(cap_per_class)
    query = f"""
        WITH pool AS (
            SELECT generation, model, (model <> 'human') AS is_ai
            FROM read_parquet([{urls_sql}])
            WHERE domain = ? AND attack = 'none' AND generation IS NOT NULL
        ),
        ranked AS (
            SELECT generation, model, is_ai,
                   row_number() OVER (PARTITION BY is_ai ORDER BY md5(generation)) AS rk
            FROM pool
        )
        SELECT generation, model FROM ranked WHERE rk <= ?
    """
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        raw = con.execute(query, [domain, fetch_per_class]).df()
    except Exception as e:
        raise RuntimeError(
            f"duckdb httpfs query over {len(urls)} RAID parquet shards failed for "
            f"domain={domain!r}. Check network access (and that httpfs can be installed)."
        ) from e
    finally:
        con.close()
    if raw.empty:
        raise RuntimeError(
            f"RAID parquet shards returned no attack='none' rows for domain={domain!r}."
        )

    raw["text"] = raw["generation"].map(_clean)
    raw = raw[raw["text"].str.split().str.len() >= min_words]
    raw = raw.drop_duplicates(subset="text")
    raw["label"] = (raw["model"] != "human").astype(int)  # AI_LABEL=1, HUMAN_LABEL=0
    # Deterministic per-class order, independent of duckdb's result row order.
    raw["_rank"] = raw["text"].map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    raw = raw.sort_values("_rank")
    n = min(
        int((raw["label"] == HUMAN_LABEL).sum()),
        int((raw["label"] == AI_LABEL).sum()),
        int(cap_per_class),
    )
    if n == 0:
        counts = raw["label"].value_counts().to_dict()
        raise RuntimeError(
            f"RAID domain={domain!r}: cannot balance classes after filtering "
            f"(label counts: {counts})."
        )
    parts = [raw[raw["label"] == lab].head(n) for lab in (HUMAN_LABEL, AI_LABEL)]
    df = (
        pd.concat(parts)[["text", "label", "model"]]
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    _write_cache(df, cache_path)
    return df
