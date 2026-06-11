"""Evaluate publicly released off-the-shelf AI-text detectors on FAITH-Detect's frames.

NOTE: these detectors were trained on OTHER distributions (HC3 question answering for
Hello-SimpleAI/chatgpt-detector-roberta; GPT-2 WebText generations for
openai-community/roberta-base-openai-detector). They are applied with NO fine-tuning and
a fixed 0.5 threshold, so all numbers are zero-shot-transfer style reference points, not
like-for-like competitors to the in-domain-trained FAITH-Detect models.

Evaluation frames (same protocol as the main grid):
  (a) the grouped MAiDE-up English test split,
  (b) its function-word-attacked and synonym-attacked versions
      (build_attack_set, rate 0.5, seed 0 -- identical to the grid's attack protocol),
  (c) the RAID hotel-reviews pool (results/cache/raid_reviews_pool.parquet).

Per-(detector, frame) probabilities are cached under results/cache/offshelf/, so an
interrupted (niced) run resumes without recomputing finished frames; the JSON output is
also written incrementally.

Example:
  python scripts/eval_offshelf.py --detectors Hello-SimpleAI/chatgpt-detector-roberta \
      openai-community/roberta-base-openai-detector --device cpu --out results/offshelf.json
"""
import argparse
import hashlib
import os
import re

import numpy as np

import _bootstrap  # noqa: F401
from faithdetect.data import load_maide_up_english, make_splits, build_attack_set
from faithdetect.evaluate import classification_metrics
from faithdetect.function_words import build_function_word_set
from faithdetect.offshelf import OffShelfDetector
from faithdetect.utils.logging import save_json, env_info

THRESHOLD = 0.5
FRAME_ORDER = ("test", "fw_attack", "synonym_attack", "raid_reviews")


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def _cache_path(cache_dir: str, model_name: str, frame_name: str, texts: list[str],
                max_length: int) -> str:
    """Cache key = detector + truncation length + content hash of the exact frame texts,
    so any change to splits/attacks/pool invalidates the cache automatically."""
    h = hashlib.md5()
    h.update(model_name.encode())
    h.update(str(max_length).encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return os.path.join(cache_dir, f"{_slug(model_name)}_{frame_name}_{h.hexdigest()[:10]}.npz")


def cached_proba_ai(det: OffShelfDetector, frame_name: str, texts: list[str],
                    cache_dir: str, batch_size: int) -> tuple[np.ndarray, int]:
    """Return (p_ai, ai_class_index), served from cache when available (the resolved
    AI class index is cached alongside so fully-cached runs never load the model)."""
    path = _cache_path(cache_dir, det.model_name, frame_name, texts, det.max_length)
    if os.path.exists(path):
        print(f"   [cache] {frame_name}: {path}")
        z = np.load(path)
        return z["p_ai"], int(z["ai_index"])
    p_ai = det.predict_proba_ai(texts, batch_size=batch_size)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, p_ai=p_ai, ai_index=int(det.ai_index))
    return p_ai, int(det.ai_index)


def build_frames(args) -> dict:
    """Return {frame_name: DataFrame(text,label)} for the three evaluation axes."""
    df = load_maide_up_english(args.data_csv)
    sp = make_splits(df, mode="grouped", seed=args.seed)
    fw = build_function_word_set(args.fw_definition)
    attacked = build_attack_set(
        sp.test, fw, attacks=("function_word", "synonym"), rate=args.attack_rate, seed=0
    )
    frames = {
        "test": sp.test,
        "fw_attack": attacked["function_word"],
        "synonym_attack": attacked["synonym"],
    }
    if args.raid_parquet and os.path.exists(args.raid_parquet):
        import pandas as pd

        raid = pd.read_parquet(args.raid_parquet)
        if args.max_raid and len(raid) > args.max_raid:
            raid = raid.sample(n=args.max_raid, random_state=args.seed).reset_index(drop=True)
        frames["raid_reviews"] = raid
    else:
        print(f"[warn] RAID pool not found at {args.raid_parquet}; skipping that frame.")
    return frames


def print_table(results: dict) -> None:
    cols = ("accuracy", "f1_macro", "precision", "recall", "roc_auc")
    header = f"{'detector':40s} {'frame':16s} {'n':>5s} " + " ".join(f"{c:>9s}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, dres in results["detectors"].items():
        short = name.split("/")[-1][:40]
        for fname in FRAME_ORDER:
            if fname not in dres:
                continue
            m = dres[fname]["metrics"]
            vals = " ".join(f"{m.get(c, float('nan')):9.3f}" for c in cols)
            print(f"{short:40s} {fname:16s} {dres[fname]['n']:5d} {vals}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detectors", nargs="+",
                    default=["Hello-SimpleAI/chatgpt-detector-roberta",
                             "openai-community/roberta-base-openai-detector"])
    ap.add_argument("--data_csv", default="data/all_data.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fw_definition", default="union")
    ap.add_argument("--attack_rate", type=float, default=0.5)
    ap.add_argument("--raid_parquet", default="results/cache/raid_reviews_pool.parquet")
    ap.add_argument("--max_raid", type=int, default=None,
                    help="seeded subsample of the RAID pool (default: full pool)")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--cache_dir", default="results/cache/offshelf")
    ap.add_argument("--out", default="results/offshelf.json")
    args = ap.parse_args()

    frames = build_frames(args)
    results = {
        "meta": {
            "args": vars(args),
            "env": env_info(),
            "threshold": THRESHOLD,
            "note": ("Off-the-shelf detectors trained on other distributions; applied "
                     "zero-shot-transfer style with no fine-tuning, threshold 0.5."),
        },
        "detectors": {},
    }

    for name in args.detectors:
        print(f"\n== off-the-shelf detector: {name} ==", flush=True)
        det = OffShelfDetector(name, device=args.device, max_length=args.max_length)
        dres: dict = {}
        for fname in FRAME_ORDER:
            if fname not in frames:
                continue
            frame = frames[fname]
            texts = frame["text"].tolist()
            y_true = frame["label"].to_numpy()
            p_ai, ai_index = cached_proba_ai(det, fname, texts, args.cache_dir, args.batch_size)
            y_pred = (p_ai >= THRESHOLD).astype(int)
            dres[fname] = {
                "n": int(len(frame)),
                "ai_class_index": ai_index,
                "metrics": classification_metrics(y_true, y_pred, p_ai),
            }
            results["detectors"][name] = dres
            save_json(args.out, results)  # incremental: survive interruption
            print(f"   {fname}: f1_macro={dres[fname]['metrics']['f1_macro']:.3f}", flush=True)

    print(f"\nSaved {args.out}")
    print_table(results)


if __name__ == "__main__":
    main()
