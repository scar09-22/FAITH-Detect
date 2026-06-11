"""Replication study: does the function-word-invariance story hold beyond MAiDE-up?

Trains and evaluates the detector variants end-to-end on a SECOND corpus (HC3 or
GPT-wiki-intro, loaded via ``faithdetect.data.external``) and asks three questions per
variant:

  indomain        - how well does it detect AI text on the external corpus itself?
  fw_attack       - does it survive the targeted function-word attack (rate 0.5) that
                    the Hard-Mask variant is invariant to by construction?
  maide_transfer  - does an externally-trained detector transfer zero-shot to the FIXED
                    MAiDE-up grouped test split (hotel reviews)?

Protocol per (variant, seed):
  1. Build the external frame (lazy import of ``faithdetect.data.external``, so this script
     stays importable when that module or its data dependencies are absent).
  2. Stratified 60/20/20 train/val/test ROW split. NOTE: unlike MAiDE-up, these corpora
     carry no entity grouping (no hotel-like unit is modelled), so a grouped, leakage-free
     split is not available; absolute in-domain numbers are therefore optimistic relative
     to MAiDE-up's grouped protocol. The replication question — the CONTRAST between
     variants — is unaffected, because all variants share the identical split per seed.
  3. Stratified subsample of the train split to --train_n rows, then ``train_model``.
  4. Evaluate the three axes above with ``evaluate_split``.

Labels follow the MAiDE-up convention (0 = human, 1 = AI). Seeds are aggregated per
variant (mean/SD/95% CI). Each (variant, seed) cell is cached as JSON under
``results/cells/replicate_<dataset>/`` keyed by its hyperparameters, so an interrupted
(e.g. niced) run resumes without retraining; memory is released between cells.

Example:
  nice python scripts/replicate_external.py --dataset hc3 --device cpu
"""
from __future__ import annotations

import argparse
import gc
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import _bootstrap  # noqa: F401
from faithdetect.data import build_attack_set, load_maide_up_english, make_splits
from faithdetect.data.maide_up import AI_LABEL, HUMAN_LABEL, SplitBundle
from faithdetect.evaluate import evaluate_split
from faithdetect.function_words import FunctionWordSet, build_function_word_set
from faithdetect.models import ModelConfig, build_tokenizer
from faithdetect.train import TrainConfig, train_model
from faithdetect.utils.logging import env_info, load_json, save_json
from faithdetect.utils.stats import aggregate_seeds

# The transfer target is ONE fixed grouped MAiDE-up test split, identical for every cell,
# so transfer numbers are comparable across variants and seeds.
MAIDE_SPLIT_SEED = 0

AXES = ("indomain", "fw_attack", "maide_transfer")
AXIS_HEADERS = {"indomain": "indomain F1", "fw_attack": "fw-attack F1", "maide_transfer": "->MAiDE F1"}


def load_external_frame(dataset: str) -> pd.DataFrame:
    """Load an external corpus as a tidy (text, label, hotel) frame.

    Imports ``faithdetect.data.external`` lazily and resolves the loader flexibly: a
    dataset-specific ``load_<dataset>()`` is preferred, falling back to a generic
    ``load_external(dataset)``. A constant ``hotel`` = None column is added because
    downstream split/dataset utilities expect it; no entity grouping exists here.
    """
    from faithdetect.data import external  # lazy: optional module / data dependencies

    loader = getattr(external, f"load_{dataset}", None)
    if loader is not None:
        df = loader()
    elif hasattr(external, "load_external"):
        df = external.load_external(dataset)
    else:
        raise AttributeError(
            f"faithdetect.data.external provides neither load_{dataset}() nor load_external()"
        )
    df = df.copy()
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"External loader for {dataset!r} is missing columns {sorted(missing)}")
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"] != ""].copy()
    df["label"] = df["label"].astype(int)
    if "hotel" not in df.columns:
        df["hotel"] = None
    return df.reset_index(drop=True)


def split_rows(df: pd.DataFrame, seed: int, train_n: int | None) -> SplitBundle:
    """Stratified 60/20/20 row split, with the train split subsampled to ``train_n``.

    Row-level (not grouped) by necessity — see module docstring. The split depends only on
    ``seed``, so all variants at the same seed see identical train/val/test rows.
    """
    trainval, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )
    train, val = train_test_split(  # 0.25 of the remaining 80% -> 20% of the whole
        trainval, test_size=0.25, random_state=seed, stratify=trainval["label"]
    )
    if train_n is not None and len(train) > train_n:
        train, _ = train_test_split(
            train, train_size=train_n, random_state=seed, stratify=train["label"]
        )
    return SplitBundle(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        mode="stratified-row",
        seed=seed,
    )


def run_cell(
    variant: str,
    seed: int,
    frame: pd.DataFrame,
    maide_test: pd.DataFrame,
    args: argparse.Namespace,
    tokenizer,
    func_id: int,
    fw_set: FunctionWordSet,
    device: torch.device,
) -> dict:
    """Train one (variant, seed) model on the external corpus and evaluate all three axes.

    Returns a JSON-serialisable cell with per-axis metric dicts (prediction arrays are
    dropped to keep the cache small). Frees the model before returning.
    """
    splits = split_rows(frame, seed=seed, train_n=args.train_n)
    model_cfg = ModelConfig(
        encoder_name=args.encoder, variant=variant, max_length=args.max_length
    )
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    model, history = train_model(
        model_cfg, train_cfg, splits, tokenizer, func_id, fw_set, device, seed
    )

    attacked = build_attack_set(
        splits.test, fw_set, attacks=("function_word",), rate=args.attack_rate, seed=0
    )["function_word"]
    frames = {"indomain": splits.test, "fw_attack": attacked, "maide_transfer": maide_test}

    cell: dict = {
        "variant": variant,
        "seed": seed,
        "history": history,
        "split_sizes": {
            "train": int(len(splits.train)),
            "val": int(len(splits.val)),
            "test": int(len(splits.test)),
        },
    }
    for axis, fr in frames.items():
        res = evaluate_split(model, fr, model_cfg, tokenizer, func_id, fw_set, device)
        cell[axis] = res["metrics"]
        print(f"  [{variant}|seed {seed}] {axis}: f1_macro={res['metrics']['f1_macro']:.4f} "
              f"acc={res['metrics']['accuracy']:.4f}", flush=True)
        del res

    del model, splits, attacked, frames
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return cell


def cell_config_key(args: argparse.Namespace) -> dict:
    """The hyperparameters a cached cell must match to be reused."""
    return {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_n": args.train_n,
        "attack_rate": args.attack_rate,
        "fw_definition": args.fw_definition,
        "maide_split_seed": MAIDE_SPLIT_SEED,
    }


def print_table(results: dict, variants: list[str]) -> None:
    """Render the variant x axis macro-F1 (mean +/- SD over seeds) summary table."""
    header = f"{'variant':<10}" + "".join(f"{AXIS_HEADERS[a]:>22}" for a in AXES)
    print("\n" + header)
    print("-" * len(header))
    for variant in variants:
        agg = results["variants"].get(variant, {}).get("aggregated", {})
        row = f"{variant:<10}"
        for axis in AXES:
            iv = agg.get(axis, {}).get("f1_macro")
            cellstr = f"{iv['mean']:.3f} +/- {iv['sd']:.3f}" if iv else "n/a"
            row += f"{cellstr:>22}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=["hc3", "gptwiki"],
                    help="external corpus served by faithdetect.data.external")
    ap.add_argument("--variants", nargs="+", default=["baseline", "hardmask"],
                    choices=["baseline", "hardmask", "softreg"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--encoder", default="distilroberta-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train_n", type=int, default=600,
                    help="stratified subsample of the train split (matches the MAiDE-up grid)")
    ap.add_argument("--attack_rate", type=float, default=0.5)
    ap.add_argument("--maide_csv", default="data/all_data.csv")
    ap.add_argument("--fw_definition", default="union")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: results/replicate_<dataset>.json)")
    args = ap.parse_args()
    out = args.out or os.path.join("results", f"replicate_{args.dataset}.json")
    cell_dir = os.path.join("results", "cells", f"replicate_{args.dataset}")
    os.makedirs(cell_dir, exist_ok=True)
    if not os.path.exists(args.maide_csv):
        raise FileNotFoundError(
            f"MAiDE-up csv not found at {args.maide_csv!r} (needed for the transfer axis)"
        )

    device = torch.device(args.device)
    fw_set = build_function_word_set(args.fw_definition)
    tokenizer, func_id = build_tokenizer(args.encoder)

    frame = load_external_frame(args.dataset)
    n_ai = int((frame["label"] == AI_LABEL).sum())
    n_human = int((frame["label"] == HUMAN_LABEL).sum())
    print(f"{args.dataset}: {len(frame)} rows ({n_human} human / {n_ai} AI)")

    maide = load_maide_up_english(args.maide_csv)
    maide_test = make_splits(maide, mode="grouped", seed=MAIDE_SPLIT_SEED).test
    print(f"MAiDE-up transfer target: grouped test split, {len(maide_test)} rows "
          f"(split seed {MAIDE_SPLIT_SEED})")
    del maide

    key = cell_config_key(args)
    results: dict = {
        "meta": {
            "args": vars(args),
            "env": env_info(),
            "cell_config": key,
            "external_rows": {"total": int(len(frame)), "human": n_human, "ai": n_ai},
            "n_maide_test": int(len(maide_test)),
            "fw_set_size": len(fw_set),
            "fw_sources": list(fw_set.sources),
        },
        "variants": {},
    }

    for variant in args.variants:
        cells: list[dict] = []
        for seed in args.seeds:
            cell_path = os.path.join(cell_dir, f"{variant}_s{seed}.json")
            cell = None
            if os.path.exists(cell_path):
                cached = load_json(cell_path)
                if cached.get("config") == key:
                    print(f"[cache] {variant} seed {seed} <- {cell_path}")
                    cell = cached
                else:
                    print(f"[cache] {variant} seed {seed}: stale config, retraining")
            if cell is None:
                print(f"== {args.dataset} | {variant} | seed {seed} ==", flush=True)
                cell = run_cell(
                    variant, seed, frame, maide_test, args, tokenizer, func_id, fw_set, device
                )
                cell["config"] = key
                save_json(cell_path, cell)
            cells.append(cell)
        results["variants"][variant] = {
            "per_seed": cells,
            "aggregated": {axis: aggregate_seeds([c[axis] for c in cells]) for axis in AXES},
        }
        save_json(out, results)  # incremental: survive interruption between variants
        gc.collect()

    save_json(out, results)
    print(f"\nSaved -> {out}")
    print_table(results, list(args.variants))


if __name__ == "__main__":
    main()
