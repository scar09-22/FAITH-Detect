"""Per-generator detection of NEW same-domain (Ollama) generators with the EXISTING
trained variants — no retraining.

Protocol
--------
* Each checkpoint (a ``ReviewDetector`` state_dict trained by run_grid.py) is reloaded into
  a fresh ``ReviewDetector(ModelConfig(variant=..., max_length=...))`` with the variant's
  own collator, so hard masking is applied exactly as at training time. Checkpoints saved
  with ``--max_length 112`` MUST be evaluated with ``--max_length 112``.
* AI rows come from the Ollama parquet (one block per generator, label=1). The PRIMARY
  negative class is the MAiDE-up grouped seed-0 TEST-split humans (same style and hotels
  distribution as training-time evaluation, never trained on); the RAID pool humans serve
  as a SECONDARY negative set to show sensitivity to the human reference.
* Per (variant, generator) we report detection macro-F1 against both negative sets plus
  the TPR at the model's own 0.5 threshold (argmax over the binary softmax). TPR depends
  only on the AI rows, so it is reported once per (variant, generator).
* Outputs: a JSON results file and a grouped-bar figure (figures/18_new_generators.png)
  in the repository's publication style.

Example:
  python scripts/eval_generators.py --gen_parquet results/cache/ollama_reviews.parquet \
      --pool results/cache/raid_reviews_pool.parquet \
      --checkpoints baseline=results/cells/baseline_s0_model.pt \
                    hardmask=results/cells/hardmask_s0_model.pt \
                    softreg=results/cells/softreg_s0_model.pt \
      --max_length 112 --device cpu --out results/new_generators.json
"""
from __future__ import annotations

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from faithdetect.data.maide_up import AI_LABEL, HUMAN_LABEL, load_maide_up_english, make_splits
from faithdetect.evaluate import classification_metrics, predict
from faithdetect.function_words import build_function_word_set
from faithdetect.models import ModelConfig, ReviewDetector, build_tokenizer
from faithdetect.train import make_collator
from faithdetect.utils.logging import env_info, save_json

MAIDE_SEG = "human_maide_test"
POOL_SEG = "human_pool"
SPLIT_SEED = 0  # the grouped split whose TEST humans form the primary negative class


def parse_checkpoints(items: list[str]) -> list[tuple[str, str]]:
    """Parse 'variant=path' pairs; the key must be a model variant (baseline/hardmask/softreg)."""
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--checkpoints items must be variant=path, got {item!r}")
        name, path = item.split("=", 1)
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        out.append((name, path))
    return out


def build_eval_frame(gen_parquet: str, pool_parquet: str, data_csv: str) -> pd.DataFrame:
    """One frame holding both negative sets and all generator AI rows.

    The 'model' column doubles as the segment id: MAIDE_SEG / POOL_SEG for the two human
    sets, and the generator tag for each AI block. A single forward pass per variant then
    covers every (generator, negative-set) pairing.
    """
    gen = pd.read_parquet(gen_parquet)
    gen = gen[gen["label"] == AI_LABEL][["text", "label", "model"]].copy()
    if gen.empty:
        raise ValueError(f"no AI rows in {gen_parquet}")
    pool = pd.read_parquet(pool_parquet)
    pool_humans = pool[pool["label"] == HUMAN_LABEL][["text", "label"]].assign(model=POOL_SEG)
    splits = make_splits(load_maide_up_english(data_csv), mode="grouped", seed=SPLIT_SEED)
    test = splits.test
    maide_humans = test[test["label"] == HUMAN_LABEL][["text", "label"]].assign(model=MAIDE_SEG)
    if maide_humans.empty or pool_humans.empty:
        raise ValueError("one of the negative sets is empty")
    return pd.concat([maide_humans, pool_humans, gen], ignore_index=True)


def per_generator_metrics(pred: dict, seg: np.ndarray, generators: list[str]) -> dict:
    """Slice one variant's predictions into per-(generator, negative-set) metrics."""
    y_true = np.asarray(pred["y_true"])
    y_pred = np.asarray(pred["y_pred"])
    p_ai = np.asarray(pred["p_ai"])
    masks = {MAIDE_SEG: seg == MAIDE_SEG, POOL_SEG: seg == POOL_SEG}
    out: dict = {
        "human_fpr": {  # fraction of humans flagged AI at the model's own 0.5 threshold
            "maide_test": float((y_pred[masks[MAIDE_SEG]] == 1).mean()),
            "pool": float((y_pred[masks[POOL_SEG]] == 1).mean()),
        },
        "generators": {},
    }
    for g in generators:
        gm = seg == g
        entry = {
            "n_ai": int(gm.sum()),
            "tpr_at_0.5": float((y_pred[gm] == 1).mean()),
        }
        for key, neg in (("vs_maide_test_humans", MAIDE_SEG), ("vs_pool_humans", POOL_SEG)):
            m = masks[neg] | gm
            entry[key] = classification_metrics(y_true[m], y_pred[m], p_ai[m])
        out["generators"][g] = entry
    return out


def make_figure(per_variant: dict, generators: list[str], path: str) -> None:
    """Grouped bars in the repo viz style: macro-F1 (primary negatives) and TPR@0.5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from faithdetect.viz.figures import (
        PUBLICATION_STYLE, VARIANT_COLOR, VARIANT_LABEL, VARIANT_ORDER,
    )

    plt.rcParams.update(PUBLICATION_STYLE)
    variants = [v for v in VARIANT_ORDER if v in per_variant]
    if not variants or not generators:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(11.0, 2.4 * len(generators) + 5), 4.6))
    x = np.arange(len(generators))
    w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        gd = per_variant[v]["generators"]
        f1 = [gd[g]["vs_maide_test_humans"]["f1_macro"] for g in generators]
        tpr = [gd[g]["tpr_at_0.5"] for g in generators]
        off = (i - (len(variants) - 1) / 2) * w
        ax1.bar(x + off, f1, w, label=VARIANT_LABEL.get(v, v), color=VARIANT_COLOR.get(v, "#999"))
        ax2.bar(x + off, tpr, w, label=VARIANT_LABEL.get(v, v), color=VARIANT_COLOR.get(v, "#999"))
    for ax, ylab, title in (
        (ax1, "Detection F1 (macro)", "F1 vs. MAiDE-up test humans"),
        (ax2, "TPR at threshold 0.5", "AI recall per generator"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(generators, rotation=30, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("New same-domain generators (Ollama) vs. existing detectors", fontweight="bold")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen_parquet", default="results/cache/ollama_reviews.parquet",
                    help="generated AI reviews with columns text,label,model")
    ap.add_argument("--pool", default="results/cache/raid_reviews_pool.parquet",
                    help="RAID reviews pool (its label==0 rows are the secondary negatives)")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="variant=path pairs, e.g. baseline=results/cells/baseline_s0_model.pt")
    ap.add_argument("--data_csv", default="data/all_data.csv")
    ap.add_argument("--encoder", default="distilroberta-base")
    ap.add_argument("--max_length", type=int, default=112,
                    help="MUST match the checkpoints' training max_length (112 for results/cells)")
    ap.add_argument("--fw_definition", default="union")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/new_generators.json")
    ap.add_argument("--fig", default="figures/18_new_generators.png")
    args = ap.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints)
    device = torch.device(args.device)
    frame = build_eval_frame(args.gen_parquet, args.pool, args.data_csv)
    seg = frame["model"].to_numpy()
    generators = sorted(g for g in pd.unique(seg) if g not in (MAIDE_SEG, POOL_SEG))
    print(f"Eval frame: {len(frame)} rows | negatives: "
          f"{int((seg == MAIDE_SEG).sum())} MAiDE test humans, "
          f"{int((seg == POOL_SEG).sum())} pool humans | generators: {generators}")

    fw_set = build_function_word_set(args.fw_definition)
    tokenizer, func_id = build_tokenizer(args.encoder)

    per_variant: dict[str, dict] = {}
    for variant, ckpt in checkpoints:
        model_cfg = ModelConfig(encoder_name=args.encoder, variant=variant,
                                max_length=args.max_length)
        model = ReviewDetector(model_cfg, vocab_size=len(tokenizer))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device).eval()
        # The variant's own collator (hard masking iff variant == 'hardmask').
        collator = make_collator(model_cfg, tokenizer, func_id, fw_set)
        print(f"[{variant}] predicting {len(frame)} rows from {ckpt}")
        pred = predict(model, frame, collator, device, batch_size=args.batch_size)
        per_variant[variant] = per_generator_metrics(pred, seg, generators)
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    payload = {
        "config": {
            "gen_parquet": args.gen_parquet, "pool": args.pool, "data_csv": args.data_csv,
            "checkpoints": dict(checkpoints), "encoder": args.encoder,
            "max_length": args.max_length, "fw_definition": args.fw_definition,
            "device": args.device, "split_seed": SPLIT_SEED, "split_mode": "grouped",
            "primary_negatives": "MAiDE-up grouped seed-0 test-split humans",
            "secondary_negatives": "RAID reviews pool humans",
        },
        "env": env_info(),
        "negatives": {
            "maide_test_humans": int((seg == MAIDE_SEG).sum()),
            "pool_humans": int((seg == POOL_SEG).sum()),
        },
        "per_variant": per_variant,
    }
    save_json(args.out, payload)
    print(f"\nSaved -> {args.out}")

    make_figure(per_variant, generators, args.fig)

    print("\nvariant x generator (macro-F1 vs MAiDE test humans | TPR@0.5):")
    for variant, _ in checkpoints:
        gd = per_variant[variant]["generators"]
        for g in generators:
            print(f"  {variant:<9} {g:<16} "
                  f"f1={gd[g]['vs_maide_test_humans']['f1_macro']:.4f} "
                  f"tpr={gd[g]['tpr_at_0.5']:.4f}")


if __name__ == "__main__":
    main()
