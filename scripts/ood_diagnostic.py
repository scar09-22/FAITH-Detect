"""Diagnose WHY a trained detector succeeds in-domain but fails out-of-domain.

Loads one trained checkpoint, attributes a small sample of in-domain (MAiDE-up English
test) and out-of-domain (RAID) texts with Integrated Gradients on the real model, and
compares WHICH CONTENT WORDS carry the attribution mass in each set. If the in-domain
top-attributed vocabulary (e.g. hotel-review words) carries little mass OOD — or, worse,
still dominates OOD where it is off-topic — the accuracy drop is explained by lexical
shortcut learning rather than by a transferable human-vs-AI signal.

Outputs a JSON with per-text records + per-set aggregates (accuracy, attributed-word
counters, vocabulary carry-over) and a two-panel figure of the top attributed content
words in-domain vs OOD.

Example:
  python scripts/ood_diagnostic.py --checkpoint results/cells/hardmask_s0_model.pt \
      --variant hardmask --n_per_set 40 --ig_steps 12 --device cpu
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import _bootstrap  # noqa: F401
from faithdetect.data.maide_up import load_maide_up_english, make_splits
from faithdetect.evaluate import classification_metrics
from faithdetect.explain.attributions import FaithfulExplainer
from faithdetect.function_words import build_function_word_set
from faithdetect.models import ModelConfig, ReviewDetector, build_tokenizer
from faithdetect.utils.logging import save_json
from faithdetect.viz.figures import PUBLICATION_STYLE, VARIANT_LABEL

TOP_K_PER_TEXT = 5      # top content words recorded per text
CARRYOVER_VOCAB = 50    # size of the in-domain reference vocabulary
FIG_TOP_WORDS = 15      # bars per panel


def balanced_head(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """First `n` rows, balanced across labels when both classes are present."""
    if "label" not in df.columns or df["label"].nunique() < 2:
        return df.head(n).reset_index(drop=True)
    per_class = max(1, n // df["label"].nunique())
    parts = [g.head(per_class) for _, g in df.groupby("label", sort=True)]
    return pd.concat(parts).head(n).reset_index(drop=True)


def diagnose_set(name: str, df: pd.DataFrame, explainer: FaithfulExplainer, ig_steps: int) -> dict:
    """Attribute every text in `df`; aggregate where the content-word attribution mass goes."""
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    generators = df["model"].astype(str).tolist() if "model" in df.columns else [None] * len(texts)

    records: list[dict] = []
    word_mass: Counter = Counter()    # lowercased content word -> total |attribution|
    word_freq: Counter = Counter()    # lowercased content word -> occurrence count
    top_counter: Counter = Counter()  # words appearing in a text's top-k explanation
    y_true: list[int] = []
    y_pred: list[int] = []
    p_ai: list[float] = []
    n_failed = 0

    for i, (text, label) in enumerate(zip(texts, labels)):
        if i % 10 == 0:
            print(f"  [{name}] attributing text {i + 1}/{len(texts)}", flush=True)
        # One pathological text (length, encoding, backend instability) must not kill the run.
        try:
            attr = explainer.integrated_gradients(text, n_steps=ig_steps)
        except Exception as e:
            n_failed += 1
            print(f"  [{name}] text {i} failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        pairs = attr.content_explanation(top_k=None)   # all content words, |score| desc
        for w, s in pairs:
            lw = w.lower()
            word_mass[lw] += abs(float(s))
            word_freq[lw] += 1
        top_pairs = pairs[:TOP_K_PER_TEXT]
        top_counter.update(w.lower() for w, _ in top_pairs)
        y_true.append(label)
        y_pred.append(int(attr.predicted_label))
        p_ai.append(float(attr.p_ai))
        records.append(
            {
                "set": name,
                "row": i,
                "generator": generators[i],
                "true_label": int(label),
                "predicted_label": int(attr.predicted_label),
                "p_ai": float(attr.p_ai),
                "correct": bool(int(attr.predicted_label) == int(label)),
                "top_content_words": [[w, float(s)] for w, s in top_pairs],
            }
        )

    metrics = classification_metrics(y_true, y_pred, np.asarray(p_ai)) if y_true else {}
    return {
        "name": name,
        "n_texts": len(texts),
        "n_attributed": len(records),
        "n_failed": n_failed,
        "metrics": metrics,
        # most_common() order so the JSON reads top-down and the figure can slice the head.
        "top_word_counter": dict(top_counter.most_common()),
        "word_frequency": dict(word_freq.most_common()),
        "word_attribution_mass": dict(word_mass.most_common()),
        "total_content_mass": float(sum(word_mass.values())),
        "records": records,
    }


def mass_fraction_on_vocab(agg: dict, vocab: set) -> float:
    """Fraction of a set's total content |attribution| carried by `vocab` words."""
    total = agg["total_content_mass"]
    if total <= 0:
        return 0.0
    return float(sum(m for w, m in agg["word_attribution_mass"].items() if w in vocab) / total)


def render_figure(indomain: dict, ood: dict, vocab: set, variant: str, path: str) -> None:
    plt.rcParams.update(PUBLICATION_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    in_vocab_color, out_vocab_color = "#2c7d59", "#c44e52"
    for ax, agg in zip(axes, (indomain, ood)):
        top = list(agg["word_attribution_mass"].items())[:FIG_TOP_WORDS]
        acc = agg["metrics"].get("accuracy")
        acc_txt = f"{acc:.2f}" if acc is not None else "n/a"
        ax.set_title(f"{agg['name']} (accuracy = {acc_txt}, n = {agg['n_attributed']})")
        if not top:
            ax.text(0.5, 0.5, "no successful attributions", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        total = agg["total_content_mass"] or 1.0
        words = [w for w, _ in top][::-1]            # largest share at the top
        shares = [m / total for _, m in top][::-1]
        colors = [in_vocab_color if w in vocab else out_vocab_color for w in words]
        ax.barh(np.arange(len(words)), shares, color=colors)
        ax.set_yticks(np.arange(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.set_xlabel("Share of total |attribution| on content words")
    fig.suptitle(
        f"Where attribution mass goes in vs out of domain — {VARIANT_LABEL.get(variant, variant)}",
        fontweight="bold", y=1.02,
    )
    fig.legend(
        handles=[
            Patch(color=in_vocab_color, label=f"in in-domain top-{CARRYOVER_VOCAB} vocabulary"),
            Patch(color=out_vocab_color, label="outside in-domain vocabulary"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="trained state_dict .pt")
    ap.add_argument("--variant", required=True, choices=["baseline", "hardmask", "softreg"],
                    help="must match the checkpoint")
    ap.add_argument("--encoder", default="distilroberta-base")
    ap.add_argument("--data_csv", default="data/all_data.csv")
    ap.add_argument("--ood_parquet", default="results/cache/raid_ood.parquet",
                    help="RAID OOD parquet with columns text,label,model")
    ap.add_argument("--n_per_set", type=int, default=40)
    ap.add_argument("--max_length", type=int, default=128,
                    help="tokenizer truncation length; MUST match training "
                         "(see results/cells/cfg.json — the local checkpoints used 112)")
    ap.add_argument("--ig_steps", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_json", default=None,
                    help="default results/ood_diagnostic_<variant>.json")
    ap.add_argument("--out_fig", default=None,
                    help="default figures/15_ood_diagnostic_<variant>.png")
    args = ap.parse_args()
    out_json = args.out_json or f"results/ood_diagnostic_{args.variant}.json"
    out_fig = args.out_fig or f"figures/15_ood_diagnostic_{args.variant}.png"

    device = torch.device(args.device)
    # macOS Accelerate/BLAS SIGBUSes on single-example attribution forwards when multi-
    # threaded; mirror experiment._compute_xai_for_variant and run single-threaded on CPU.
    if device.type == "cpu":
        torch.set_num_threads(1)

    # Rebuild tokenizer / function-word set exactly as training did.
    fw_set = build_function_word_set("union")
    tokenizer, func_id = build_tokenizer(args.encoder)
    mcfg = ModelConfig(encoder_name=args.encoder, variant=args.variant,
                       max_length=args.max_length)
    model = ReviewDetector(mcfg, vocab_size=len(tokenizer))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device).eval()
    explainer = FaithfulExplainer(model, mcfg, tokenizer, func_id, fw_set, device)

    splits = make_splits(load_maide_up_english(args.data_csv), "grouped", seed=0)
    # The grouped test split is ordered with all human rows first; a plain head() would be
    # single-class, making in-domain accuracy and the human-vs-AI attribution mix meaningless.
    in_df = balanced_head(splits.test, args.n_per_set)
    ood_df = balanced_head(pd.read_parquet(args.ood_parquet), args.n_per_set)
    print(f"Diagnosing {args.variant} from {args.checkpoint}: "
          f"{len(in_df)} in-domain + {len(ood_df)} OOD texts, ig_steps={args.ig_steps}")

    indomain = diagnose_set("in-domain (MAiDE-up test)", in_df, explainer, args.ig_steps)
    ood = diagnose_set("OOD (RAID)", ood_df, explainer, args.ig_steps)

    # Vocabulary carry-over: how much of each set's attribution mass falls on the 50
    # most-frequent in-domain content words. Low OOD carry-over with low OOD accuracy
    # means the in-domain evidence vocabulary simply does not exist out of domain.
    vocab = set(list(indomain["word_frequency"].keys())[:CARRYOVER_VOCAB])
    carryover = {
        "vocab_size": len(vocab),
        "indomain_top_words": sorted(vocab),
        "indomain_mass_fraction": mass_fraction_on_vocab(indomain, vocab),
        "ood_mass_fraction": mass_fraction_on_vocab(ood, vocab),
    }
    print(f"Accuracy: in-domain={indomain['metrics'].get('accuracy')} "
          f"OOD={ood['metrics'].get('accuracy')}")
    print(f"Attribution mass on in-domain top-{CARRYOVER_VOCAB} vocabulary: "
          f"in-domain={carryover['indomain_mass_fraction']:.3f} "
          f"OOD={carryover['ood_mass_fraction']:.3f}")

    payload = {
        "meta": {
            "checkpoint": args.checkpoint,
            "variant": args.variant,
            "encoder": args.encoder,
            "data_csv": args.data_csv,
            "ood_parquet": args.ood_parquet,
            "n_per_set": args.n_per_set,
            "max_length": args.max_length,
            "ig_steps": args.ig_steps,
            "device": args.device,
            "fw_set_size": len(fw_set),
            "top_k_per_text": TOP_K_PER_TEXT,
        },
        "indomain": indomain,
        "ood": ood,
        "vocab_carryover": carryover,
    }
    save_json(out_json, payload)
    print(f"Saved diagnostic -> {out_json}")
    render_figure(indomain, ood, vocab, args.variant, out_fig)


if __name__ == "__main__":
    main()
