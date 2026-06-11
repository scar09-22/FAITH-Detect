"""Which function words carry the AI-vs-human signal, and does it survive a domain shift?

For every word in the union function-word set we compute the Monroe, Colaresi & Quinn
(2008) "Fightin' Words" log-odds ratio with an informative Dirichlet prior between AI and
human texts, together with its z-score:

    delta_w = log[(y_aw + a_w) / (n_a + a0 - y_aw - a_w)]
            - log[(y_hw + a_w) / (n_h + a0 - y_hw - a_w)]
    var(delta_w) ~= 1/(y_aw + a_w) + 1/(y_hw + a_w),    z_w = delta_w / sqrt(var)

where y_iw are word counts, n_i corpus token totals, and the prior a_w is the combined
(AI+human) corpus distribution scaled to a0 = 0.01 * total tokens — so a_w = 0.01 * y_w.
Positive z means AI-leaning, negative human-leaning. Tokenisation is a simple whitespace
split with the same lower-case/punctuation-strip normalisation the masking pipeline uses
(`function_words._normalize_surface`), so the statistic is computed over exactly the
surface forms the [FUNC] mask removes.

Three questions, matching the paper's category-ablation story:
  1. In-domain (MAiDE-up English train+val, grouped split seed 0): which function words
     separate AI from human, and how strongly (z_in)?
  2. Cross-domain stability: the same statistic on the RAID reviews pool (z_ood); per-word
     sign agreement and the overall sign-agreement rate among words with |z_in| > 2. High
     agreement would mean function-word style is a transferable cue (an argument AGAINST
     masking it); low agreement supports treating it as a domain-specific shortcut.
  3. Per-category aggregates (FW_CATEGORIES): mean |z_in| and share of significant words,
     i.e. which grammatical categories concentrate the in-domain signal.

Outputs a JSON with per-word records + category aggregates + agreement stats, and a
two-panel figure: (a) top function words by |z_in| as diverging bars, (b) z_in vs z_ood
scatter with quadrant shading.

Example:
  python scripts/fw_stats.py --data_csv data/all_data.csv \
      --ood_parquet results/cache/raid_reviews_pool.parquet \
      --out results/fw_stats.json --fig figures/17_fw_words.png
"""
import argparse
import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from faithdetect.data.maide_up import AI_LABEL, load_maide_up_english, make_splits
from faithdetect.function_words import FW_CATEGORIES, _normalize_surface, build_function_word_set
from faithdetect.utils.logging import env_info, save_json
from faithdetect.viz.figures import PUBLICATION_STYLE

Z_SIGNIFICANT = 2.0     # |z| threshold for "carries signal" (approx. two-sided p < 0.05)
FIG_TOP_WORDS = 20      # bars in panel (a)
AI_COLOR = "#c44e52"    # AI-leaning (z > 0)
HUMAN_COLOR = "#4c72b0" # human-leaning (z < 0)


def tokenize(text: str) -> list[str]:
    """Whitespace split + the masking pipeline's surface normalisation; drop empties."""
    return [w for w in (_normalize_surface(t) for t in str(text).split()) if w]


def corpus_counts(texts: pd.Series) -> tuple[Counter, int]:
    """Token counts over a corpus and the total token count (full vocabulary)."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return counts, int(sum(counts.values()))


def fightin_words(
    ai_counts: Counter,
    n_ai: int,
    hu_counts: Counter,
    n_hu: int,
    vocab: list[str],
    prior_scale: float = 0.01,
) -> dict[str, dict]:
    """Monroe et al. (2008) log-odds with informative Dirichlet prior, per word in `vocab`.

    The prior is the combined-corpus distribution scaled to a0 = prior_scale * total
    tokens, so each word's pseudo-count is a_w = prior_scale * (y_aw + y_hw). Words absent
    from BOTH corpora get no evidence and are reported with delta = z = 0.
    """
    total = n_ai + n_hu
    a0 = prior_scale * total
    out: dict[str, dict] = {}
    for w in vocab:
        y_a, y_h = ai_counts.get(w, 0), hu_counts.get(w, 0)
        a_w = prior_scale * (y_a + y_h)
        if a_w <= 0:
            out[w] = {"count_ai": 0, "count_human": 0, "delta": 0.0, "z": 0.0}
            continue
        delta = math.log((y_a + a_w) / (n_ai + a0 - y_a - a_w)) - math.log(
            (y_h + a_w) / (n_hu + a0 - y_h - a_w)
        )
        var = 1.0 / (y_a + a_w) + 1.0 / (y_h + a_w)
        out[w] = {
            "count_ai": int(y_a),
            "count_human": int(y_h),
            "delta": float(delta),
            "z": float(delta / math.sqrt(var)),
        }
    return out


def word_category(word: str) -> str:
    """Curated category of a function word ('other_stopwords' for union-only words)."""
    for cat, words in FW_CATEGORIES.items():
        if word in words:
            return cat
    return "other_stopwords"


def sign_agreement(records: list[dict], z_threshold: float) -> dict:
    """Sign-agreement stats among words significant in-domain and present OOD."""
    sig = [r for r in records if abs(r["z_in"]) > z_threshold]
    evaluable = [r for r in sig if (r["count_ai_ood"] + r["count_human_ood"]) > 0]
    agree = [r for r in evaluable if r["z_in"] * r["z_ood"] > 0]
    return {
        "z_threshold": z_threshold,
        "n_significant_in": len(sig),
        "n_evaluable_ood": len(evaluable),
        "n_sign_agree": len(agree),
        "sign_agreement_rate": (len(agree) / len(evaluable)) if evaluable else None,
    }


def render_figure(records: list[dict], agreement: dict, path: str) -> None:
    plt.rcParams.update(PUBLICATION_STYLE)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.5))

    # (a) Top function words by |z_in|, diverging horizontal bars.
    top = sorted(records, key=lambda r: abs(r["z_in"]), reverse=True)[:FIG_TOP_WORDS]
    top = top[::-1]  # strongest at the top of the axis
    words = [r["word"] for r in top]
    zs = [r["z_in"] for r in top]
    colors = [AI_COLOR if z > 0 else HUMAN_COLOR for z in zs]
    ax_a.barh(np.arange(len(words)), zs, color=colors)
    ax_a.set_yticks(np.arange(len(words)))
    ax_a.set_yticklabels(words, fontsize=9)
    ax_a.axvline(0.0, color="black", lw=0.8)
    for s in (-Z_SIGNIFICANT, Z_SIGNIFICANT):
        ax_a.axvline(s, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax_a.set_xlabel("log-odds z (in-domain)   <- human-leaning | AI-leaning ->")
    ax_a.set_title(f"(a) Top {len(words)} function words by |z| in-domain")

    # (b) Cross-domain stability scatter with agreeing-quadrant shading.
    z_in = np.array([r["z_in"] for r in records])
    z_ood = np.array([r["z_ood"] for r in records])
    lim_x = max(1.0, float(np.max(np.abs(z_in)))) * 1.08
    lim_y = max(1.0, float(np.max(np.abs(z_ood)))) * 1.08
    ax_b.axhspan(0, lim_y, xmin=0.5, xmax=1.0, color="#2c7d59", alpha=0.08)
    ax_b.axhspan(-lim_y, 0, xmin=0.0, xmax=0.5, color="#2c7d59", alpha=0.08)
    ax_b.axhspan(-lim_y, 0, xmin=0.5, xmax=1.0, color="#c44e52", alpha=0.08)
    ax_b.axhspan(0, lim_y, xmin=0.0, xmax=0.5, color="#c44e52", alpha=0.08)
    sig = np.abs(z_in) > Z_SIGNIFICANT
    ax_b.scatter(z_in[~sig], z_ood[~sig], s=12, color="grey", alpha=0.45,
                 label=f"|z_in| <= {Z_SIGNIFICANT:g}")
    ax_b.scatter(z_in[sig], z_ood[sig], s=22, color="#333333", alpha=0.85,
                 label=f"|z_in| > {Z_SIGNIFICANT:g}")
    # Annotate the most extreme in-domain words for readability.
    for r in sorted(records, key=lambda r: abs(r["z_in"]), reverse=True)[:8]:
        ax_b.annotate(r["word"], (r["z_in"], r["z_ood"]), fontsize=8,
                      xytext=(3, 3), textcoords="offset points")
    ax_b.axhline(0.0, color="black", lw=0.8)
    ax_b.axvline(0.0, color="black", lw=0.8)
    ax_b.set_xlim(-lim_x, lim_x)
    ax_b.set_ylim(-lim_y, lim_y)
    ax_b.set_xlabel("z in-domain (MAiDE-up)")
    ax_b.set_ylabel("z out-of-domain (RAID reviews)")
    rate = agreement["sign_agreement_rate"]
    rate_txt = f"{rate:.0%}" if rate is not None else "n/a"
    ax_b.set_title(f"(b) Cross-domain stability — sign agreement {rate_txt} "
                   f"(|z_in| > {Z_SIGNIFICANT:g}, n = {agreement['n_evaluable_ood']})")
    ax_b.legend(loc="best", fontsize=8)

    fig.suptitle("Function-word 'Fightin' Words' log-odds: in-domain signal vs OOD stability",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_csv", default="data/all_data.csv")
    ap.add_argument("--ood_parquet", default="results/cache/raid_reviews_pool.parquet",
                    help="RAID reviews pool with columns text,label,model")
    ap.add_argument("--out", default="results/fw_stats.json")
    ap.add_argument("--fig", default="figures/17_fw_words.png")
    ap.add_argument("--fw_definition", default="union")
    ap.add_argument("--prior_scale", type=float, default=0.01,
                    help="a0 = prior_scale * total tokens (Monroe et al. informative prior)")
    args = ap.parse_args()

    fw_set = build_function_word_set(args.fw_definition)
    vocab = sorted(fw_set.words)

    # In-domain: MAiDE-up English, grouped split seed 0, train+val portion (test untouched).
    splits = make_splits(load_maide_up_english(args.data_csv), "grouped", seed=0)
    in_df = pd.concat([splits.train, splits.val], ignore_index=True)
    ai_in, n_ai_in = corpus_counts(in_df.loc[in_df["label"] == AI_LABEL, "text"])
    hu_in, n_hu_in = corpus_counts(in_df.loc[in_df["label"] != AI_LABEL, "text"])

    # OOD: RAID reviews pool, AI vs human.
    ood_df = pd.read_parquet(args.ood_parquet)
    ai_ood, n_ai_ood = corpus_counts(ood_df.loc[ood_df["label"] == AI_LABEL, "text"])
    hu_ood, n_hu_ood = corpus_counts(ood_df.loc[ood_df["label"] != AI_LABEL, "text"])
    print(f"In-domain tokens: AI={n_ai_in} human={n_hu_in} | "
          f"OOD tokens: AI={n_ai_ood} human={n_hu_ood} | fw_set={len(fw_set)} words")

    stats_in = fightin_words(ai_in, n_ai_in, hu_in, n_hu_in, vocab, args.prior_scale)
    stats_ood = fightin_words(ai_ood, n_ai_ood, hu_ood, n_hu_ood, vocab, args.prior_scale)

    records: list[dict] = []
    for w in vocab:
        si, so = stats_in[w], stats_ood[w]
        present_both = (si["count_ai"] + si["count_human"] > 0) and (
            so["count_ai"] + so["count_human"] > 0
        )
        records.append({
            "word": w,
            "category": word_category(w),
            "count_ai_in": si["count_ai"], "count_human_in": si["count_human"],
            "count_ai_ood": so["count_ai"], "count_human_ood": so["count_human"],
            "delta_in": si["delta"], "z_in": si["z"],
            "delta_ood": so["delta"], "z_ood": so["z"],
            "sign_agree": bool(si["z"] * so["z"] > 0) if present_both else None,
        })
    records.sort(key=lambda r: abs(r["z_in"]), reverse=True)

    agreement = sign_agreement(records, Z_SIGNIFICANT)
    agreement_all = sign_agreement(records, 0.0)  # context: agreement over all evaluable words
    rate = agreement["sign_agreement_rate"]
    print(f"Sign agreement among |z_in|>{Z_SIGNIFICANT:g}: "
          f"{agreement['n_sign_agree']}/{agreement['n_evaluable_ood']}"
          + (f" = {rate:.3f}" if rate is not None else " (none evaluable)"))

    # Per-category aggregation (curated categories + union-only stopwords), aligned with
    # the category-masking ablation: which grammatical class concentrates the signal?
    categories: list[dict] = []
    for cat in [*FW_CATEGORIES.keys(), "other_stopwords"]:
        rs = [r for r in records if r["category"] == cat]
        present = [r for r in rs if r["count_ai_in"] + r["count_human_in"] > 0]
        n_sig = sum(1 for r in present if abs(r["z_in"]) > Z_SIGNIFICANT)
        agree = [r for r in present if abs(r["z_in"]) > Z_SIGNIFICANT and r["sign_agree"]]
        categories.append({
            "category": cat,
            "n_words": len(rs),
            "n_present_in": len(present),
            "mean_abs_z_in": float(np.mean([abs(r["z_in"]) for r in present])) if present else None,
            "share_significant": (n_sig / len(present)) if present else None,
            "n_significant": n_sig,
            "n_significant_sign_agree": len(agree),
            "mean_abs_z_ood": float(np.mean([abs(r["z_ood"]) for r in present])) if present else None,
        })
    categories.sort(key=lambda c: (c["mean_abs_z_in"] is None, -(c["mean_abs_z_in"] or 0.0)))
    print(f"{'category':<18}{'n':>5}{'mean|z_in|':>12}{'share |z|>2':>13}")
    for c in categories:
        mz = f"{c['mean_abs_z_in']:.2f}" if c["mean_abs_z_in"] is not None else "n/a"
        sh = f"{c['share_significant']:.2f}" if c["share_significant"] is not None else "n/a"
        print(f"{c['category']:<18}{c['n_present_in']:>5}{mz:>12}{sh:>13}")

    payload = {
        "meta": {
            "data_csv": args.data_csv,
            "ood_parquet": args.ood_parquet,
            "split": {"mode": "grouped", "seed": 0, "portion": "train+val"},
            "fw_definition": args.fw_definition,
            "fw_set_size": len(fw_set),
            "prior_scale": args.prior_scale,
            "z_significant": Z_SIGNIFICANT,
            "tokens": {"in_ai": n_ai_in, "in_human": n_hu_in,
                       "ood_ai": n_ai_ood, "ood_human": n_hu_ood},
            "n_texts": {"in_ai": int((in_df["label"] == AI_LABEL).sum()),
                        "in_human": int((in_df["label"] != AI_LABEL).sum()),
                        "ood_ai": int((ood_df["label"] == AI_LABEL).sum()),
                        "ood_human": int((ood_df["label"] != AI_LABEL).sum())},
            "sign_convention": "z > 0 means AI-leaning, z < 0 human-leaning",
            "env": env_info(),
        },
        "agreement": {"significant": agreement, "all_words": agreement_all},
        "categories": categories,
        "words": records,
    }
    save_json(args.out, payload)
    print(f"Saved stats -> {args.out}")
    render_figure(records, agreement, args.fig)


if __name__ == "__main__":
    main()
