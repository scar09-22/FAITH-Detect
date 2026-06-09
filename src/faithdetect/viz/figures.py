"""Publication-quality figures, rendered STRICTLY from a results dict.

No figure contains hard-coded data (fixes flaw F2): every number comes from
`results/*.json` produced by `experiment.run_full_experiment`. Each function is defensive
and skips gracefully if its inputs are absent (so partial smoke results still render).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PUBLICATION_STYLE = {
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "figure.autolayout": False,
}

VARIANT_ORDER = ["baseline", "softreg", "hardmask"]
VARIANT_LABEL = {"baseline": "Baseline", "softreg": "SoftReg", "hardmask": "Hard-Mask (FAITH)"}
VARIANT_COLOR = {"baseline": "#6c757d", "softreg": "#e08214", "hardmask": "#2c7d59"}
BASELINE_COLOR = {"tfidf_lr": "#8e7cc3", "tfidf_lr_content": "#b39ddb"}


def _variants(results):
    return [v for v in VARIANT_ORDER if v in results.get("variants", {})]


def _agg(results, variant, axis, metric):
    try:
        return results["variants"][variant][axis]["aggregated"][metric]
    except Exception:
        return None


def _errbar(interval):
    if interval is None:
        return 0.0, [[0.0], [0.0]]
    m = interval["mean"]
    return m, [[m - interval["lo"]], [interval["hi"] - m]]


def _save(fig, path):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
def fig_dataset_overview(results, path):
    sd = results.get("split_describe")
    if not sd:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    splits = ["train", "val", "test"]
    human = [sd[s]["n_human"] for s in splits]
    ai = [sd[s]["n_ai"] for s in splits]
    x = np.arange(len(splits))
    ax.bar(x, human, label="Human", color="#4c78a8")
    ax.bar(x, ai, bottom=human, label="AI (GPT-4)", color="#f58518")
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Reviews")
    ax.set_title(f"MAiDE-up English splits ({sd['mode']}, leakage={sd['hotel_leakage_train_test']} hotels)")
    for i, (h, a) in enumerate(zip(human, ai)):
        ax.text(i, h + a + 5, str(h + a), ha="center", fontsize=9)
    ax.legend()
    _save(fig, path)


def fig_leakage_gap(results, path):
    lk = results.get("leakage")
    if not lk:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    modes = ["grouped", "random"]
    metrics = ["f1_macro", "accuracy", "roc_auc"]
    x = np.arange(len(metrics)); w = 0.35
    for i, mode in enumerate(modes):
        vals = [lk[mode].get(m, np.nan) for m in metrics]
        ax.bar(x + (i - 0.5) * w, vals, w,
               label=f"{mode} split", color=["#2c7d59", "#c0392b"][i])
    ax.set_xticks(x); ax.set_xticklabels(["F1 (macro)", "Accuracy", "ROC-AUC"])
    ax.set_ylim(0, 1.02); ax.set_ylabel("Score")
    ax.set_title("Leakage inflates scores: grouped vs. random split (baseline)")
    ax.legend()
    _save(fig, path)


def fig_indomain_performance(results, path):
    variants = _variants(results)
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels, means, errs, colors = [], [], [], []
    for v in variants:
        it = _agg(results, v, "indomain", "f1_macro")
        if it is None:
            continue
        m, e = _errbar(it)
        labels.append(VARIANT_LABEL[v]); means.append(m); errs.append(e); colors.append(VARIANT_COLOR[v])
    # classic baselines (single point, no CI)
    for bname, b in results.get("baselines", {}).items():
        f1 = b["metrics"].get("f1_macro")
        if f1 is not None:
            labels.append(bname); means.append(f1); errs.append([[0], [0]]); colors.append(BASELINE_COLOR.get(bname, "#999"))
    x = np.arange(len(labels))
    yerr = np.array([[e[0][0] for e in errs], [e[1][0] for e in errs]])
    ax.bar(x, means, yerr=yerr, capsize=4, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1 (macro)")
    ax.set_title("In-domain performance (mean ± 95% CI over seeds)")
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    _save(fig, path)


def fig_confusion_matrices(results, path):
    rp = results.get("ref_preds", {})
    variants = [v for v in _variants(results) if v in rp and "indomain" in rp[v]]
    if not variants:
        return
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 3.6))
    if len(variants) == 1:
        axes = [axes]
    for ax, v in zip(axes, variants):
        d = rp[v]["indomain"]
        cm = confusion_matrix(d["y_true"], d["y_pred"])
        im = ax.imshow(cm, cmap="Blues")
        for (r, c), val in np.ndenumerate(cm):
            ax.text(c, r, int(val), ha="center", va="center",
                    color="white" if val > cm.max() / 2 else "black", fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Human", "AI"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Human", "AI"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(VARIANT_LABEL[v])
        ax.grid(False)
    fig.suptitle("In-domain confusion matrices", fontweight="bold")
    _save(fig, path)


def fig_roc_pr(results, path):
    rp = results.get("ref_preds", {})
    variants = [v for v in _variants(results) if v in rp]
    if not variants:
        return
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for v in variants:
        d = rp[v]["indomain"]
        y, p = np.array(d["y_true"]), np.array(d["p_ai"])
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=VARIANT_COLOR[v], label=f"{VARIANT_LABEL[v]} (AUC={roc_auc_score(y,p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax2.plot(rec, prec, color=VARIANT_COLOR[v], label=f"{VARIANT_LABEL[v]} (AP={average_precision_score(y,p):.3f})")
    for bname, b in results.get("baselines", {}).items():
        if "p_ai" not in b:
            continue
        y, p = np.array(b["y_true"]), np.array(b["p_ai"])
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, "--", color=BASELINE_COLOR.get(bname, "#999"), label=f"{bname} (AUC={roc_auc_score(y,p):.3f})")
    ax1.plot([0, 1], [0, 1], ":", color="gray")
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate"); ax1.set_title("ROC (in-domain)"); ax1.legend(fontsize=8)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall (in-domain)"); ax2.legend(fontsize=8)
    _save(fig, path)


def fig_ood_transfer(results, path):
    variants = [v for v in _variants(results) if _agg(results, v, "ood", "f1_macro")]
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(variants)); w = 0.38
    for j, axis in enumerate(["indomain", "ood"]):
        means, yerr = [], [[], []]
        for v in variants:
            m, e = _errbar(_agg(results, v, axis, "f1_macro"))
            means.append(m); yerr[0].append(e[0][0]); yerr[1].append(e[1][0])
        ax.bar(x + (j - 0.5) * w, means, w, yerr=yerr, capsize=4,
               label=["In-domain", "OOD (RAID)"][j], color=["#4c78a8", "#e45756"][j])
    ax.set_xticks(x); ax.set_xticklabels([VARIANT_LABEL[v] for v in variants], rotation=15)
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1 (macro)")
    ax.set_title("Cross-distribution generalization (in-domain vs. RAID OOD)")
    ax.legend()
    _save(fig, path)


def fig_robustness(results, path):
    variants = _variants(results)
    if not variants:
        return
    attacks = list(results["variants"][variants[0]].get("attacks", {}).keys())
    if not attacks:
        return
    conditions = ["clean"] + attacks
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(conditions)); w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        means, yerr = [], [[], []]
        clean = _agg(results, v, "indomain", "f1_macro")
        m, e = _errbar(clean); means.append(m); yerr[0].append(e[0][0]); yerr[1].append(e[1][0])
        for a in attacks:
            it = results["variants"][v]["attacks"][a]["aggregated"].get("f1_macro")
            m, e = _errbar(it); means.append(m); yerr[0].append(e[0][0]); yerr[1].append(e[1][0])
        ax.bar(x + (i - (len(variants) - 1) / 2) * w, means, w, yerr=yerr, capsize=3,
               label=VARIANT_LABEL[v], color=VARIANT_COLOR[v])
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_", " ") for c in conditions])
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1 (macro)")
    ax.set_title("Robustness to text attacks (function-word attack should not move Hard-Mask)")
    ax.legend(fontsize=9)
    _save(fig, path)


def fig_fw_attribution_mass(results, path):
    fwm = results.get("fw_mass", {})
    variants = [v for v in VARIANT_ORDER if v in fwm and fwm[v].get("values")]
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    data = [np.array(fwm[v]["values"]) * 100 for v in variants]
    bp = ax.boxplot(data, positions=range(len(variants)), widths=0.55, patch_artist=True, showmeans=True)
    for patch, v in zip(bp["boxes"], variants):
        patch.set_facecolor(VARIANT_COLOR[v]); patch.set_alpha(0.6)
    for i, v in enumerate(variants):
        ax.text(i, np.mean(data[i]) + 2, f"{np.mean(data[i]):.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(variants))); ax.set_xticklabels([VARIANT_LABEL[v] for v in variants])
    ax.set_ylabel("Attribution mass on function words (%)")
    ax.set_title("Function words carry ~0 attribution under Hard-Mask (IG)")
    _save(fig, path)


def fig_fw_identity_sensitivity(results, path):
    """The core requirement, directly measured: |Δp(AI)| when function-word IDENTITY is
    swapped. Exactly 0 for Hard-Mask (its decision cannot depend on which function word)."""
    sens = results.get("fw_identity_sensitivity", {})
    variants = [v for v in VARIANT_ORDER if v in sens and sens[v].get("values")]
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    means = [np.mean(sens[v]["values"]) * 100 for v in variants]
    errs = []
    for v in variants:
        ci = sens[v].get("ci")
        if ci:
            errs.append([(ci["mean"] - ci["lo"]) * 100, (ci["hi"] - ci["mean"]) * 100])
        else:
            errs.append([0, 0])
    yerr = np.array(errs).T
    bars = ax.bar(range(len(variants)), means, yerr=yerr, capsize=4,
                  color=[VARIANT_COLOR[v] for v in variants])
    for i, m in enumerate(means):
        ax.text(i, m + max(means) * 0.02 + 0.05, f"{m:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(variants))); ax.set_xticklabels([VARIANT_LABEL[v] for v in variants])
    ax.set_ylabel("|Δ p(AI)| when function words are swapped (%)")
    ax.set_title("Function-word identity cannot move the Hard-Mask decision")
    _save(fig, path)


def fig_faithfulness(results, path):
    variants = [v for v in _variants(results) if "faithfulness" in results["variants"][v]]
    if not variants:
        return
    metrics = ["comprehensiveness", "sufficiency", "deletion_auc", "insertion_auc"]
    labels = ["Comprehensiveness↑", "Sufficiency↓", "Deletion-AUC↓", "Insertion-AUC↑"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(metrics)); w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        s = results["variants"][v]["faithfulness"]["summary"]
        vals = [s.get(m, np.nan) for m in metrics]
        ax.bar(x + (i - (len(variants) - 1) / 2) * w, vals, w, label=VARIANT_LABEL[v], color=VARIANT_COLOR[v])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_title("Explanation faithfulness (IG, content words)")
    ax.legend(fontsize=9)
    _save(fig, path)


def fig_deletion_insertion_curves(results, path):
    variants = [v for v in _variants(results)
                if results["variants"][v].get("faithfulness", {}).get("curves", {}).get("deletion_mean")]
    if not variants:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for v in variants:
        c = results["variants"][v]["faithfulness"]["curves"]
        ax1.plot(c["grid"], c["deletion_mean"], marker="o", ms=3, color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
        ax2.plot(c["grid"], c["insertion_mean"], marker="o", ms=3, color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
    ax1.set_title("Deletion (content words, important first) ↓"); ax1.set_xlabel("Fraction removed"); ax1.set_ylabel("p(predicted class)")
    ax2.set_title("Insertion (content words, important first) ↑"); ax2.set_xlabel("Fraction inserted"); ax2.set_ylabel("p(predicted class)")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    _save(fig, path)


def fig_calibration(results, path):
    rp = results.get("ref_preds", {})
    variants = [v for v in _variants(results) if v in rp]
    if not variants:
        return
    from ..utils.stats import expected_calibration_error
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ":", color="gray", label="Perfect")
    for v in variants:
        d = rp[v]["indomain"]
        cal = expected_calibration_error(d["y_true"], d["p_ai"], n_bins=10)
        ax.plot(cal["bin_confidence"], cal["bin_accuracy"], marker="o", ms=4,
                color=VARIANT_COLOR[v], label=f"{VARIANT_LABEL[v]} (ECE={cal['ece']:.3f})")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Calibration (reliability diagram)")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_learning_curves(results, path):
    variants = [v for v in _variants(results) if results["variants"][v].get("histories")]
    if not variants:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for v in variants:
        hs = results["variants"][v]["histories"]
        max_ep = max(len(h["train_loss"]) for h in hs)
        loss = np.full((len(hs), max_ep), np.nan); f1 = np.full((len(hs), max_ep), np.nan)
        for i, h in enumerate(hs):
            loss[i, :len(h["train_loss"])] = h["train_loss"]
            f1[i, :len(h["val_macro_f1"])] = h["val_macro_f1"]
        ep = np.arange(1, max_ep + 1)
        ax1.plot(ep, np.nanmean(loss, 0), marker="o", color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
        ax2.plot(ep, np.nanmean(f1, 0), marker="o", color=VARIANT_COLOR[v], label=VARIANT_LABEL[v])
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss"); ax1.set_title("Training loss"); ax1.legend(fontsize=8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val F1 (macro)"); ax2.set_title("Validation F1"); ax2.legend(fontsize=8)
    _save(fig, path)


def fig_embedding_projection(results, path):
    emb = results.get("embeddings", {})
    variants = [v for v in VARIANT_ORDER if v in emb]
    if not variants:
        return
    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 4))
    if len(variants) == 1:
        axes = [axes]
    for ax, v in zip(axes, variants):
        c = np.array(emb[v]["coords"]); y = np.array(emb[v]["labels"])
        for lab, col, name in [(0, "#4c78a8", "Human"), (1, "#f58518", "AI")]:
            m = y == lab
            ax.scatter(c[m, 0], c[m, 1], s=12, c=col, alpha=0.6, label=name)
        ax.set_title(f"{VARIANT_LABEL[v]} ({emb[v]['method'].upper()})")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.legend(fontsize=8)
    fig.suptitle("Encoder representations of the test set", fontweight="bold")
    _save(fig, path)


def fig_example_explanation(results, path, max_examples=3):
    exs = results.get("example_explanations", [])
    # Prefer hard-mask examples (function words greyed, content highlighted).
    exs = [e for e in exs if e["variant"] == "hardmask"][:max_examples] or exs[:max_examples]
    if not exs:
        return
    fig, axes = plt.subplots(len(exs), 1, figsize=(11, 1.7 * len(exs) + 0.5))
    if len(exs) == 1:
        axes = [axes]
    import matplotlib.cm as cm
    for ax, ex in zip(axes, exs):
        ax.axis("off")
        scores = np.array(ex["word_scores"], dtype=float)
        denom = np.abs(scores[[not f for f in ex["word_is_fw"]]]).max() if any(not f for f in ex["word_is_fw"]) else 1.0
        denom = denom or 1.0
        x, yline = 0.0, 0.6
        for w, is_fw, s in zip(ex["words"], ex["word_is_fw"], ex["word_scores"]):
            if is_fw or not any(ch.isalnum() for ch in w):
                color = "#e9e9e9"  # function words greyed out
            else:
                norm = np.clip(s / denom, -1, 1)
                # red toward AI (positive for AI class), blue away
                color = cm.RdBu_r(0.5 + 0.5 * norm)
            txt = ax.text(x, yline, w + " ", fontsize=12, transform=ax.transAxes,
                          bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="none"))
            fig.canvas.draw()
            bb = txt.get_window_extent().transformed(ax.transAxes.inverted())
            x = bb.x1 + 0.004
            if x > 0.92:
                x = 0.0; yline -= 0.42
        pred = "AI" if ex["predicted_label"] == 1 else "Human"
        ax.text(0, 0.97, f"pred={pred}  p(AI)={ex['p_ai']:.2f}  FW-mass={ex['fw_mass']*100:.1f}%",
                fontsize=9, transform=ax.transAxes, color="#444")
    fig.suptitle("Faithful, content-focused explanations (function words greyed)", fontweight="bold")
    _save(fig, path)


ALL_FIGURES = [
    ("01_dataset_overview", fig_dataset_overview),
    ("02_leakage_gap", fig_leakage_gap),
    ("03_indomain_performance", fig_indomain_performance),
    ("04_confusion_matrices", fig_confusion_matrices),
    ("05_roc_pr", fig_roc_pr),
    ("06_ood_transfer", fig_ood_transfer),
    ("07_robustness", fig_robustness),
    ("08_fw_attribution_mass", fig_fw_attribution_mass),
    ("08b_fw_identity_sensitivity", fig_fw_identity_sensitivity),
    ("09_faithfulness", fig_faithfulness),
    ("10_deletion_insertion_curves", fig_deletion_insertion_curves),
    ("11_calibration", fig_calibration),
    ("12_learning_curves", fig_learning_curves),
    ("13_embedding_projection", fig_embedding_projection),
    ("14_example_explanation", fig_example_explanation),
]


def make_all_figures(results: dict, out_dir: str = "figures") -> list[str]:
    plt.rcParams.update(PUBLICATION_STYLE)
    made = []
    for name, fn in ALL_FIGURES:
        path = os.path.join(out_dir, f"{name}.png")
        try:
            fn(results, path)
            if os.path.exists(path):
                made.append(path)
        except Exception as e:
            print(f"  [warn] {name} failed: {type(e).__name__}: {str(e)[:120]}")
    return made
