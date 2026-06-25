#!/usr/bin/env python3
"""Figures for the reviewer-revision results, in the house style of faithdetect.viz.

Generates, into paper/figures/:
  slot_attribution_mass.png  -- FW/[FUNC] slot attribution (IG pad-baseline + occlusion)
  attack_split.png           -- FW attack decomposed (swap / delete / duplicate)
  ood_threshold.png          -- OOD F1 gap vs threshold-free ROC-AUC gap
All numbers come from results/*.json (no hard-coded data).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})
VORDER = ["baseline", "softreg", "hardmask"]
VLABEL = {"baseline": "Baseline", "softreg": "SoftReg", "hardmask": "Hard-Mask (FAITH)"}
VCOLOR = {"baseline": "#6c757d", "softreg": "#e08214", "hardmask": "#2c7d59"}
OUT = Path("paper/figures")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def slot_mass():
    d = json.load(open("results/placeholder_mass.json"))["variants"]
    ig = [d[v]["ig_fw_position_mass_mean"] * 100 for v in VORDER]
    occ = [d[v]["occlusion_fw_mass_mean"] * 100 for v in VORDER]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(VORDER)); w = 0.38
    for i, v in enumerate(VORDER):
        ax.bar(x[i] - w / 2, ig[i], w, color=VCOLOR[v])
        ax.bar(x[i] + w / 2, occ[i], w, color=VCOLOR[v], alpha=0.5, hatch="//", edgecolor="white")
        ax.text(x[i] - w / 2, ig[i] + 1, f"{ig[i]:.1f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(x[i] + w / 2, occ[i] + 1, f"{occ[i]:.1f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([VLABEL[v] for v in VORDER])
    ax.set_ylabel("Slot attribution mass (%)")
    ax.set_ylim(0, max(occ + ig) + 8)
    ax.set_title("Attribution on function-word SLOTS ([FUNC] for Hard-Mask):\nidentity is erased (Table 3 $=0$) but the slots are still used")
    ax.legend(handles=[Patch(facecolor="#888", label="Integrated Gradients (pad baseline)"),
                       Patch(facecolor="#888", alpha=0.5, hatch="//", label="Occlusion")],
              fontsize=9, loc="upper right")
    _save(fig, "slot_attribution_mass.png")


def attack_split():
    a = json.load(open("results/attack_split.json"))["variants"]
    conds = ["clean", "fw_swap", "fw_delete", "fw_duplicate"]
    clabels = ["Clean", "Swap\n(identity)", "Delete", "Duplicate"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(conds)); w = 0.8 / len(VORDER)
    ax.axvspan(0.5, 1.5, color="#f0e6d2", alpha=0.6, zorder=0)
    for i, v in enumerate(VORDER):
        vals = [a[v][c]["f1_macro"] * 100 for c in conds]
        ax.bar(x + (i - (len(VORDER) - 1) / 2) * w, vals, w, color=VCOLOR[v], label=VLABEL[v])
    ax.set_xticks(x); ax.set_xticklabels(clabels)
    ax.set_ylim(60, 100); ax.set_ylabel("F1 (macro)")
    ax.set_title("Function-word attack decomposed (pilot: distilroberta, seed 0)\nHard-Mask is flat under swap (covered by the proof), not delete/duplicate")
    ax.text(1.0, 61, "proof covers\nthis column", ha="center", va="bottom", fontsize=8, style="italic", color="#7a5c1e")
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, "attack_split.png")


def ood_threshold():
    f = json.load(open("results_colab/results/full_results.json"))["variants"]
    def agg(v, m):
        a = f[v]["ood"]["aggregated"][m]
        return a["mean"] * 100, (a["mean"] - a["lo"]) * 100, (a["hi"] - a["mean"]) * 100
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    groups = ["OOD F1", "OOD ROC-AUC\n($\\times100$)"]
    x = np.arange(len(groups)); w = 0.8 / len(VORDER)
    for i, v in enumerate(VORDER):
        f1 = agg(v, "f1_macro"); au = agg(v, "roc_auc")
        means = [f1[0], au[0]]
        yerr = [[f1[1], au[1]], [f1[2], au[2]]]
        ax.bar(x + (i - (len(VORDER) - 1) / 2) * w, means, w, yerr=yerr, capsize=3,
               color=VCOLOR[v], label=VLABEL[v])
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 100); ax.set_ylabel("Score")
    ax.set_title("OOD: the F1 gap is largely a threshold artefact\n(threshold-free ROC-AUC barely separates the variants)")
    ax.annotate("large F1\ngap", xy=(0.0, 47), xytext=(-0.35, 25), fontsize=8.5, color="#b00",
                ha="center", arrowprops=dict(arrowstyle="->", color="#b00"))
    ax.annotate("AUC gap\n$\\approx$ noise", xy=(1.0, 76), xytext=(1.35, 55), fontsize=8.5, color="#176",
                ha="center", arrowprops=dict(arrowstyle="->", color="#176"))
    ax.legend(fontsize=9, loc="upper right")
    _save(fig, "ood_threshold.png")


def variant_comparison():
    d = json.load(open("results/variant_comparison.json"))["variants"]
    order = ["baseline", "hardmask", "random", "deletion"]
    labels = {"baseline": "Baseline", "hardmask": "Hard-Mask\n([FUNC])",
              "random": "Random\nplaceholder", "deletion": "Deletion"}
    colors = {"baseline": "#6c757d", "hardmask": "#2c7d59", "random": "#9467bd", "deletion": "#c0504d"}
    ind = [d[v]["indomain_f1"]["mean"] for v in order]
    ood = [d[v]["ood_f1"]["mean"] for v in order]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    x = np.arange(len(order)); w = 0.38
    b1 = ax.bar(x - w / 2, ind, w, color=[colors[v] for v in order], label="In-domain F1")
    b2 = ax.bar(x + w / 2, ood, w, color=[colors[v] for v in order], alpha=0.5, hatch="//",
                edgecolor="white", label="OOD F1")
    for xi, vi in zip(x - w / 2, ind):
        ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=8.5, fontweight="bold")
    for xi, vi in zip(x + w / 2, ood):
        ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([labels[v] for v in order])
    ax.set_ylim(0, 100); ax.set_ylabel("F1 (macro)")
    ax.set_title("Removing function words by any route keeps in-domain F1\nbut costs cross-domain transfer (pilot: distilroberta, seed 0)")
    ax.legend(handles=[Patch(facecolor="#888", label="In-domain F1"),
                       Patch(facecolor="#888", alpha=0.5, hatch="//", label="OOD F1")],
              fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    _save(fig, "variant_comparison.png")


def lambda_sweep():
    import os
    if not os.path.exists("results/lambda_sweep.json"):
        print("skip lambda_sweep (no json yet)"); return
    d = json.load(open("results/lambda_sweep.json")); s = d["sweep"]; meta = d.get("meta", {})
    enc = meta.get("encoder", "?"); ns = len(meta.get("seeds", []) or [])
    scale = "full data" if meta.get("train_subsample") in (0, None) else f"n={meta['train_subsample']}"
    lam = [r["lambda"] for r in s]
    def me(key): return ([r[key]["mean"] for r in s],
                         [r[key]["mean"] - r[key]["lo"] for r in s],
                         [r[key]["hi"] - r[key]["mean"] for r in s])
    f1, f1lo, f1hi = me("indomain_f1"); se, selo, sehi = me("identity_sensitivity")
    fig, ax1 = plt.subplots(figsize=(6.8, 4.3))
    ax2 = ax1.twinx()
    l1 = ax1.errorbar(lam, f1, yerr=[f1lo, f1hi], fmt="o-", color="#2c7d59", capsize=3, label="In-domain F1")
    l2 = ax2.errorbar(lam, se, yerr=[selo, sehi], fmt="s--", color="#e08214", capsize=3,
                      label="Identity-sensitivity $|\\Delta p|$ (%)")
    ax1.set_xlabel("SoftReg penalty weight $\\lambda$")
    ax1.set_ylabel("In-domain F1 (macro)", color="#2c7d59")
    ax2.set_ylabel("Identity-sensitivity (%)", color="#e08214")
    ax1.set_title(f"SoftReg $\\lambda$ sweep ({enc}, {scale}, {ns} seeds): in-domain F1 is stable\n"
                  f"up to $\\lambda{{=}}2$ then collapses; reliance shows no clean monotonic trend")
    ax1.grid(alpha=0.25)
    ax1.legend(handles=[l1, l2], fontsize=9, loc="lower left")
    _save(fig, "lambda_sweep.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["slot", "attack", "ood", "variants", "lambda"]
    if "slot" in which: slot_mass()
    if "attack" in which: attack_split()
    if "ood" in which: ood_threshold()
    if "variants" in which: variant_comparison()
    if "lambda" in which: lambda_sweep()
