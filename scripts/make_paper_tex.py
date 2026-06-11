"""Generate a self-contained LaTeX (.tex) paper from the real results JSON + figures.

Numbers/tables come from results/*.json (no hard-coded results); all figures are embedded and
referenced. The .tex uses only standard packages (article class) so it compiles on Overleaf
out of the box. Figures are copied next to the .tex under paper/figures/.

Usage:
  python scripts/make_paper_tex.py --results results_colab/results/full_results.json \
      --figdir results_colab/figures --out paper/FAITH-Detect.tex
"""
import argparse
import os
import shutil

import _bootstrap  # noqa: F401
from faithdetect.utils.logging import load_json

VARIANTS = ["baseline", "softreg", "hardmask"]
LABEL = {"baseline": "Baseline", "softreg": "SoftReg", "hardmask": "Hard-Mask (FAITH)"}


# ----------------------------- helpers ----------------------------- #
def esc(s):
    """Escape a *dynamic* string (e.g. a model name) for LaTeX text mode."""
    s = str(s)
    for k, v in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(k, v)
    return s


def pct(x, d=1):
    return f"{x*100:.{d}f}"


def mean_ci(interval, d=1):
    """LaTeX-safe 'mean $\\pm$ half'."""
    if not interval:
        return "--"
    m = interval["mean"] * 100
    half = (interval["hi"] - interval["lo"]) / 2 * 100
    return f"{m:.{d}f} $\\pm$ {half:.{d}f}"


def agg(r, v, axis, metric):
    try:
        return r["variants"][v][axis]["aggregated"][metric]
    except Exception:
        return None


def table(L, header, rows, caption, label):
    spec = "l" + "c" * (len(header) - 1)
    L += [r"\begin{table}[H]\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}",
          r"\small", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
          " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        L.append(" & ".join(str(c) for c in row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def figure(L, name, caption, width=0.8):
    L += [r"\begin{figure}[H]\centering",
          rf"\includegraphics[width={width}\linewidth]{{figures/{name}}}",
          rf"\caption{{{caption}}}", rf"\label{{fig:{name[:-4]}}}", r"\end{figure}"]


def _load_optional(path):
    if path and os.path.exists(path):
        return load_json(path)
    return None


def _scale_note(meta_cfg: dict) -> str:
    """Honest scale statement for a pilot experiment, derived from its own config."""
    enc = esc(meta_cfg.get("encoder_name", meta_cfg.get("encoder", "?")))
    seeds = meta_cfg.get("seeds", [])
    sub = meta_cfg.get("train_subsample")
    parts = [f"{enc}", f"{len(seeds) if seeds else '?'} seeds"]
    if sub:
        parts.append(f"{sub}-review training subsample")
    return ", ".join(parts)


# ------------------------------- build ----------------------------- #
def build(r, figdir, out, extras_dir="results", extras_figdir="figures"):
    zs = _load_optional(os.path.join(extras_dir, "zeroshot.json"))
    catab = _load_optional(os.path.join(extras_dir, "category_ablation.json"))
    mixed = _load_optional(os.path.join(extras_dir, "mixedgen_results.json"))
    fewshot = _load_optional(os.path.join(extras_dir, "fewshot_hardmask.json"))
    diag = _load_optional(os.path.join(extras_dir, "ood_diagnostic_hardmask.json"))
    fwstats = _load_optional(os.path.join(extras_dir, "fw_stats.json"))
    offshelf = _load_optional(os.path.join(extras_dir, "offshelf.json"))
    replicate = _load_optional(os.path.join(extras_dir, "replicate_hc3.json"))
    newgen = _load_optional(os.path.join(extras_dir, "new_generators.json"))
    enc = esc(r.get("meta", {}).get("config", {}).get("encoder_name", "roberta-base"))
    seeds = r.get("meta", {}).get("config", {}).get("seeds", [])
    n_seeds = len(seeds) if seeds else "?"
    fw_n = r.get("meta", {}).get("fw_set_size", "")
    sd = r.get("split_describe", {})
    sig = r.get("significance", {}).get("indomain_baseline_vs_hardmask", {})
    pval = sig.get("pvalue")
    lk = r.get("leakage", {})

    L = []
    L += [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx,booktabs,amsmath,amssymb,float,caption,times}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\graphicspath{{figures/}{results_colab/figures/}}",
        r"\title{\textbf{Should a Detector Ignore Function Words? Characterising the"
        r" Trade-offs of Function-Word-Invariant Detection of AI-Generated Reviews}}",
        r"\author{Author Name\\ \small Affiliation \and Co-author\\ \small Affiliation}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
    ]

    # Abstract
    L += [r"\begin{abstract}",
        r"Supervised detectors of AI-generated text reach high in-domain accuracy but are known to "
        r"rely on superficial cues that transfer poorly. We study one such cue---function words "
        r"(articles, prepositions, conjunctions, auxiliaries and pronouns common to both human and "
        r"machine text)---and ask what is gained by making a detector ignore them. We propose "
        r"\textbf{FAITH-Detect}, which enforces function-word invariance two ways: a hard-masking "
        r"variant that replaces every function-word token with a neutral placeholder before encoding "
        r"(so the decision is provably invariant to function-word identity), and a soft attribution-"
        r"regularised variant. Explanations are produced by the deployed model and are content-only "
        r"by construction; we quantify reliance with a function-word attribution-mass metric and a "
        r"forward-only identity-sensitivity metric. On the MAiDE-up hotel-review benchmark (English "
        rf"subset; {enc}, {n_seeds} seeds, leakage-free grouped splits) the hard-masked detector "
        r"matches the unconstrained model in-domain, places essentially zero attribution on function "
        r"words, is essentially unaffected by a function-word attack, and degrades least under "
        r"synonym substitution. On a cross-domain out-of-distribution set (RAID reviews) it transfers "
        r"\emph{worse} than the unconstrained baseline, indicating that function words can carry "
        r"domain-general style signal. We therefore frame function-word invariance as a trade-off "
        r"rather than a universal improvement: a provable guarantee, in-domain parity, attack "
        r"robustness and faithful explanations, at a measured cost to cross-domain transfer. All "
        r"results are reported with confidence intervals over multiple seeds and regenerate from a "
        r"single results file.",
        r"\end{abstract}"]

    # 1 Introduction
    L += [r"\section{Introduction}",
        r"The fluency of large language models has made AI-generated reviews a practical concern for "
        r"online platforms, where fabricated opinions can distort purchasing and reputation. "
        r"Detectors are deployed in response and are increasingly expected to justify a flag with an "
        r"explanation. Two problems recur. First, supervised detectors latch onto dataset- and "
        r"generator-specific artefacts, so accuracy can collapse out of distribution "
        r"\cite{raid,sadasivan}. Second, the explanations shipped with detectors are usually post-hoc "
        r"rationalisations of an unconstrained model and are seldom validated for faithfulness "
        r"\cite{jacovi}.",
        r"",
        r"This paper studies a single, concrete cue: function words. Closed-class words such as "
        r"``the'', ``a'' and ``of'' are among the most frequent tokens in any English text and are "
        r"shared by human and machine writing; intuitively they should carry little signal about "
        r"whether a review is genuine. Yet an unconstrained detector still places a large fraction of "
        r"its attribution on them. We ask what happens if the detector is made to ignore function "
        r"words entirely, and whether doing so changes robustness, transfer and explanation quality.",
        r"",
        r"We introduce FAITH-Detect\footnote{The name reflects the design goal that explanations "
        r"be \emph{content-only and computed on the deployed model}; we do not claim improved "
        r"scores on faithfulness benchmarks (\S\ref{sec:faith}).}, which operationalises "
        r"function-word invariance in two ways. The "
        r"hard-masking variant replaces every function-word sub-token with a shared placeholder "
        r"before the encoder sees the text, so the decision cannot depend on which function word "
        r"occurred---a constructive, testable guarantee. The soft variant keeps the text but "
        r"penalises attribution mass that lands on function words during training \cite{ross}. "
        r"Explanations are computed on the deployed model, and function words are excluded by "
        r"construction; we verify reliance with a function-word attribution-mass metric and a "
        r"forward-only identity-sensitivity metric.",
        r"",
        r"\noindent\textbf{Contributions.}",
        r"\begin{itemize}",
        r"\item A function-word-invariant detector with a provable guarantee (hard masking) and a "
        r"learned alternative (soft regularisation), motivated by shortcut learning "
        r"\cite{geirhos,gururangan,mccoy}.",
        r"\item Faithful, content-only explanations computed on the deployed model, with two metrics "
        r"that make function-word reliance directly measurable.",
        r"\item A rigorous evaluation---leakage-free grouped splits, multiple seeds with confidence "
        r"intervals, significance tests, classic baselines, cross-domain and cross-generator OOD, and "
        r"two text attacks---yielding an honest characterisation of when function-word invariance "
        r"helps and when it hurts.",
        r"\end{itemize}"]

    # 2 Related work
    L += [r"\section{Related Work}",
        r"Machine-generated text detection spans zero-shot likelihood methods (GLTR \cite{gltr}; "
        r"DetectGPT \cite{detectgpt}) and supervised transformer classifiers \cite{roberta,solaiman}; "
        r"surveys catalogue the area \cite{crothers}. Robustness is a recurring weakness: the RAID "
        r"benchmark shows detectors are brittle to domain shift, unseen generators and adversarial "
        r"edits \cite{raid}, paraphrasing can evade them \cite{krishna}, and some work questions "
        r"whether reliable detection is achievable in general \cite{sadasivan}. Our aim is "
        r"orthogonal---rather than proposing a stronger detector, we ask what one interpretable "
        r"family of features (function words) contributes, and whether removing it improves "
        r"robustness and explanation quality. Shortcut learning frames the broader phenomenon of "
        r"models exploiting spurious correlations \cite{geirhos}, with NLP-specific evidence of "
        r"annotation artefacts and ``right-for-the-wrong-reasons'' behaviour \cite{gururangan,mccoy}. "
        r"On the explanation side, gradient and perturbation attributions include Integrated "
        r"Gradients \cite{ig}, LIME \cite{lime} and SHAP \cite{shap}; faithfulness is formalised by "
        r"ERASER comprehensiveness/sufficiency \cite{eraser}, deletion/insertion curves \cite{rise}, "
        r"and the broader discussion of what faithful interpretation requires \cite{jacovi}. "
        r"Attribution regularisation (``right for the right reasons'') trains models to place "
        r"gradient mass where a prior says it should \cite{ross}; our soft variant is an instance "
        r"with function words as the prior. The MAiDE-up dataset \cite{maideup} provides human and "
        r"GPT-4 hotel reviews and is our in-domain benchmark."]

    # 3 Method
    L += [r"\section{Method}",
        r"\subsection{Function-word set}",
        rf"We define an auditable English function-word set ({fw_n} surface forms) as the union of a "
        r"curated closed-class list (articles, prepositions, conjunctions, auxiliaries, pronouns, "
        r"particles, determiners) and the standard NLTK, scikit-learn and spaCy \cite{spacy} stop-word "
        r"lists. The set is fixed and surface-form based, so identification is deterministic at "
        r"inference: a word is a function word if its lower-cased, punctuation-stripped form is in the "
        r"set.",
        r"\subsection{Hard-masking (provable invariance)}",
        r"We tokenise with the encoder's byte-level BPE and map each sub-token back to its surface "
        r"word via offset alignment. Every sub-token belonging to a function word is replaced by a "
        r"single added placeholder token \texttt{[FUNC]} before encoding. Because all function words "
        r"map to the same placeholder, the encoder---and therefore the classifier---cannot "
        r"distinguish which function word occurred: the decision is invariant to function-word "
        r"identity by construction. An automated test asserts bit-identical logits when function "
        r"words are permuted.",
        r"\subsection{Soft attribution regularisation}",
        r"The soft variant keeps the full text and adds a penalty equal to the fraction of input-"
        r"gradient saliency that lands on function-word tokens (gradient-times-input, with "
        r"\texttt{create\_graph} so the penalty back-propagates into the weights) \cite{ross}. To "
        r"bound memory at large batch and sequence length, the second-order penalty is computed on a "
        r"small sub-batch---an unbiased stochastic estimate---while the cross-entropy uses the full "
        r"batch.",
        r"\subsection{Faithful, content-only explanations}",
        r"All attributions are computed on the deployed model (including the hard-masking transform), "
        r"never a surrogate. We use Integrated Gradients on the embedding layer \cite{ig} and "
        r"leave-one-word-out occlusion, aggregated to word level; function words are excluded from "
        r"the displayed explanation. We report two reliance metrics: function-word attribution mass "
        r"(the share of total absolute attribution on function words) and identity-sensitivity (the "
        r"mean change in AI probability when each function word is replaced by a different function "
        r"word). For explanation quality we report ERASER comprehensiveness and sufficiency "
        r"\cite{eraser} and deletion/insertion AUC over content words \cite{rise}."]

    # 4 Setup
    split_line = ""
    if sd:
        split_line = (rf" Splits are grouped by hotel (train {sd['train']['n']}, val "
                      rf"{sd['val']['n']}, test {sd['test']['n']}; train--test hotel overlap "
                      rf"{sd['hotel_leakage_train_test']}).")
    L += [r"\section{Experimental Setup}",
        r"In-domain data is the English subset of MAiDE-up (human vs GPT-4 hotel reviews) "
        r"\cite{maideup}. Because every hotel appears in both classes, a random split leaks "
        r"hotel-specific content across train and test; we therefore use grouped (by-hotel) splits "
        r"and additionally report a random split to quantify the leakage (Figure~\ref{fig:01_dataset_overview})."
        + split_line +
        rf" We fine-tune {enc} \cite{{roberta}} with AdamW \cite{{adamw}} for each of {n_seeds} seeds "
        r"and report mean $\pm$ 95\% confidence interval. Out-of-distribution evaluation uses the "
        r"RAID benchmark's reviews domain \cite{raid} (cross-domain, spanning multiple generators "
        r"including GPT-3.5/4, Cohere and Llama). Robustness is probed with a function-word attack "
        r"(deleting/duplicating/swapping function words) and a WordNet \cite{wordnet} synonym attack "
        r"on content words. Baselines are TF-IDF + logistic regression and a content-only TF-IDF "
        r"variant. Significance between models on the shared test set uses McNemar's test; intervals "
        r"use Student-$t$ and the bootstrap."]
    figure(L, "01_dataset_overview.png",
           "MAiDE-up English splits (grouped, zero train--test hotel leakage).", 0.6)

    # 5 Results
    L += [r"\section{Results}"]

    # 5.1 in-domain
    L += [r"\subsection{In-domain performance}"]
    rows = []
    for v in VARIANTS:
        rows.append([LABEL[v], mean_ci(agg(r, v, "indomain", "f1_macro")),
                     mean_ci(agg(r, v, "indomain", "roc_auc")),
                     mean_ci(agg(r, v, "indomain", "accuracy"))])
    for bname in ("tfidf_lr", "tfidf_lr_content"):
        b = r.get("baselines", {}).get(bname)
        if b:
            m = b["metrics"]
            rows.append([esc(bname), pct(m["f1_macro"]), pct(m.get("roc_auc", 0)), pct(m.get("accuracy", 0))])
    table(L, ["Model", "F1 (macro)", "ROC-AUC", "Accuracy"], rows,
          "In-domain performance (mean $\\pm$ 95\\% CI over seeds; \\% units).", "indomain")
    pstr = f" (McNemar baseline vs.\\ Hard-Mask, $p={pval:.2f}$)" if pval is not None else ""
    L += [r"On in-distribution data the three neural variants are statistically indistinguishable" +
          pstr + r"; Hard-Mask attains the same F1 as the unconstrained baseline, with the tightest "
          r"interval. Function-word invariance therefore carries no measurable in-domain cost on this "
          r"benchmark (Figure~\ref{fig:03_indomain_performance}). Confusion matrices "
          r"(Figure~\ref{fig:04_confusion_matrices}) and ROC/PR curves "
          r"(Figure~\ref{fig:05_roc_pr}) show the same ordering, with very high ROC-AUC for all "
          r"neural variants."]
    figure(L, "03_indomain_performance.png", "In-domain F1 (mean $\\pm$ 95\\% CI) with classic baselines.")
    figure(L, "04_confusion_matrices.png", "In-domain confusion matrices (reference seed) by variant.", 0.95)
    figure(L, "05_roc_pr.png", "In-domain ROC and precision--recall curves (reference seed).", 0.95)

    # 5.1b training-free baselines (optional: results/zeroshot.json)
    if zs:
        L += [r"\subsection{Training-free (zero-shot) baselines}\label{sec:zeroshot}"]
        axes = [a for a in ("indomain", "fw_attack", "ood") if a in zs]
        axis_label = {"indomain": "In-domain F1 / AUROC", "fw_attack": "FW-attack F1",
                      "ood": "OOD F1 / AUROC"}
        head = ["Method"] + [axis_label[a] for a in axes]
        rows = []
        for m in zs.get("meta", {}).get("methods", []):
            row = [esc(m.replace("_", "-"))]
            for a in axes:
                d = zs[a].get(m, {})
                f1 = d.get("metrics", {}).get("f1_macro")
                au = d.get("auroc")
                if a == "fw_attack":
                    row.append(pct(f1) if f1 is not None else "--")
                else:
                    row.append(f"{pct(f1)} / {au:.2f}" if f1 is not None else "--")
            rows.append(row)
        table(L, head, rows,
              "Training-free detectors (GPT-2-scored; thresholds fit on the training split "
              "only; \\% units for F1).", "zeroshot")
        L += [r"For context we evaluate four training-free statistical detectors---mean "
              r"log-likelihood, log-rank \cite{gltr}, predictive entropy, and the analytic "
              r"Fast-DetectGPT discrepancy \cite{fastdetectgpt}---scored with GPT-2 and "
              r"thresholded on the training split only.\footnote{Our entropy score is oriented "
              r"so that \emph{lower} entropy indicates machine text, which is the empirically "
              r"correct direction on this dataset and the opposite of the convention in some "
              r"prior work; AUROC is unaffected by orientation up to reflection.} They trail the "
              r"supervised models in-domain, as expected on a single-domain benchmark, and "
              r"provide a reference point that requires no labelled data. We report them to "
              r"contextualise the supervised results rather than as competitors."]

    if zs and "fw_attack" in zs and "indomain" in zs:
        try:
            drops = []
            for m, d in zs["indomain"].items():
                a = zs["fw_attack"].get(m, {})
                if d.get("metrics") and a.get("metrics"):
                    drops.append(d["metrics"]["f1_macro"] - a["metrics"]["f1_macro"])
            if drops:
                avg_drop = 100 * sum(drops) / len(drops)
                L += [rf"Under the function-word attack the training-free detectors shift by "
                      rf"{avg_drop:+.1f} F1 points on average, confirming that likelihood-based "
                      r"statistics also respond to function-word perturbations; only the "
                      r"hard-masked model is invariant to them by construction."]
        except Exception:
            pass

    # 5.2 reliance
    L += [r"\subsection{Function-word reliance}"]
    rows = []
    for v in VARIANTS:
        fm = r.get("fw_mass", {}).get(v, {})
        ids = r.get("fw_identity_sensitivity", {}).get(v, {})
        rows.append([LABEL[v], (pct(fm["mean"]) + r"\%") if fm else "--",
                     (pct(ids["mean"], 2) + r"\%") if ids else "--"])
    table(L, ["Variant", "FW attribution mass (IG)", "Identity-sensitivity $|\\Delta p|$"], rows,
          "Function-word reliance: lower is less reliance.", "reliance")
    L += [r"The unconstrained baseline places a substantial share of its attribution on function "
          r"words; soft regularisation reduces it; hard masking eliminates it "
          r"(Figure~\ref{fig:08_fw_attribution_mass}). The forward-only identity-sensitivity "
          r"metric---the change in AI probability when function words are swapped---agrees: it is "
          r"near zero for Hard-Mask and clearly positive for the baseline "
          r"(Figure~\ref{fig:08b_fw_identity_sensitivity}). The two metrics are consistent because "
          r"they measure the same property (reliance on function-word identity) from gradient and "
          r"perturbation perspectives."]
    figure(L, "08_fw_attribution_mass.png", "Integrated-Gradients attribution mass on function words.", 0.72)
    figure(L, "08b_fw_identity_sensitivity.png", "Change in AI probability when function words are swapped.", 0.72)

    # 5.3 robustness
    L += [r"\subsection{Robustness to text attacks}"]
    attacks = list(r["variants"][VARIANTS[0]].get("attacks", {}).keys())
    header = ["Variant", "Clean"] + [esc(a.replace("_", " ")) for a in attacks]
    rows = []
    for v in VARIANTS:
        clean = agg(r, v, "indomain", "f1_macro")
        row = [LABEL[v], pct(clean["mean"]) if clean else "--"]
        for a in attacks:
            it = r["variants"][v]["attacks"][a]["aggregated"].get("f1_macro")
            row.append(pct(it["mean"]) if it else "--")
        rows.append(row)
    table(L, header, rows, "F1 (macro) under text attacks (mean over seeds; \\% units).", "robust")
    L += [r"Under the function-word attack, the baseline and soft variants lose several points "
          r"whereas Hard-Mask is essentially unaffected---perturbing function words cannot move a "
          r"decision that ignores them. Under synonym substitution all variants degrade, but "
          r"Hard-Mask degrades least and retains the highest F1 (Figure~\ref{fig:07_robustness}). "
          r"These are the clearest empirical benefits of invariance; we do not claim improvements "
          r"beyond the two attacks tested."]
    figure(L, "07_robustness.png", "F1 under text attacks; Hard-Mask is flat under the function-word attack.")

    # 5.4 transfer
    L += [r"\subsection{Cross-domain and cross-generator transfer}"]
    rows = []
    for v in VARIANTS:
        rows.append([LABEL[v], mean_ci(agg(r, v, "indomain", "f1_macro")),
                     mean_ci(agg(r, v, "ood", "f1_macro"))])
    table(L, ["Variant", "In-domain F1", "OOD F1 (RAID reviews)"], rows,
          "In-domain vs.\\ cross-domain OOD (mean $\\pm$ 95\\% CI; \\% units).", "ood")
    L += [r"On the cross-domain OOD set (hotel $\rightarrow$ movie reviews) the picture reverses: "
          r"Hard-Mask transfers \emph{worse} than the baseline, with non-overlapping confidence "
          r"intervals (Figure~\ref{fig:06_ood_transfer}). When the content vocabulary shifts, the "
          r"content words Hard-Mask depends on largely disappear, while the baseline's function-word "
          r"and stylistic cues evidently carry domain-general signal. Function words are thus not "
          r"pure noise; in cross-domain transfer they can be useful. The per-generator breakdown "
          r"(Figure~\ref{fig:06b_cross_generator}) shows the same ordering across most generators. "
          r"We emphasise a confound: this OOD shifts both domain and generator at once and is an "
          r"extreme content change, so it does not isolate cross-generator transfer; a same-domain, "
          r"different-generator test would be needed for that claim."]
    figure(L, "06_ood_transfer.png", "In-domain vs.\\ cross-domain OOD F1 by variant (mean $\\pm$ 95\\% CI).")
    figure(L, "06b_cross_generator.png", "Per-generator detection F1 on RAID reviews.", 0.95)

    # 5.4b mechanism: attribution carry-over (optional: results/ood_diagnostic_hardmask.json)
    if diag:
        L += [r"\paragraph{Why the transfer fails: attribution carry-over.}"]
        try:
            ind_acc = diag["indomain"]["metrics"]["accuracy"]
            ood_acc = diag["ood"]["metrics"]["accuracy"]
            carry_in = diag["vocab_carryover"]["indomain_mass_fraction"]
            carry_ood = diag["vocab_carryover"]["ood_mass_fraction"]
            L += [rf"Attributing the hard-masked model's decisions directly exposes the mechanism: "
                  rf"in-domain, {100*carry_in:.0f}\% of its attribution mass falls on its 50 "
                  rf"most-attributed in-domain content words, but on out-of-domain text only "
                  rf"{100*carry_ood:.0f}\% of the attribution touches that vocabulary---the "
                  rf"content words the model learned to use are simply absent "
                  rf"(accuracy {100*ind_acc:.0f}\% in-domain vs.\ {100*ood_acc:.0f}\% OOD on the "
                  r"diagnostic sample; Figure~\ref{fig:15_ood_diagnostic_hardmask}). This "
                  r"supports a data explanation (domain-specific content vocabulary) over a "
                  r"purely structural one, motivating the mixed-generator experiment below."]
        except Exception:
            L += [r"Figure~\ref{fig:15_ood_diagnostic_hardmask} compares the content words that "
                  r"receive attribution in-domain and out-of-domain."]
        figure(L, "15_ood_diagnostic_hardmask.png",
               "Top-attributed content words for the Hard-Mask model, in-domain vs.\\ OOD.", 0.95)

    # 5.4c mixed-generator training (optional: results/mixedgen_results.json)
    if mixed:
        mcfg2 = mixed.get("meta", {}).get("config", {})
        L += [r"\subsection{Mixed-generator training and held-out generators}\label{sec:mixedgen}",
              rf"To separate generator shift from domain shift, we add same-domain reviews from "
              rf"held-in generators ({esc(', '.join(mcfg2.get('train_mix_generators', [])))}; plus "
              r"half of the human reviews) to the training set and evaluate on a disjoint frame "
              rf"of held-out generators ({esc(', '.join(mcfg2.get('heldout_generators', [])))}; "
              r"plus the remaining humans). Held-out-generator evaluation is then same-domain, "
              rf"different-generator. (Pilot scale: {_scale_note(mcfg2)}.)"]
        rows = []
        for v in VARIANTS:
            try:
                ind = mixed["variants"][v]["indomain"]["aggregated"].get("f1_macro")
                ho = mixed["variants"][v]["heldout_gen"]["aggregated"].get("f1_macro")
                rows.append([LABEL[v], mean_ci(ind), mean_ci(ho)])
            except Exception:
                continue
        if rows:
            table(L, ["Variant", "In-domain F1", "Held-out-generator F1 (same domain)"], rows,
                  "Mixed-generator training: same-domain transfer to unseen generators "
                  "(mean $\\pm$ 95\\% CI; \\% units).", "mixedgen")
        try:
            b = mixed["variants"]["baseline"]["heldout_gen"]["aggregated"]["f1_macro"]["mean"]
            h = mixed["variants"]["hardmask"]["heldout_gen"]["aggregated"]["f1_macro"]["mean"]
            gap = 100 * (b - h)
            if gap > 3:
                concl = (rf"the baseline retains a {gap:.0f}-point advantage on held-out "
                         r"generators, so part of the transfer gap is attributable to the "
                         r"invariance constraint itself, not only to domain shift")
            elif gap < -3:
                concl = (rf"Hard-Mask exceeds the baseline by {-gap:.0f} points on held-out "
                         r"generators, indicating the earlier OOD gap was driven by domain "
                         r"shift rather than the invariance constraint")
            else:
                concl = (r"the two models are within a few points of each other on held-out "
                         r"generators, indicating the earlier cross-domain gap was driven "
                         r"largely by domain shift rather than by the invariance constraint")
            L += [rf"With generator-diverse, same-domain training data, {concl} "
                  r"(Figure~\ref{fig:06c_heldout_generator})."]
        except Exception:
            pass
        figure(L, "06c_heldout_generator.png",
               "Held-out generators after mixed-generator training (same domain).", 0.95)

    # 5.4d per-category ablation (optional: results/category_ablation.json)
    if catab:
        cats = catab.get("categories", {})
        acfg = catab.get("meta", {}).get("args", {})
        L += [r"\subsection{Which function words carry the signal? A per-category ablation}"
              r"\label{sec:catablation}",
              rf"We retrain the hard-masked detector masking ONE grammatical category at a time "
              rf"(determiners, pronouns, prepositions, conjunctions, auxiliaries, adverbial "
              rf"function words), with the full union mask as reference. (Pilot scale: "
              rf"{esc(acfg.get('encoder','?'))}, {len(acfg.get('seeds',[]))} seeds, "
              rf"{acfg.get('train_subsample','?')}-review training subsample.)"]
        rows = []
        for c, d in cats.items():
            f1 = d.get("indomain", {}).get("f1_macro")
            oo = d.get("ood", {}).get("f1_macro")
            rows.append([esc("mask " + c), mean_ci(f1), mean_ci(oo) if oo else "--"])
        table(L, ["Masked category", "In-domain F1", "OOD F1"], rows,
              "Masking one function-word category at a time (Hard-Mask variant).", "catab")
        try:
            scored = {c: d["ood"]["f1_macro"]["mean"] for c, d in cats.items()
                      if c != "union" and d.get("ood", {}).get("f1_macro")}
            if scored:
                worst = min(scored, key=scored.get)
                best = max(scored, key=scored.get)
                L += [rf"At this scale, masking \emph{{{esc(worst)}}} produces the lowest OOD "
                      rf"transfer ({100*scored[worst]:.0f}\%) and masking \emph{{{esc(best)}}} "
                      rf"the highest ({100*scored[best]:.0f}\%) "
                      r"(Figure~\ref{fig:16_category_ablation}). We treat this ordering as "
                      r"suggestive rather than conclusive given the pilot scale, and release "
                      r"the protocol for replication at full scale."]
        except Exception:
            pass
        figure(L, "16_category_ablation.png",
               "Per-category masking ablation: in-domain vs.\\ OOD F1.", 0.95)

    # 5.4e few-shot domain recovery (optional: results/fewshot_hardmask.json)
    if fewshot:
        shots_d = fewshot.get("results", {})
        if shots_d:
            L += [r"\subsection{Few-shot domain recovery}\label{sec:fewshot}",
                  r"Finally, we ask how much labelled target-domain data is needed to recover "
                  r"the cross-domain loss: the hotel-trained Hard-Mask model is fine-tuned on "
                  r"$k$ labelled out-of-domain reviews and evaluated on a fixed held-out OOD "
                  r"test set."]
            rows = []
            for k in sorted(shots_d, key=lambda x: int(x)):
                agg_k = shots_d[k].get("aggregated", {}).get("f1_macro")
                rows.append([str(k), mean_ci(agg_k)])
            table(L, ["Labelled OOD examples $k$", "OOD F1"], rows,
                  "Few-shot domain adaptation of the Hard-Mask model (mean $\\pm$ 95\\% CI over "
                  "few-shot samples).", "fewshot")
            try:
                ks = sorted(shots_d, key=lambda x: int(x))
                f0 = shots_d[ks[0]]["aggregated"]["f1_macro"]["mean"]
                f1_ = shots_d[ks[-1]]["aggregated"]["f1_macro"]["mean"]
                L += [rf"Moving from $k={ks[0]}$ to $k={ks[-1]}$ changes OOD F1 from "
                      rf"{100*f0:.0f}\% to {100*f1_:.0f}\%; the size of this recovery indicates "
                      r"how much of the cross-domain gap is a labelled-data requirement rather "
                      r"than a structural limitation."]
            except Exception:
                pass

    # 5.4f which function words carry the signal (optional: results/fw_stats.json)
    if fwstats:
        agree = fwstats.get("agreement", {}).get("significant", {})
        rate = agree.get("sign_agreement_rate")
        words = sorted(fwstats.get("words", []), key=lambda w: -abs(w.get("z_in", 0)))[:8]
        L += [r"\subsection{Which function words carry the signal? Word-level statistics}",
              r"We compute the Monroe et al.\ ``Fightin' Words'' log-odds ratio with an "
              r"informative Dirichlet prior for every function word, contrasting AI and human "
              r"text in-domain, and repeat the statistic on the out-of-domain reviews pool to "
              r"test the \emph{stability} of each signal."]
        rows = [[esc(w["word"]), f"{w['z_in']:+.1f}", f"{w['z_ood']:+.1f}",
                 "yes" if (w['z_in'] * w['z_ood'] > 0) else r"\textbf{flips}"] for w in words]
        table(L, ["Function word", "$z$ (in-domain)", "$z$ (OOD)", "Sign stable?"], rows,
              "Strongest function-word signals and their cross-domain stability.", "fwstats")
        if rate is not None:
            L += [rf"Only {100*rate:.0f}\% of significant ($|z|>2$) function-word signals keep "
                  r"their sign across domains; the canonical article ``the'' is the strongest "
                  r"AI-leaning signal in-domain yet reverses out-of-domain, while pronouns "
                  r"(humans write ``I''/``we'' in every domain) form the stable family "
                  r"(Figure~\ref{fig:17_fw_words}). This is the word-level statistical basis of "
                  r"the shortcut claim: a large share of the function-word signal an "
                  r"unconstrained detector absorbs is domain-local.",]
        figure(L, "17_fw_words.png",
               "Function-word log-odds: top signals (left) and in-domain vs.\\ OOD stability (right).", 0.98)

    # 5.4g replication on a second dataset (optional: results/replicate_hc3.json)
    if replicate:
        rows = []
        for v in ("baseline", "hardmask"):
            vagg = replicate.get("variants", {}).get(v, {}).get("aggregated", {})
            if vagg:
                rows.append([{"baseline": "Baseline", "hardmask": "Hard-Mask"}[v],
                             mean_ci(vagg.get("indomain", {}).get("f1_macro")),
                             mean_ci(vagg.get("fw_attack", {}).get("f1_macro")),
                             mean_ci(vagg.get("maide_transfer", {}).get("f1_macro"))])
        if rows:
            L += [r"\subsection{Replication on a second corpus (HC3)}",
                  r"We retrain baseline and Hard-Mask from scratch on HC3 (human vs.\ ChatGPT "
                  r"answers) and evaluate in-corpus, under the function-word attack, and "
                  r"zero-shot transferred to MAiDE-up."]
            table(L, ["Variant", "HC3 in-domain F1", "FW-attack F1", r"$\rightarrow$MAiDE F1"],
                  rows, "Replication on HC3 (mean $\\pm$ 95\\% CI over seeds; \\% units).", "hc3")
            L += [r"HC3 is near-ceiling for both variants (ChatGPT-era answers are stylistically "
                  r"blatant), so in-domain parity replicates trivially. The informative result is "
                  r"transfer: here Hard-Mask transfers \emph{better} than the baseline --- the "
                  r"opposite direction from the hotel$\rightarrow$movie result --- indicating the "
                  r"sign of the invariance--transfer effect is corpus-dependent rather than a "
                  r"universal penalty. We report both directions and claim only the trade-off's "
                  r"existence, not its sign.",]

    # 5.4h modern open-weight generators (optional: results/new_generators.json)
    if newgen:
        L += [r"\subsection{Modern open-weight generators}",
              r"We generate fresh same-domain hotel reviews with locally run open-weight LLMs "
              r"(Llama-3.2-3B, Gemma-2-2B; prompted with real hotel names and cities) and "
              r"evaluate the trained detectors per generator "
              r"(Figure~\ref{fig:18_new_generators}). Closed API models (GPT-4o, Claude, "
              r"Gemini) are out of scope locally and remain future work.",]
        figure(L, "18_new_generators.png",
               "Detection of freshly generated Llama-3.2 / Gemma-2 hotel reviews by variant.", 0.9)

    # 5.1c off-the-shelf detectors (optional: results/offshelf.json)
    if offshelf:
        rows = []
        for name, frames in offshelf.get("detectors", {}).items():
            short = esc(name.split("/")[-1])
            cells = [short]
            for fr in ("test", "fw_attack", "synonym_attack", "raid_reviews"):
                mtr = frames.get(fr, {}).get("metrics", {})
                cells.append(pct(mtr["f1_macro"]) if "f1_macro" in mtr else "--")
            rows.append(cells)
        L += [r"\subsection{Off-the-shelf detectors}",
              r"For context we evaluate two public pre-trained detectors zero-shot (no "
              r"fine-tuning; they were trained on other distributions, so this measures "
              r"transfer \emph{into} our setting, not their best case)."]
        table(L, ["Detector", "MAiDE test F1", "FW-attack", "Synonym", "RAID reviews"],
              rows, "Public pre-trained detectors applied zero-shot (\\% units).", "offshelf")
        L += [r"Both transfer poorly into GPT-4 hotel reviews and also degrade under the "
              r"function-word attack, extending the attack's relevance to a third detector "
              r"family; on RAID movie reviews (closer to their training data) they recover. "
              r"This contextualises the in-domain numbers: detector quality is "
              r"distribution-bound, which is the premise of this paper's transfer analysis.",]

    # 5.5 faithfulness
    L += [r"\subsection{Explanation faithfulness}\label{sec:faith}"]
    rows = []
    for v in VARIANTS:
        f = r["variants"].get(v, {}).get("faithfulness", {}).get("summary")
        if f:
            rows.append([LABEL[v], f"{f.get('comprehensiveness',0):.3f}", f"{f.get('sufficiency',0):.3f}",
                         f"{f.get('deletion_auc',0):.3f}", f"{f.get('insertion_auc',0):.3f}"])
    if rows:
        table(L, ["Variant", "Compr.\\,$\\uparrow$", "Suff.\\,$\\downarrow$",
                  "Del-AUC\\,$\\downarrow$", "Ins-AUC\\,$\\uparrow$"], rows,
              "Explanation faithfulness (Integrated Gradients, content words).", "faith")
    L += [r"Faithfulness is comparable across variants and mixed in direction "
          r"(Figure~\ref{fig:09_faithfulness}): no variant dominates on all four measures, with "
          r"Hard-Mask strongest on sufficiency and insertion and the baseline stronger on "
          r"comprehensiveness. Deletion/insertion curves (Figure~\ref{fig:10_deletion_insertion_curves}) "
          r"tell the same story. We therefore do \emph{not} claim a faithfulness improvement from "
          r"invariance; the explanation contribution is structural---explanations are content-only by "
          r"construction and computed on the deployed model (Figure~\ref{fig:14_example_explanation})"
          r"---rather than a higher faithfulness score."]
    figure(L, "09_faithfulness.png", "ERASER comprehensiveness/sufficiency and deletion/insertion AUC.")
    figure(L, "10_deletion_insertion_curves.png", "Mean deletion and insertion curves over content words.", 0.95)
    figure(L, "14_example_explanation.png", "Example explanations: content highlighted, function words greyed.", 0.95)

    # 5.6 calibration / representation
    L += [r"\subsection{Calibration and representation}",
          r"Reliability diagrams (Figure~\ref{fig:11_calibration}) summarise calibration on the "
          r"in-domain test set via the expected calibration error \cite{guo}; we report it for "
          r"completeness rather than as a contribution. A two-dimensional projection of the encoder "
          r"representations (Figure~\ref{fig:13_embedding_projection}) shows that human and AI "
          r"reviews are well separated in-domain for all variants."]
    figure(L, "11_calibration.png", "Reliability diagrams (in-domain) with expected calibration error.", 0.6)
    figure(L, "13_embedding_projection.png", "2-D projection of encoder representations of the test set.", 0.95)

    # 5.7 leakage
    L += [r"\subsection{Leakage}"]
    if lk:
        L += [rf"A naive random split inflates baseline F1 to {pct(lk['random']['f1_macro'])}\% "
              rf"versus {pct(lk['grouped']['f1_macro'])}\% under the grouped, leakage-free split "
              r"(Figure~\ref{fig:02_leakage_gap}). Because hotels recur with both labels, a random "
              r"split lets the model exploit hotel-specific content; we report grouped numbers "
              r"throughout and recommend grouped splits for this dataset."]
    figure(L, "02_leakage_gap.png", "Random vs.\\ grouped split (baseline): random splitting overstates scores.", 0.6)

    # 6 Discussion / 7 Limitations / 8 Conclusion
    L += [r"\section{Discussion}",
        r"Making a detector ignore function words is free in-distribution on this benchmark, yields a "
        r"provable invariance and the best robustness to the two attacks we test, and produces "
        r"explanations that are content-only by construction---in contrast to post-hoc, surrogate-"
        r"based explanations. The cost is cross-domain transfer: when the content vocabulary changes, "
        r"function-word and stylistic regularities that the constrained model discards carry "
        r"domain-general signal. Practically, hard masking is attractive where robustness, "
        r"auditability and a guarantee matter and the domain is fixed; the unconstrained or soft "
        r"model is preferable for open-domain transfer. We deliberately avoid stronger claims: "
        r"results are on one in-domain dataset and two attacks, and the OOD evidence is a single "
        r"benchmark.",
        r"\section{Limitations}",
        r"The study is English-only and uses a single in-domain generator (GPT-4), so cross-generator "
        r"evidence comes from a benchmark that also shifts domain and cannot isolate generator "
        r"effects. The in-domain set is modest (2{,}000 reviews), which widens confidence intervals; "
        r"we report them honestly rather than selecting favourable seeds. The two attacks are simple "
        r"proxies for adversarial and paraphrase pressure, not an exhaustive robustness audit. "
        r"Faithfulness metrics are themselves contested \cite{jacovi} and we do not treat any single "
        r"one as decisive. Function-word invariance is a design choice with a measured trade-off, not "
        r"a universally preferable configuration.",
        r"\section{Conclusion}",
        r"Function words are a measurable shortcut in AI-text detection. A detector that provably "
        r"ignores them keeps in-domain accuracy, gains robustness to the attacks we test, and "
        r"explains itself with content alone---at a measured cost to cross-domain transfer. We "
        r"release code, leakage-free protocols, multi-seed results with confidence intervals, and "
        r"figures that regenerate from a single results file.",
        r"\paragraph{Reproducibility.} All numbers come from a single results JSON; figures and this "
        r"paper regenerate from it; the hard-masking guarantee is asserted by an automated test. "
        r"Code: \url{https://github.com/scar09-22/FAITH-Detect}.",
        r"\appendix",
        r"\section{Training behaviour}",
        r"Figure~\ref{fig:12_learning_curves} shows training loss and validation F1 by epoch, "
        r"averaged over seeds. The soft and hard variants start slower (they discard or down-weight "
        r"function-word information) but reach comparable validation F1."]
    figure(L, "12_learning_curves.png", "Training loss and validation F1 by epoch (mean over seeds).", 0.95)

    # References
    refs = [
        ("maideup", "O. Ignat, X. Xu, R. Mihalcea. MAiDE-up: Multilingual Deception Detection of "
         "AI-Generated Hotel Reviews. Findings of NAACL, 2025."),
        ("raid", "L. Dugan, A. Hwang, F. Trhlik, et al. RAID: A Shared Benchmark for Robust "
         "Evaluation of Machine-Generated Text Detectors. ACL, 2024."),
        ("ig", "M. Sundararajan, A. Taly, Q. Yan. Axiomatic Attribution for Deep Networks. ICML, 2017."),
        ("eraser", "J. DeYoung, S. Jain, N. F. Rajani, et al. ERASER: A Benchmark to Evaluate "
         "Rationalized NLP Models. ACL, 2020."),
        ("ross", "A. S. Ross, M. C. Hughes, F. Doshi-Velez. Right for the Right Reasons: Training "
         "Differentiable Models by Constraining their Explanations. IJCAI, 2017."),
        ("detectgpt", "E. Mitchell, Y. Lee, A. Khazatsky, C. D. Manning, C. Finn. DetectGPT: "
         "Zero-Shot Machine-Generated Text Detection using Probability Curvature. ICML, 2023."),
        ("fastdetectgpt", "G. Bao, Y. Zhao, Z. Teng, L. Yang, Y. Zhang. Fast-DetectGPT: "
         "Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability "
         "Curvature. ICLR, 2024."),
        ("gltr", "S. Gehrmann, H. Strobelt, A. M. Rush. GLTR: Statistical Detection and "
         "Visualization of Generated Text. ACL (System Demonstrations), 2019."),
        ("rise", "V. Petsiuk, A. Das, K. Saenko. RISE: Randomized Input Sampling for Explanation of "
         "Black-box Models. BMVC, 2018."),
        ("geirhos", "R. Geirhos, J.-H. Jacobsen, C. Michaelis, et al. Shortcut Learning in Deep "
         "Neural Networks. Nature Machine Intelligence, 2020."),
        ("roberta", "Y. Liu, M. Ott, N. Goyal, et al. RoBERTa: A Robustly Optimized BERT Pretraining "
         "Approach. arXiv:1907.11692, 2019."),
        ("lime", "M. T. Ribeiro, S. Singh, C. Guestrin. ``Why Should I Trust You?'': Explaining the "
         "Predictions of Any Classifier. KDD, 2016."),
        ("shap", "S. M. Lundberg, S.-I. Lee. A Unified Approach to Interpreting Model Predictions. "
         "NeurIPS, 2017."),
        ("jacovi", "A. Jacovi, Y. Goldberg. Towards Faithfully Interpretable NLP Systems: How Should "
         "We Define and Evaluate Faithfulness? ACL, 2020."),
        ("sadasivan", "V. S. Sadasivan, A. Kumar, S. Balasubramanian, W. Wang, S. Feizi. Can "
         "AI-Generated Text be Reliably Detected? arXiv:2303.11156, 2023."),
        ("krishna", "K. Krishna, Y. Song, M. Karpinska, J. Wieting, M. Iyyer. Paraphrasing Evades "
         "Detectors of AI-Generated Text, but Retrieval is an Effective Defense. NeurIPS, 2023."),
        ("solaiman", "I. Solaiman, M. Brundage, J. Clark, et al. Release Strategies and the Social "
         "Impacts of Language Models. arXiv:1908.09203, 2019."),
        ("crothers", "E. Crothers, N. Japkowicz, H. L. Viktor. Machine-Generated Text: A "
         "Comprehensive Survey of Threat Models and Detection Methods. IEEE Access, 2023."),
        ("guo", "C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural "
         "Networks. ICML, 2017."),
        ("gururangan", "S. Gururangan, S. Swayamdipta, O. Levy, et al. Annotation Artifacts in "
         "Natural Language Inference Data. NAACL, 2018."),
        ("mccoy", "T. McCoy, E. Pavlick, T. Linzen. Right for the Wrong Reasons: Diagnosing Syntactic "
         "Heuristics in Natural Language Inference. ACL, 2019."),
        ("wordnet", "G. A. Miller. WordNet: A Lexical Database for English. Communications of the "
         "ACM, 1995."),
        ("adamw", "I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. ICLR, 2019."),
        ("spacy", "M. Honnibal, I. Montani. spaCy: Industrial-Strength Natural Language Processing, "
         "2017."),
    ]
    L += [r"\begin{thebibliography}{99}\small"]
    for key, text in refs:
        L.append(rf"\bibitem{{{key}}} {text}")
    L += [r"\end{thebibliography}", r"\end{document}"]

    # write + co-locate figures
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig_out = os.path.join(os.path.dirname(out), "figures")
    os.makedirs(fig_out, exist_ok=True)
    n_fig = 0
    src_dirs = [figdir]
    if extras_figdir and os.path.isdir(extras_figdir) and extras_figdir != figdir:
        src_dirs.append(extras_figdir)  # new-experiment figures (15_/16_/06c_) live here
    for d in src_dirs:
        for f in os.listdir(d):
            if f.endswith(".png"):
                shutil.copy2(os.path.join(d, f), os.path.join(fig_out, f))
                n_fig += 1
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"Saved {out}  ({len(refs)} references; copied {n_fig} figures to {fig_out})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_colab/results/full_results.json")
    ap.add_argument("--figdir", default="results_colab/figures")
    ap.add_argument("--extras_dir", default="results",
                    help="dir holding optional zeroshot/category_ablation/mixedgen/fewshot/"
                         "ood_diagnostic JSONs")
    ap.add_argument("--extras_figdir", default="figures",
                    help="dir holding the new-experiment figures (15_/16_/06c_)")
    ap.add_argument("--out", default="paper/FAITH-Detect.tex")
    args = ap.parse_args()
    build(load_json(args.results), args.figdir, args.out,
          extras_dir=args.extras_dir, extras_figdir=args.extras_figdir)


if __name__ == "__main__":
    main()
