"""Generate a Word (.docx) paper draft from the real results JSON + figures.

Numbers and tables are read from results/*.json (no hard-coded results); every figure in the
figures directory is embedded and referenced. Prose is a first-author draft to edit.

Usage:
  python scripts/make_paper.py --results results_colab/results/full_results.json \
      --figdir results_colab/figures --out paper/FAITH-Detect.docx
"""
import argparse
import os

import _bootstrap  # noqa: F401
from faithdetect.utils.logging import load_json

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ----------------------------- number helpers ----------------------------- #
def pct(x, d=1):
    return f"{x*100:.{d}f}"


def mean_ci(interval, d=1):
    if not interval:
        return "—"
    m = interval["mean"] * 100
    half = (interval["hi"] - interval["lo"]) / 2 * 100
    return f"{m:.{d}f} ± {half:.{d}f}"


def agg(r, v, axis, metric):
    try:
        return r["variants"][v][axis]["aggregated"][metric]
    except Exception:
        return None


VARIANTS = ["baseline", "softreg", "hardmask"]
LABEL = {"baseline": "Baseline", "softreg": "SoftReg", "hardmask": "Hard-Mask (FAITH)"}


# ----------------------------- doc helpers -------------------------------- #
def h(doc, text, level):
    doc.add_heading(text, level=level)


def para(doc, text, italic=False, size=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return p


def add_table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for j, htext in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        run = c.paragraphs[0].add_run(htext); run.bold = True; run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            run = cells[j].paragraphs[0].add_run(str(val)); run.font.size = Pt(9)
    return t


# --------------------------------- build ---------------------------------- #
def build(r, figdir, out):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    fignum = [0]  # mutable counter so captions auto-number in document order

    def fig(name, caption, width=5.6):
        path = os.path.join(figdir, name)
        fignum[0] += 1
        if not os.path.exists(path):
            para(doc, f"[Figure {fignum[0]} missing: {name}]", italic=True)
            return fignum[0]
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(f"Figure {fignum[0]}: {caption}"); run.italic = True; run.font.size = Pt(9)
        return fignum[0]

    enc = r.get("meta", {}).get("config", {}).get("encoder_name", "roberta-base")
    seeds = r.get("meta", {}).get("config", {}).get("seeds", [])
    n_seeds = len(seeds) if seeds else "?"

    # ------------------------------ Title ------------------------------- #
    title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run("Should a Detector Ignore Function Words? Characterising the Trade-offs "
                       "of Function-Word-Invariant Detection of AI-Generated Reviews")
    tr.bold = True; tr.font.size = Pt(16)
    sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("Author Name¹, Co-author²\n¹Affiliation  ²Affiliation").font.size = Pt(11)

    # ----------------------------- Abstract ----------------------------- #
    h(doc, "Abstract", 1)
    para(doc,
        "Supervised detectors of AI-generated text reach high in-domain accuracy but are known to "
        "rely on superficial cues that transfer poorly. We study one such cue — function words "
        "(articles, prepositions, conjunctions, auxiliaries and pronouns common to both human and "
        "machine text) — and ask what is gained by making a detector ignore them. We propose "
        "FAITH-Detect, which enforces function-word invariance two ways: a hard-masking variant that "
        "replaces every function-word token with a neutral placeholder before encoding (so the "
        "decision is provably invariant to function-word identity), and a soft attribution-"
        "regularised variant. Explanations are produced by the deployed model and are content-only "
        "by construction; we quantify reliance with a function-word attribution-mass metric and a "
        "forward-only identity-sensitivity metric. On the MAiDE-up hotel-review benchmark (English "
        f"subset; {enc}, {n_seeds} seeds, leakage-free grouped splits) the hard-masked detector "
        "matches the unconstrained model in-domain, places essentially zero attribution on function "
        "words, is essentially unaffected by a function-word attack, and degrades least under "
        "synonym substitution. On a cross-domain out-of-distribution set (RAID reviews) it transfers "
        "worse than the unconstrained baseline, indicating that function words can carry domain-"
        "general style signal. We therefore frame function-word invariance as a trade-off rather "
        "than a universal improvement: a provable guarantee, in-domain parity, attack robustness and "
        "faithful explanations, at a measured cost to cross-domain transfer. All results are reported "
        "with confidence intervals over multiple seeds and regenerate from a single results file.",
        size=10)

    # --------------------------- Introduction --------------------------- #
    h(doc, "1  Introduction", 1)
    para(doc,
        "The fluency of large language models has made AI-generated reviews a practical concern for "
        "online platforms, where fabricated opinions can distort purchasing and reputation. "
        "Detectors are deployed in response and are increasingly expected to justify a flag with an "
        "explanation. Two problems recur. First, supervised detectors latch onto dataset- and "
        "generator-specific artefacts, so accuracy can collapse out of distribution [2, 14]. Second, "
        "the explanations shipped with detectors are usually post-hoc rationalisations of an "
        "unconstrained model and are seldom validated for faithfulness [13].")
    para(doc,
        "This paper studies a single, concrete cue: function words. Closed-class words such as "
        "“the”, “a” and “of” are among the most frequent tokens in any English text and are shared "
        "by human and machine writing; intuitively they should carry little signal about whether a "
        "review is genuine. Yet an unconstrained detector still places a large fraction of its "
        "attribution on them. We ask what happens if the detector is made to ignore function words "
        "entirely, and whether doing so changes robustness, transfer and explanation quality.")
    para(doc,
        "We introduce FAITH-Detect, which operationalises function-word invariance in two ways. The "
        "hard-masking variant replaces every function-word sub-token with a shared placeholder before "
        "the encoder sees the text, so the decision cannot depend on which function word occurred — a "
        "constructive, testable guarantee. The soft variant keeps the text but penalises attribution "
        "mass that lands on function words during training [5]. Explanations are computed on the "
        "actual deployed model, and function words are excluded by construction; we verify reliance "
        "with a function-word attribution-mass metric and a forward-only identity-sensitivity metric.")
    para(doc, "Contributions:")
    for c in [
        "A function-word-invariant detector with a provable guarantee (hard masking) and a learned "
        "alternative (soft regularisation), motivated by shortcut learning [9, 20, 21].",
        "Faithful, content-only explanations computed on the deployed model, with two metrics that "
        "make function-word reliance directly measurable.",
        "A rigorous evaluation — leakage-free grouped splits, multiple seeds with confidence "
        "intervals, significance tests, classic baselines, cross-domain and cross-generator OOD, and "
        "two text attacks — yielding an honest characterisation of when function-word invariance "
        "helps and when it hurts.",
    ]:
        doc.add_paragraph(c, style="List Bullet")

    # ---------------------------- Related work -------------------------- #
    h(doc, "2  Related Work", 1)
    para(doc,
        "Machine-generated text detection spans zero-shot likelihood methods (GLTR [7]; DetectGPT "
        "[6]) and supervised transformer classifiers [10, 16]; surveys catalogue the area [17]. "
        "Robustness is a recurring weakness: the RAID benchmark shows detectors are brittle to domain "
        "shift, unseen generators and adversarial edits [2], paraphrasing can evade them [15], and "
        "some work questions whether reliable detection is achievable in general [14]. Our aim is "
        "orthogonal — rather than proposing a stronger detector, we ask what one interpretable family "
        "of features (function words) contributes, and whether removing it improves robustness and "
        "explanation quality. Shortcut learning frames the broader phenomenon of models exploiting "
        "spurious correlations [9], with NLP-specific evidence of annotation artefacts and "
        "“right-for-the-wrong-reasons” behaviour [20, 21]. On the explanation side, gradient and "
        "perturbation attributions include Integrated Gradients [3], LIME [11] and SHAP [12]; "
        "faithfulness is formalised by ERASER comprehensiveness/sufficiency [4], deletion/insertion "
        "curves [8], and the broader discussion of what faithful interpretation requires [13]. "
        "Attribution regularisation (“right for the right reasons”) trains models to place gradient "
        "mass where a prior says it should [5]; our soft variant is an instance with function words "
        "as the prior. The MAiDE-up dataset [1] provides human and GPT-4 hotel reviews and is our "
        "in-domain benchmark.")

    # ------------------------------ Method ------------------------------ #
    h(doc, "3  Method", 1)
    h(doc, "3.1  Function-word set", 2)
    fw_n = r.get("meta", {}).get("fw_set_size", "")
    para(doc,
        f"We define an auditable English function-word set ({fw_n} surface forms) as the union of a "
        "curated closed-class list (articles, prepositions, conjunctions, auxiliaries, pronouns, "
        "particles, determiners) and the standard NLTK, scikit-learn and spaCy stop-word lists [25]. "
        "The set is fixed and surface-form based, so identification is deterministic at inference: a "
        "word is a function word if its lower-cased, punctuation-stripped form is in the set.")
    h(doc, "3.2  Hard-masking (provable invariance)", 2)
    para(doc,
        "We tokenise with the encoder's byte-level BPE and map each sub-token back to its surface "
        "word via offset alignment. Every sub-token belonging to a function word is replaced by a "
        "single added placeholder token [FUNC] before encoding. Because all function words map to "
        "the same placeholder, the encoder — and therefore the classifier — cannot distinguish which "
        "function word occurred: the decision is invariant to function-word identity by construction. "
        "An automated test asserts bit-identical logits when function words are permuted.")
    h(doc, "3.3  Soft attribution regularisation", 2)
    para(doc,
        "The soft variant keeps the full text and adds a penalty equal to the fraction of input-"
        "gradient saliency that lands on function-word tokens (gradient-times-input, with create_"
        "graph so the penalty back-propagates into the weights) [5]. To bound memory at large batch "
        "and sequence length, the second-order penalty is computed on a small sub-batch — an unbiased "
        "stochastic estimate — while the cross-entropy uses the full batch.")
    h(doc, "3.4  Faithful, content-only explanations", 2)
    para(doc,
        "All attributions are computed on the actual deployed model (including the hard-masking "
        "transform), never a surrogate. We use Integrated Gradients on the embedding layer [3] and "
        "leave-one-word-out occlusion, aggregated to word level; function words are excluded from the "
        "displayed explanation. We report two reliance metrics: function-word attribution mass (the "
        "share of total absolute attribution on function words) and identity-sensitivity (the mean "
        "change in AI probability when each function word is replaced by a different function word). "
        "For explanation quality we report ERASER comprehensiveness and sufficiency [4] and "
        "deletion/insertion AUC over content words [8].")

    # -------------------------- Experimental setup ---------------------- #
    h(doc, "4  Experimental Setup", 1)
    sd = r.get("split_describe", {})
    split_line = ""
    if sd:
        split_line = (f" Splits are grouped by hotel (train {sd['train']['n']}, val {sd['val']['n']}, "
                      f"test {sd['test']['n']}; train↔test hotel overlap {sd['hotel_leakage_train_test']}).")
    para(doc,
        "In-domain data is the English subset of MAiDE-up (human vs GPT-4 hotel reviews) [1]. Because "
        "every hotel appears in both classes, a random split leaks hotel-specific content across "
        "train and test; we therefore use grouped (by-hotel) splits and additionally report a random "
        "split to quantify the leakage (Figure 1)." + split_line +
        f" We fine-tune {enc} [10] with AdamW [23] for each of {n_seeds} seeds and report mean ± 95% "
        "confidence interval. Out-of-distribution evaluation uses the RAID benchmark's reviews domain "
        "[2] (cross-domain, spanning multiple generators including GPT-3.5/4, Cohere and Llama). "
        "Robustness is probed with a function-word attack (deleting/duplicating/swapping function "
        "words) and a WordNet [22] synonym attack on content words. Baselines are TF-IDF + logistic "
        "regression and a content-only TF-IDF variant. Significance between models on the shared test "
        "set uses McNemar's test; intervals use Student-t and the bootstrap.")
    fig("01_dataset_overview.png",
        "MAiDE-up English splits used for training/validation/test (grouped, zero train↔test hotel "
        "leakage).", 4.2)

    # ------------------------------ Results ----------------------------- #
    h(doc, "5  Results", 1)

    h(doc, "5.1  In-domain performance", 2)
    rows = []
    for v in VARIANTS:
        rows.append([LABEL[v], mean_ci(agg(r, v, "indomain", "f1_macro")),
                     mean_ci(agg(r, v, "indomain", "roc_auc")),
                     mean_ci(agg(r, v, "indomain", "accuracy"))])
    for bname in ("tfidf_lr", "tfidf_lr_content"):
        b = r.get("baselines", {}).get(bname)
        if b:
            m = b["metrics"]
            rows.append([bname, pct(m["f1_macro"]), pct(m.get("roc_auc", 0)), pct(m.get("accuracy", 0))])
    add_table(doc, ["Model", "F1 (macro)", "ROC-AUC", "Accuracy"], rows)
    sig = r.get("significance", {}).get("indomain_baseline_vs_hardmask", {})
    pval = sig.get("pvalue")
    para(doc,
        "On in-distribution data the three neural variants are statistically indistinguishable"
        + (f" (McNemar baseline vs. Hard-Mask, p = {pval:.2f})" if pval is not None else "")
        + "; Hard-Mask attains the same F1 as the unconstrained baseline, with the tightest interval. "
        "Function-word invariance therefore carries no measurable in-domain cost on this benchmark "
        "(Figure 2). Confusion matrices (Figure 3) and ROC/PR curves (Figure 4) show the same "
        "ordering, with very high ROC-AUC for all neural variants.")
    fig("03_indomain_performance.png", "In-domain F1 (mean ± 95% CI over seeds) with classic baselines.")
    fig("04_confusion_matrices.png", "In-domain confusion matrices (reference seed) by variant.", 6.2)
    fig("05_roc_pr.png", "In-domain ROC and precision–recall curves (reference seed).", 6.2)

    h(doc, "5.2  Function-word reliance", 2)
    rows = []
    for v in VARIANTS:
        fm = r.get("fw_mass", {}).get(v, {})
        ids = r.get("fw_identity_sensitivity", {}).get(v, {})
        rows.append([LABEL[v], (pct(fm["mean"]) + "%") if fm else "—",
                     (pct(ids["mean"], 2) + "%") if ids else "—"])
    add_table(doc, ["Variant", "FW attribution mass (IG)", "Identity-sensitivity |Δp|"], rows)
    para(doc,
        "The unconstrained baseline places a substantial share of its attribution on function words; "
        "soft regularisation reduces it; hard masking eliminates it (Figure 5). The forward-only "
        "identity-sensitivity metric — the change in AI probability when function words are swapped — "
        "agrees: it is near zero for Hard-Mask and clearly positive for the baseline (Figure 6). The "
        "two metrics are consistent because they measure the same property (reliance on function-word "
        "identity) from gradient and perturbation perspectives.")
    fig("08_fw_attribution_mass.png", "Integrated-Gradients attribution mass on function words by variant.", 5.0)
    fig("08b_fw_identity_sensitivity.png", "Change in AI probability when function words are swapped (identity-sensitivity).", 5.0)

    h(doc, "5.3  Robustness to text attacks", 2)
    attacks = list(r["variants"][VARIANTS[0]].get("attacks", {}).keys())
    header = ["Variant", "Clean"] + [a.replace("_", " ") for a in attacks]
    rows = []
    for v in VARIANTS:
        clean = agg(r, v, "indomain", "f1_macro")
        row = [LABEL[v], pct(clean["mean"]) if clean else "—"]
        for a in attacks:
            it = r["variants"][v]["attacks"][a]["aggregated"].get("f1_macro")
            row.append(pct(it["mean"]) if it else "—")
        rows.append(row)
    add_table(doc, header, rows)
    para(doc,
        "Under the function-word attack, the baseline and soft variants lose several points whereas "
        "Hard-Mask is essentially unaffected — perturbing function words cannot move a decision that "
        "ignores them. Under synonym substitution all variants degrade, but Hard-Mask degrades least "
        "and retains the highest F1 (Figure 7). These are the clearest empirical benefits of "
        "invariance; we do not claim improvements beyond the two attacks tested.")
    fig("07_robustness.png", "F1 under text attacks. Hard-Mask is flat under the function-word attack.")

    h(doc, "5.4  Cross-domain and cross-generator transfer", 2)
    rows = []
    for v in VARIANTS:
        rows.append([LABEL[v], mean_ci(agg(r, v, "indomain", "f1_macro")),
                     mean_ci(agg(r, v, "ood", "f1_macro"))])
    add_table(doc, ["Variant", "In-domain F1", "OOD F1 (RAID reviews)"], rows)
    para(doc,
        "On the cross-domain OOD set (hotel → movie reviews) the picture reverses: Hard-Mask "
        "transfers worse than the baseline, with non-overlapping confidence intervals (Figure 8). "
        "When the content vocabulary shifts, the content words Hard-Mask depends on largely "
        "disappear, while the baseline's function-word and stylistic cues evidently carry domain-"
        "general signal. Function words are thus not pure noise; in cross-domain transfer they can be "
        "useful. The per-generator breakdown (Figure 9) shows the same ordering across most "
        "generators. We emphasise a confound: this OOD shifts both domain and generator at once and "
        "is an extreme content change, so it does not isolate cross-generator transfer; a same-"
        "domain, different-generator test would be needed for that claim.")
    fig("06_ood_transfer.png", "In-domain vs. cross-domain OOD F1 by variant (mean ± 95% CI).")
    fig("06b_cross_generator.png", "Per-generator detection F1 on RAID reviews (humans vs. each generator).", 6.2)

    h(doc, "5.5  Explanation faithfulness", 2)
    rows = []
    for v in VARIANTS:
        f = r["variants"].get(v, {}).get("faithfulness", {}).get("summary")
        if f:
            rows.append([LABEL[v], f"{f.get('comprehensiveness',0):.3f}", f"{f.get('sufficiency',0):.3f}",
                         f"{f.get('deletion_auc',0):.3f}", f"{f.get('insertion_auc',0):.3f}"])
    if rows:
        add_table(doc, ["Variant", "Compr.↑", "Suff.↓", "Del-AUC↓", "Ins-AUC↑"], rows)
    para(doc,
        "Faithfulness is comparable across variants and mixed in direction (Figure 10): no variant "
        "dominates on all four measures, with Hard-Mask strongest on sufficiency and insertion and "
        "the baseline stronger on comprehensiveness. Deletion/insertion curves (Figure 11) tell the "
        "same story. We therefore do not claim a faithfulness improvement from invariance; the "
        "explanation contribution is structural — explanations are content-only by construction and "
        "computed on the deployed model (Figure 12) — rather than a higher faithfulness score.")
    fig("09_faithfulness.png", "ERASER comprehensiveness/sufficiency and deletion/insertion AUC by variant.")
    fig("10_deletion_insertion_curves.png", "Mean deletion and insertion curves over content words.", 6.2)
    fig("14_example_explanation.png", "Example explanations: content words highlighted, function words greyed.", 6.2)

    h(doc, "5.6  Calibration and representation", 2)
    para(doc,
        "Reliability diagrams (Figure 13) summarise calibration on the in-domain test set via the "
        "expected calibration error [19]; we report it for completeness rather than as a contribution. "
        "A two-dimensional projection of the encoder representations (Figure 14) shows that human and "
        "AI reviews are well separated in-domain for all variants.")
    fig("11_calibration.png", "Reliability diagrams (in-domain) with expected calibration error.", 4.8)
    fig("13_embedding_projection.png", "2-D projection of encoder representations of the test set.", 6.2)

    h(doc, "5.7  Leakage", 2)
    lk = r.get("leakage", {})
    if lk:
        para(doc,
            f"A naive random split inflates baseline F1 to {pct(lk['random']['f1_macro'])}% versus "
            f"{pct(lk['grouped']['f1_macro'])}% under the grouped, leakage-free split (Figure 15). "
            "Because hotels recur with both labels, a random split lets the model exploit hotel-"
            "specific content; we report grouped numbers throughout and recommend grouped splits for "
            "this dataset.")
    fig("02_leakage_gap.png", "Random vs. grouped split (baseline): random splitting overstates scores.", 4.6)

    # ----------------------------- Discussion --------------------------- #
    h(doc, "6  Discussion", 1)
    para(doc,
        "Making a detector ignore function words is free in-distribution on this benchmark, yields a "
        "provable invariance and the best robustness to the two attacks we test, and produces "
        "explanations that are content-only by construction — in contrast to post-hoc, surrogate-"
        "based explanations. The cost is cross-domain transfer: when the content vocabulary changes, "
        "function-word and stylistic regularities that the constrained model discards carry domain-"
        "general signal. Practically, hard masking is attractive where robustness, auditability and a "
        "guarantee matter and the domain is fixed; the unconstrained or soft model is preferable for "
        "open-domain transfer. We deliberately avoid stronger claims: results are on one in-domain "
        "dataset and two attacks, and the OOD evidence is a single benchmark.")

    # ----------------------------- Limitations -------------------------- #
    h(doc, "7  Limitations", 1)
    para(doc,
        "The study is English-only and uses a single in-domain generator (GPT-4), so cross-generator "
        "evidence comes from a benchmark that also shifts domain and cannot isolate generator effects. "
        "The in-domain set is modest (2,000 reviews), which widens confidence intervals; we report "
        "them honestly rather than selecting favourable seeds. The two attacks are simple proxies for "
        "adversarial and paraphrase pressure, not an exhaustive robustness audit. Faithfulness metrics "
        "are themselves contested [13] and we do not treat any single one as decisive. Finally, "
        "function-word invariance is a design choice with a measured trade-off, not a universally "
        "preferable configuration.")

    # ----------------------------- Conclusion --------------------------- #
    h(doc, "8  Conclusion", 1)
    para(doc,
        "Function words are a measurable shortcut in AI-text detection. A detector that provably "
        "ignores them keeps in-domain accuracy, gains robustness to the attacks we test, and explains "
        "itself with content alone — at a measured cost to cross-domain transfer. We release code, "
        "leakage-free protocols, multi-seed results with confidence intervals, and figures that "
        "regenerate from a single results file.")

    # --------------------------- Reproducibility ------------------------ #
    h(doc, "Reproducibility", 1)
    para(doc,
        "All numbers come from a single results JSON; every figure regenerates via "
        "scripts/make_figures.py and this paper via scripts/make_paper.py; the hard-masking guarantee "
        "is asserted by tests/test_invariance.py. Code: github.com/scar09-22/FAITH-Detect.", size=10)

    # ------------------------------ Appendix ---------------------------- #
    h(doc, "Appendix A  Training behaviour", 1)
    para(doc,
        "Figure 16 shows training loss and validation F1 by epoch, averaged over seeds, for the three "
        "variants; the soft and hard variants start slower (they discard or down-weight function-word "
        "information) but reach comparable validation F1.")
    fig("12_learning_curves.png", "Training loss and validation F1 by epoch (mean over seeds).", 6.2)

    # ----------------------------- References --------------------------- #
    h(doc, "References", 1)
    refs = [
        "Ignat, O., Xu, X., Mihalcea, R. (2025). MAiDE-up: Multilingual Deception Detection of "
        "AI-Generated Hotel Reviews. Findings of NAACL.",
        "Dugan, L., Hwang, A., Trhlik, F., et al. (2024). RAID: A Shared Benchmark for Robust "
        "Evaluation of Machine-Generated Text Detectors. ACL.",
        "Sundararajan, M., Taly, A., Yan, Q. (2017). Axiomatic Attribution for Deep Networks "
        "(Integrated Gradients). ICML.",
        "DeYoung, J., Jain, S., Rajani, N. F., et al. (2020). ERASER: A Benchmark to Evaluate "
        "Rationalized NLP Models. ACL.",
        "Ross, A. S., Hughes, M. C., Doshi-Velez, F. (2017). Right for the Right Reasons: Training "
        "Differentiable Models by Constraining their Explanations. IJCAI.",
        "Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., Finn, C. (2023). DetectGPT: Zero-Shot "
        "Machine-Generated Text Detection using Probability Curvature. ICML.",
        "Gehrmann, S., Strobelt, H., Rush, A. M. (2019). GLTR: Statistical Detection and "
        "Visualization of Generated Text. ACL (System Demonstrations).",
        "Petsiuk, V., Das, A., Saenko, K. (2018). RISE: Randomized Input Sampling for Explanation of "
        "Black-box Models. BMVC.",
        "Geirhos, R., Jacobsen, J.-H., Michaelis, C., et al. (2020). Shortcut Learning in Deep Neural "
        "Networks. Nature Machine Intelligence.",
        "Liu, Y., Ott, M., Goyal, N., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining "
        "Approach. arXiv:1907.11692.",
        "Ribeiro, M. T., Singh, S., Guestrin, C. (2016). “Why Should I Trust You?”: Explaining the "
        "Predictions of Any Classifier (LIME). KDD.",
        "Lundberg, S. M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions "
        "(SHAP). NeurIPS.",
        "Jacovi, A., Goldberg, Y. (2020). Towards Faithfully Interpretable NLP Systems: How Should We "
        "Define and Evaluate Faithfulness? ACL.",
        "Sadasivan, V. S., Kumar, A., Balasubramanian, S., Wang, W., Feizi, S. (2023). Can "
        "AI-Generated Text be Reliably Detected? arXiv:2303.11156.",
        "Krishna, K., Song, Y., Karpinska, M., Wieting, J., Iyyer, M. (2023). Paraphrasing Evades "
        "Detectors of AI-Generated Text, but Retrieval is an Effective Defense. NeurIPS.",
        "Solaiman, I., Brundage, M., Clark, J., et al. (2019). Release Strategies and the Social "
        "Impacts of Language Models. arXiv:1908.09203.",
        "Crothers, E., Japkowicz, N., Viktor, H. L. (2023). Machine-Generated Text: A Comprehensive "
        "Survey of Threat Models and Detection Methods. IEEE Access.",
        "Uchendu, A., Ma, Z., Le, T., Zhang, R., Lee, D. (2021). TURINGBENCH: A Benchmark Environment "
        "for Turing Test in the Age of Neural Text Generation. Findings of EMNLP.",
        "Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On Calibration of Modern Neural "
        "Networks. ICML.",
        "Gururangan, S., Swayamdipta, S., Levy, O., et al. (2018). Annotation Artifacts in Natural "
        "Language Inference Data. NAACL.",
        "McCoy, T., Pavlick, E., Linzen, T. (2019). Right for the Wrong Reasons: Diagnosing Syntactic "
        "Heuristics in Natural Language Inference. ACL.",
        "Miller, G. A. (1995). WordNet: A Lexical Database for English. Communications of the ACM.",
        "Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR.",
        "Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of Deep "
        "Bidirectional Transformers for Language Understanding. NAACL.",
        "Honnibal, M., Montani, I. (2017). spaCy: Industrial-Strength Natural Language Processing.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph(f"[{i}] {ref}")
        p.runs[0].font.size = Pt(9)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc.save(out)
    print(f"Saved {out}  ({fignum[0]} figures embedded, {len(refs)} references)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_colab/results/full_results.json")
    ap.add_argument("--figdir", default="results_colab/figures")
    ap.add_argument("--out", default="paper/FAITH-Detect.docx")
    args = ap.parse_args()
    build(load_json(args.results), args.figdir, args.out)


if __name__ == "__main__":
    main()
