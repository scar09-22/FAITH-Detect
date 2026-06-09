"""Quantitative explanation-faithfulness metrics, all computed on the real model.

* function_word_attribution_mass : fraction of total |attribution| that lands on function
  words. This is the metric that operationalises the user's requirement -- it should be
  ~0 for the hard-masked model and is measured (not assumed) for every variant.
* comprehensiveness / sufficiency : ERASER (DeYoung et al., 2020). Remove / keep the most
  important CONTENT words and measure the change in the predicted-class probability.
* deletion / insertion AUC : progressively delete (or insert) content words in importance
  order and integrate the predicted-probability curve (Petsiuk et al., 2018).

Manipulations operate on CONTENT words and keep function words in place, consistent with the
claim that the decision should rest on content, not function words.
"""
from __future__ import annotations

import numpy as np

from .attributions import FaithfulExplainer, WordAttribution


def _p_pred(p_ai: float, pred: int) -> float:
    return p_ai if pred == 1 else 1.0 - p_ai


_FW_PERTURB_POOL = [
    "the", "a", "an", "this", "that", "these", "those", "of", "in", "on", "at",
    "and", "but", "or", "is", "was", "to", "for", "with", "by", "from",
]


def function_word_identity_sensitivity(explainer, texts, seed: int = 0, max_texts=None) -> list:
    """|Δp(AI)| when every function word is swapped for a *different* random function word.

    This is the most direct test of the user's requirement ("function-word identity must not
    affect the decision"): for the Hard-Mask model the swap leaves the encoded input identical
    (all function words map to ``[FUNC]``) so the change is exactly 0; for the baseline it is
    nonzero. Uses forward passes only (robust on every backend).
    """
    import random

    from ..function_words import _normalize_surface

    fw_set = explainer.fw_set
    pool = [w for w in _FW_PERTURB_POOL if w in fw_set.words] or ["the", "a"]
    rng = random.Random(seed)
    if max_texts:
        texts = list(texts)[:max_texts]
    vals = []
    for t in texts:
        p0 = float(explainer.proba(t)[0])
        perturbed = [
            rng.choice(pool) if _normalize_surface(w) in fw_set.words else w
            for w in t.split()
        ]
        p1 = float(explainer.proba(" ".join(perturbed))[0])
        vals.append(abs(p0 - p1))
    return vals


def function_word_attribution_mass(attr: WordAttribution) -> dict:
    """Share of total |attribution| on function words (word-level)."""
    abs_scores = np.abs(np.array(attr.word_scores, dtype=float))
    is_fw = np.array(attr.word_is_fw, dtype=bool)
    total = float(abs_scores.sum())
    if total <= 0:
        return {"fw_mass": 0.0, "content_mass": 0.0, "n_words": len(attr.words)}
    fw_mass = float(abs_scores[is_fw].sum() / total)
    return {
        "fw_mass": fw_mass,
        "content_mass": 1.0 - fw_mass,
        "n_words": len(attr.words),
        "n_fw_words": int(is_fw.sum()),
    }


def _is_content(word: str, is_fw: bool) -> bool:
    return (not is_fw) and any(c.isalnum() for c in word)


def _content_order(attr: WordAttribution) -> list[int]:
    """Content-word indices sorted by importance toward the PREDICTED class (desc).
    Function words and punctuation-only tokens are excluded."""
    sign = 1.0 if attr.predicted_label == 1 else -1.0
    ranked = [
        (i, sign * attr.word_scores[i])
        for i in range(len(attr.words))
        if _is_content(attr.words[i], attr.word_is_fw[i])
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked]


def _text_from_indices(attr: WordAttribution, keep: set[int]) -> str:
    return " ".join(w for i, w in enumerate(attr.words) if i in keep)


def comprehensiveness_sufficiency(
    explainer: FaithfulExplainer, attr: WordAttribution, k_fraction: float = 0.2
) -> dict:
    order = _content_order(attr)
    if not order:
        return {"comprehensiveness": 0.0, "sufficiency": 0.0}
    k = max(1, int(round(k_fraction * len(order))))
    top = set(order[:k])
    all_idx = set(range(len(attr.words)))
    fw_idx = {i for i in all_idx if attr.word_is_fw[i]}

    p0 = _p_pred(attr.p_ai, attr.predicted_label)
    # Comprehensiveness: remove top-k content words (keep everything else incl. function words).
    keep_comp = all_idx - top
    p_comp = _p_pred(float(explainer.proba(_text_from_indices(attr, keep_comp))[0]), attr.predicted_label)
    # Sufficiency: keep only top-k content words plus function words.
    keep_suff = top | fw_idx
    p_suff = _p_pred(float(explainer.proba(_text_from_indices(attr, keep_suff))[0]), attr.predicted_label)
    return {
        "comprehensiveness": float(p0 - p_comp),   # higher is better
        "sufficiency": float(p0 - p_suff),         # lower (closer to 0) is better
        "k_fraction": k_fraction,
    }


def deletion_insertion_auc(explainer: FaithfulExplainer, attr: WordAttribution) -> dict:
    order = _content_order(attr)
    all_idx = set(range(len(attr.words)))
    fw_idx = {i for i in all_idx if attr.word_is_fw[i]}
    if not order:
        return {"deletion_auc": float("nan"), "insertion_auc": float("nan")}

    # Deletion: start from full text, remove content words most-important first.
    del_curve = [_p_pred(attr.p_ai, attr.predicted_label)]
    present = set(all_idx)
    for idx in order:
        present = present - {idx}
        txt = _text_from_indices(attr, present)
        p = _p_pred(float(explainer.proba(txt)[0]) if txt.strip() else 0.5, attr.predicted_label)
        del_curve.append(p)

    # Insertion: start from function words only, add content words most-important first.
    ins_present = set(fw_idx)
    txt0 = _text_from_indices(attr, ins_present)
    ins_curve = [_p_pred(float(explainer.proba(txt0)[0]) if txt0.strip() else 0.5, attr.predicted_label)]
    for idx in order:
        ins_present = ins_present | {idx}
        txt = _text_from_indices(attr, ins_present)
        p = _p_pred(float(explainer.proba(txt)[0]), attr.predicted_label)
        ins_curve.append(p)

    xs = np.linspace(0, 1, len(del_curve))
    return {
        "deletion_auc": float(np.trapz(del_curve, xs)),     # lower is better
        "insertion_auc": float(np.trapz(ins_curve, xs)),    # higher is better
        "deletion_curve": del_curve,
        "insertion_curve": ins_curve,
    }


def faithfulness_report(
    explainer: FaithfulExplainer,
    texts,
    method: str = "ig",
    n_steps: int = 32,
    k_fraction: float = 0.2,
    max_texts: int | None = None,
) -> dict:
    """Aggregate faithfulness + FW-attribution-mass over a sample of texts."""
    if max_texts:
        texts = list(texts)[:max_texts]
    grid = np.linspace(0, 1, 11)
    rows, del_curves, ins_curves = [], [], []
    for t in texts:
        attr = (
            explainer.integrated_gradients(t, n_steps=n_steps)
            if method == "ig" else explainer.occlusion(t)
        )
        fw = function_word_attribution_mass(attr)
        cs = comprehensiveness_sufficiency(explainer, attr, k_fraction)
        di = deletion_insertion_auc(explainer, attr)
        rows.append({
            "fw_mass": fw["fw_mass"],
            "comprehensiveness": cs["comprehensiveness"],
            "sufficiency": cs["sufficiency"],
            "deletion_auc": di["deletion_auc"],
            "insertion_auc": di["insertion_auc"],
        })
        # Interpolate variable-length curves onto a common grid for averaging.
        for curve, store in ((di.get("deletion_curve"), del_curves),
                             (di.get("insertion_curve"), ins_curves)):
            if curve and len(curve) > 1:
                xs = np.linspace(0, 1, len(curve))
                store.append(np.interp(grid, xs, curve))
    keys = rows[0].keys() if rows else []
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    summary["n_texts"] = len(rows)
    summary["method"] = method
    curves = {
        "grid": grid.tolist(),
        "deletion_mean": (np.mean(del_curves, axis=0).tolist() if del_curves else []),
        "insertion_mean": (np.mean(ins_curves, axis=0).tolist() if ins_curves else []),
    }
    return {"summary": summary, "per_text": rows, "curves": curves}
