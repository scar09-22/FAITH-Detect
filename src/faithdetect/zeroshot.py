"""Zero-shot machine-generated-text detectors used as modern baselines.

Unlike the supervised variants in this repo, these detectors require NO training: each one
scores a text with a small causal LM (GPT-2 by default) and thresholds the score. All scores
are oriented so that HIGHER = more likely AI-generated:

  * ``loglik``        : +mean token log-likelihood (AI text is more probable under the LM).
  * ``logrank``       : -mean log rank of each observed token in the LM's sorted next-token
                        distribution (AI text uses lower-rank tokens).
  * ``entropy``       : -mean predictive entropy of the next-token distribution (the LM is
                        less uncertain on AI text).
  * ``fast_detectgpt``: Fast-DetectGPT sampling discrepancy (Bao et al., ICLR 2024) with the
                        SAME model as scoring and sampling model, in the paper's analytic form:
                        (sum_t log p(x_t) - sum_t E_{x~p}[log p(x)]) / sqrt(sum_t Var_{x~p}[log p(x)]),
                        with E and Var computed exactly from the softmax distribution.

Thresholds are picked on (a subsample of) the training split by maximising macro-F1, so the
comparison to the supervised models stays honest: no test-set information is used.

These scores are NOT probabilities. For the metric fields that expect probabilities
(``pr_auc``/``ece`` inside :func:`faithdetect.evaluate.classification_metrics`), test scores
are min-max normalised to [0, 1] as a pseudo-probability; interpret those fields accordingly.
AUROC is additionally reported on the raw scores (it is rank-based, so the normalisation is
irrelevant there).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .evaluate import classification_metrics

ZEROSHOT_METHODS = ("loglik", "logrank", "entropy", "fast_detectgpt")

# Cap on threshold candidates: above this the sweep switches from exact score midpoints to
# quantiles so threshold selection stays O(n * candidates) cheap on large train subsamples.
_MAX_THRESHOLD_CANDIDATES = 512


class ZeroShotScorer:
    """Lazy causal-LM wrapper scoring texts with training-free detection statistics."""

    def __init__(self, model_name: str = "gpt2", device: str = "cpu", max_tokens: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self._model = None
        self._tok = None

    def _ensure(self) -> None:
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device).eval()
            )

    def _forward(self, text: str):
        """One batch-of-1 forward pass. Returns (logits[T-1, V], targets[T-1]) aligned so
        logits[i] is the distribution that predicted targets[i], or None for texts that are
        empty or tokenise to fewer than 2 tokens (no next-token prediction exists)."""
        import torch
        self._ensure()
        if not text or not text.strip():
            return None
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=self.max_tokens)
        ids = enc["input_ids"].to(self.device)
        if ids.shape[1] < 2:
            return None
        with torch.no_grad():
            out = self._model(ids)
        # float32 keeps log_softmax/entropy numerically stable across cpu and mps.
        return out.logits[0, :-1].float(), ids[0, 1:]

    def _score_one(self, text: str, method: str) -> float:
        import torch
        fwd = self._forward(text)
        if fwd is None:
            return float("nan")
        logits, targets = fwd
        with torch.no_grad():
            logp = torch.log_softmax(logits, dim=-1)
            obs = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
            if method == "loglik":
                return float(obs.mean().item())
            if method == "logrank":
                # 1-based rank of the observed token; ties resolve to the best rank.
                ranks = (logp > obs.unsqueeze(1)).sum(dim=1).float() + 1.0
                return float((-torch.log(ranks).mean()).item())
            p = logp.exp()
            if method == "entropy":
                ent = -(p * logp).sum(dim=1)
                return float((-ent.mean()).item())
            if method == "fast_detectgpt":
                # Reference analytic statistic (Bao et al.): one global z-score of the total
                # log-likelihood, NOT an average of per-position z-scores.
                mean = (p * logp).sum(dim=1)
                var = (p * logp.square()).sum(dim=1) - mean.square()
                denom = var.sum().clamp_min(1e-12).sqrt()
                return float(((obs.sum() - mean.sum()) / denom).item())
        raise ValueError(f"method must be one of {ZEROSHOT_METHODS}, got {method!r}")

    def scores(self, texts, method: str) -> np.ndarray:
        """Score each text; higher = more likely AI-generated. NaN for unscorable texts."""
        if method not in ZEROSHOT_METHODS:
            raise ValueError(f"method must be one of {ZEROSHOT_METHODS}, got {method!r}")
        return np.array([self._score_one(t, method) for t in texts], dtype=float)


def _stratified_subsample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    sub, _ = train_test_split(df, train_size=n, random_state=seed, stratify=df["label"])
    return sub.reset_index(drop=True)


def _impute_nan(scores: np.ndarray) -> np.ndarray:
    """Replace NaN scores with the finite minimum (= maximally human-like) so downstream
    metrics see complete arrays. NaNs only arise from empty/single-token texts."""
    finite = scores[np.isfinite(scores)]
    fill = float(finite.min()) if finite.size else 0.0
    return np.where(np.isfinite(scores), scores, fill)


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising macro-F1 of the rule ``score >= threshold -> AI``."""
    finite = np.isfinite(scores)
    y, s = np.asarray(y_true)[finite], scores[finite]
    uniq = np.unique(s)
    if uniq.size < 2:
        return float(uniq[0]) if uniq.size else 0.0
    candidates = (uniq[:-1] + uniq[1:]) / 2.0
    if candidates.size > _MAX_THRESHOLD_CANDIDATES:
        candidates = np.unique(
            np.quantile(s, np.linspace(0.0, 1.0, _MAX_THRESHOLD_CANDIDATES + 2)[1:-1])
        )
    best_thr, best_f1 = float(candidates[0]), -1.0
    for thr in candidates:
        f1 = f1_score(y, (s >= thr).astype(int), average="macro")
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr


def evaluate_zeroshot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    methods=ZEROSHOT_METHODS,
    model_name: str = "gpt2",
    device: str = "cpu",
    max_train: int | None = 400,
    max_test: int | None = None,
    seed: int = 0,
) -> dict:
    """Run training-free detectors: fit a threshold on train scores, evaluate on test.

    The train frame is subsampled to ``max_train`` rows (stratified by label, fixed seed)
    purely to pick the per-method decision threshold (macro-F1-optimal over score midpoints,
    or quantiles when there are many candidates); the test frame may be capped via
    ``max_test`` the same way. Scores are not probabilities — the pseudo-probability passed
    to :func:`classification_metrics` (which feeds its ``pr_auc``/``ece`` fields) is the
    min-max normalisation of the raw test scores to [0, 1]. ``auroc`` is computed on the raw
    scores directly.

    Returns ``{method: {"metrics": ..., "threshold": float, "auroc": float}}``.
    """
    train = _stratified_subsample(train_df, max_train, seed)
    test = _stratified_subsample(test_df, max_test, seed)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    scorer = ZeroShotScorer(model_name=model_name, device=device)

    out: dict = {}
    for method in methods:
        s_train = scorer.scores(train["text"].tolist(), method)
        s_test = _impute_nan(scorer.scores(test["text"].tolist(), method))
        threshold = _best_threshold(y_train, s_train)
        y_pred = (s_test >= threshold).astype(int)
        lo, hi = float(s_test.min()), float(s_test.max())
        p_like = (s_test - lo) / (hi - lo) if hi > lo else np.full_like(s_test, 0.5)
        out[method] = {
            "metrics": classification_metrics(y_test, y_pred, p_like),
            "threshold": float(threshold),
            "auroc": float(roc_auc_score(y_test, s_test)) if len(np.unique(y_test)) == 2 else float("nan"),
        }
    return out
