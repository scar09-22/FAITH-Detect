"""Statistical rigor utilities: confidence intervals, significance tests, calibration.

Addresses flaw F9 (no CIs / no significance tests). Every headline number in the paper is
reported as mean +/- SD with a 95% CI, and model-vs-model comparisons use a paired
significance test (McNemar on the shared test set, plus a paired bootstrap on a metric).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
from scipy import stats as scipy_stats

try:  # statsmodels is optional at import time; only needed for McNemar.
    from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar
except Exception:  # pragma: no cover
    _sm_mcnemar = None


@dataclass
class Interval:
    mean: float
    sd: float
    lo: float
    hi: float
    n: int
    method: str

    def as_dict(self) -> dict:
        return asdict(self)


def t_interval(values: Sequence[float], confidence: float = 0.95) -> Interval:
    """Student-t confidence interval for the mean of a small sample (e.g. seeds)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = sd / np.sqrt(n)
        tcrit = scipy_stats.t.ppf(0.5 + confidence / 2, df=n - 1)
        half = float(tcrit * se)
    else:
        half = 0.0
    return Interval(mean, sd, mean - half, mean + half, n, f"t-{int(confidence*100)}")


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 10_000,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for an arbitrary statistic of a 1-D sample."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    rng = np.random.default_rng(seed)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), float("nan"), 0, "bootstrap")
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = statistic(arr[rng.integers(0, n, size=n)])
    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return Interval(
        float(statistic(arr)), float(arr.std(ddof=1) if n > 1 else 0.0),
        float(lo), float(hi), n, f"bootstrap-{int(confidence*100)}",
    )


def mean_sd_ci(values: Sequence[float], confidence: float = 0.95) -> Interval:
    """Default reporting helper: t-interval (right for small seed counts)."""
    return t_interval(values, confidence)


def aggregate_seeds(per_seed_metrics: list[dict[str, float]], confidence: float = 0.95) -> dict[str, dict]:
    """Given a list of per-seed metric dicts, return {metric: Interval.as_dict()}."""
    if not per_seed_metrics:
        return {}
    keys = set().union(*[set(d.keys()) for d in per_seed_metrics])
    out: dict[str, dict] = {}
    for k in sorted(keys):
        vals = [d[k] for d in per_seed_metrics if k in d and d[k] is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            out[k] = mean_sd_ci(vals, confidence).as_dict()
    return out


def fmt_mean_ci(interval: dict | Interval, pct: bool = True, decimals: int = 1) -> str:
    """Format an Interval as 'mean +/- half [lo, hi]' for tables/captions."""
    d = interval.as_dict() if isinstance(interval, Interval) else interval
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    half = (d["hi"] - d["lo"]) / 2
    return (
        f"{d['mean']*scale:.{decimals}f}{suffix} "
        f"(+/-{half*scale:.{decimals}f}, 95% CI [{d['lo']*scale:.{decimals}f}, {d['hi']*scale:.{decimals}f}])"
    )


def mcnemar_test(y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int]) -> dict:
    """McNemar's test comparing two classifiers on the SAME test set.

    Returns the discordant counts and p-value. b = A-correct/B-wrong, c = A-wrong/B-correct.
    Uses the exact binomial test for small discordant counts (statsmodels), else chi-square
    with continuity correction.
    """
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    a_correct = a == y
    b_correct = b == y
    n01 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    n10 = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    result = {"n_a_right_b_wrong": n01, "n_a_wrong_b_right": n10}
    if _sm_mcnemar is not None:
        table = [[0, n01], [n10, 0]]
        exact = (n01 + n10) < 25
        res = _sm_mcnemar(table, exact=exact, correction=True)
        result["statistic"] = float(res.statistic)
        result["pvalue"] = float(res.pvalue)
        result["method"] = "exact" if exact else "chi2-cc"
    else:  # pragma: no cover
        n = n01 + n10
        stat = (abs(n01 - n10) - 1) ** 2 / n if n > 0 else 0.0
        result["statistic"] = float(stat)
        result["pvalue"] = float(scipy_stats.chi2.sf(stat, df=1))
        result["method"] = "chi2-cc-fallback"
    return result


def paired_bootstrap_diff(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap test for the difference in a metric between two models.

    Resamples test indices once per bootstrap and applies them to BOTH models (paired),
    yielding a CI on (metric_A - metric_B) and a two-sided p-value for H0: diff = 0.
    `score_*` are predictions or scores accepted by `metric_fn(y_true, score)`.
    """
    y = np.asarray(y_true)
    sa = np.asarray(score_a)
    sb = np.asarray(score_b)
    n = y.size
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric_fn(y[idx], sa[idx]) - metric_fn(y[idx], sb[idx])
    observed = metric_fn(y, sa) - metric_fn(y, sb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p-value: proportion of bootstrap diffs on the opposite side of 0.
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "observed_diff": float(observed),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "pvalue": float(min(1.0, p)),
        "n_boot": n_boot,
    }


def expected_calibration_error(
    y_true: Sequence[int], probs: Sequence[float], n_bins: int = 10
) -> dict:
    """Expected Calibration Error (ECE) and per-bin reliability data.

    `probs` is the predicted probability of the positive (AI) class. Bins by confidence of
    the predicted class. Returns ECE plus arrays for a reliability diagram.
    """
    y = np.asarray(y_true)
    p_pos = np.asarray(probs, dtype=float)
    pred = (p_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, p_pos, 1 - p_pos)  # confidence of the predicted class
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc, bin_conf, bin_count = [], [], []
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            acc = float(correct[mask].mean())
            cf = float(conf[mask].mean())
            ece += (cnt / n) * abs(acc - cf)
        else:
            acc, cf = float("nan"), float((lo + hi) / 2)
        bin_acc.append(acc)
        bin_conf.append(cf)
        bin_count.append(cnt)
    return {
        "ece": float(ece),
        "bin_edges": bins.tolist(),
        "bin_accuracy": bin_acc,
        "bin_confidence": bin_conf,
        "bin_count": bin_count,
    }
