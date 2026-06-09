"""Interpretable linguistic features — provided as an optional ABLATION.

The main detector is intentionally encoder-only; this module offers a small set of clean,
interpretable English features for a "+interpretable features" ablation:

* `causal_lm_perplexity` uses a genuine **causal** LM (GPT-2): perplexity = exp(mean
  token negative-log-likelihood) over the *whole* text (token-level cap, no char truncation).
* `burstiness` uses sentence-length coefficient of variation over proper sentence splits.
* `content_pos_ratios` reports POS ratios for **content** classes only (NOUN/VERB/ADJ/ADV/
  PROPN/NUM) — deliberately excluding function-word POS, consistent with the project's thesis.

Features are standardised by `StandardFeatureScaler`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTENT_POS = ("NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM")


class CausalPerplexity:
    """Lazy GPT-2 wrapper for genuine causal-LM perplexity."""

    def __init__(self, model_name: str = "gpt2", device: str = "cpu", max_tokens: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self._model = None
        self._tok = None

    def _ensure(self):
        if self._model is None:
            import torch  # noqa
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device).eval()

    def perplexity(self, text: str) -> float:
        import torch
        self._ensure()
        if not text or not text.strip():
            return float("nan")
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=self.max_tokens)
        ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            out = self._model(ids, labels=ids)
        return float(torch.exp(out.loss).item())


def burstiness(text: str) -> float:
    """Coefficient of variation of sentence lengths (a common human-vs-AI signal)."""
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
    except Exception:
        sents = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    lengths = [len(s.split()) for s in sents if s.strip()]
    if len(lengths) < 2:
        return 0.0
    mean = float(np.mean(lengths))
    return float(np.std(lengths) / mean) if mean > 0 else 0.0


def content_pos_ratios(text: str, nlp=None) -> dict:
    """POS ratios for CONTENT classes only (function-word POS excluded by design)."""
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    doc = nlp(text)
    toks = [t for t in doc if not t.is_space]
    n = len(toks) or 1
    counts = {pos: 0 for pos in CONTENT_POS}
    for t in toks:
        if t.pos_ in counts:
            counts[t.pos_] += 1
    return {f"pos_{pos.lower()}": counts[pos] / n for pos in CONTENT_POS}


@dataclass
class StandardFeatureScaler:
    mean_: np.ndarray = None
    std_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "StandardFeatureScaler":
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.where(np.isnan(X), self.mean_, X) - self.mean_) / self.std_
        return Z


def extract_feature_matrix(texts, device: str = "cpu") -> tuple[np.ndarray, list[str]]:
    """Return (standardised feature matrix, feature names) for a list of texts."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    ppl = CausalPerplexity(device=device)
    rows, names = [], None
    for t in texts:
        pos = content_pos_ratios(t, nlp)
        feat = {"perplexity": ppl.perplexity(t), "burstiness": burstiness(t), **pos}
        if names is None:
            names = list(feat.keys())
        rows.append([feat[k] for k in names])
    X = np.array(rows, dtype=float)
    X = StandardFeatureScaler().fit(X).transform(X)
    return X, names
